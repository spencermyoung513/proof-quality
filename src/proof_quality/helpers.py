import random
import numpy as np
import torch
from pathlib import Path
from torch import Tensor
import lightning as L
from lightning.pytorch.callbacks import Callback
from transformers import PreTrainedTokenizerBase
from proof_quality.schema import DecodedPreferenceExample, TokenizedPreferenceBatch


def fix_random_seed(random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


def get_ckpt_callbacks(ckpt_dir: Path) -> list[Callback]:
    return [SaveLoRAAdapter(ckpt_dir)]


class SaveLoRAAdapter(Callback):
    """Save the LoRA adapter and reward head, not the frozen base model.

    `best_val_loss.ckpt` is replaced only when validation loss improves.
    `last.ckpt` is replaced after every validation epoch.
    """

    def __init__(self, ckpt_dir: Path):
        self.ckpt_dir = ckpt_dir
        self.best_val_loss = float("inf")

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        loss = trainer.callback_metrics.get("val/loss")
        if loss is None:
            return
        loss_value = float(loss)
        self._save(pl_module, self.ckpt_dir / "last.ckpt")
        if loss_value < self.best_val_loss:
            self.best_val_loss = loss_value
            self._save(pl_module, self.ckpt_dir / "best_val_loss.ckpt")

    def _save(self, pl_module: L.LightningModule, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(pl_module.adapter_state(), tmp)
        tmp.replace(path)


def decode_batch(
    batch: TokenizedPreferenceBatch,
    tokenizer: PreTrainedTokenizerBase,
    skip_special_tokens: bool = False,
) -> list[DecodedPreferenceExample]:
    """Decode a tokenized preference batch into the raw strings it represents.

    Args:
        batch: A tokenized preference batch (presumably to be passed to the model).
        tokenizer: The tokenizer used to encode the strings in the first place.
        skip_special_tokens: Whether/not to skip special tokens in the decoded strings. Defaults to False.

    Returns:
        A list of decoded preference examples (as strings).
    """
    chosen = _decode_token_rows(
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        tokenizer,
        skip_special_tokens,
    )
    rejected = _decode_token_rows(
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        tokenizer,
        skip_special_tokens,
    )
    return [
        {"chosen": chosen_text, "rejected": rejected_text}
        for chosen_text, rejected_text in zip(chosen, rejected, strict=True)
    ]


def _decode_token_rows(
    input_ids: Tensor,
    attention_mask: Tensor,
    tokenizer: PreTrainedTokenizerBase,
    skip_special_tokens: bool,
) -> list[str]:
    ids = input_ids.detach().cpu()
    mask = attention_mask.detach().cpu().bool()
    return [
        tokenizer.decode(
            row[row_mask].tolist(),
            skip_special_tokens=skip_special_tokens,
        )
        for row, row_mask in zip(ids, mask, strict=True)
    ]


class LogExampleRewards(Callback):
    """Log a fixed set of training examples and their rewards at each epoch start.

    At each epoch, we display and score the same examples (in eval mode) so we can
    we can see how the rewards evolve during training.
    """

    _TABLE_COLUMNS = [
        "epoch",
        "global_step",
        "example",
        "chosen",
        "chosen_reward",
        "rejected",
        "rejected_reward",
        "reward_margin",
    ]

    def __init__(self, num_examples: int = 4):
        self.num_examples = num_examples
        self._batch: TokenizedPreferenceBatch | None = None
        self._examples: list[DecodedPreferenceExample] | None = None
        self._rows: list[list[int | float | str]] = []

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return
        logger = trainer.logger
        log_text = getattr(logger, "log_text", None)
        log_metrics = getattr(logger, "log_metrics", None)
        if log_text is None and log_metrics is None:
            return
        if self._batch is None or self._examples is None:
            self._batch = trainer.datamodule.preview_batch(self.num_examples)
            self._examples = decode_batch(self._batch, trainer.datamodule.tokenizer)

        chosen_rewards = self._score(
            trainer,
            pl_module,
            self._batch["chosen_input_ids"],
            self._batch["chosen_attention_mask"],
        )
        rejected_rewards = self._score(
            trainer,
            pl_module,
            self._batch["rejected_input_ids"],
            self._batch["rejected_attention_mask"],
        )
        metrics: dict[str, float] = {}
        for example_idx, (example, chosen_reward, rejected_reward) in enumerate(
            zip(self._examples, chosen_rewards, rejected_rewards, strict=True)
        ):
            reward_margin = chosen_reward - rejected_reward
            self._rows.append(
                [
                    trainer.current_epoch,
                    trainer.global_step,
                    example_idx,
                    example["chosen"],
                    chosen_reward,
                    example["rejected"],
                    rejected_reward,
                    reward_margin,
                ]
            )
            metrics[f"train/examples/{example_idx}/chosen_reward"] = chosen_reward
            metrics[f"train/examples/{example_idx}/rejected_reward"] = rejected_reward
            metrics[f"train/examples/{example_idx}/reward_margin"] = reward_margin
        if log_metrics is not None:
            log_metrics(metrics, step=trainer.global_step)
        if log_text is not None:
            log_text(
                key="train/examples",
                columns=self._TABLE_COLUMNS,
                data=self._rows,
                step=trainer.global_step,
            )

    def _score(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> list[float]:
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.inference_mode(), trainer.precision_plugin.forward_context():
                rewards = pl_module(
                    input_ids.to(pl_module.device),
                    attention_mask.to(pl_module.device),
                )
                return rewards.detach().float().cpu().tolist()
        finally:
            pl_module.train(was_training)

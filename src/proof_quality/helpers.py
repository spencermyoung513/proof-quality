import random
import numpy as np
import torch
from pathlib import Path
from torch import Tensor
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformers import PreTrainedTokenizerBase
from proof_quality.types import DecodedPreferenceExample, TokenizedPreferenceBatch


def fix_random_seed(random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


def get_ckpt_callbacks(ckpt_dir: Path) -> list[ModelCheckpoint]:
    return [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/loss",
            filename="best_val_loss",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        )
    ]


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

    def __init__(self, num_examples: int = 4):
        self.num_examples = num_examples
        self._batch: TokenizedPreferenceBatch | None = None
        self._examples: list[DecodedPreferenceExample] | None = None

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return
        log_text = getattr(trainer.logger, "log_text", None)
        if log_text is None:
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
        rows = [
            [
                example["chosen"],
                chosen_reward,
                example["rejected"],
                rejected_reward,
                chosen_reward - rejected_reward,
            ]
            for example, chosen_reward, rejected_reward in zip(
                self._examples, chosen_rewards, rejected_rewards, strict=True
            )
        ]
        log_text(
            key="train/examples",
            columns=[
                "chosen",
                "chosen_reward",
                "rejected",
                "rejected_reward",
                "reward_margin",
            ],
            data=rows,
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

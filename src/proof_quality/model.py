import math
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from jaxtyping import Float, Int
from proof_quality.schema import TokenizedPreferenceBatch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def preference_loss(
    chosen_rewards: Float[Tensor, " B"],
    rejected_rewards: Float[Tensor, " B"],
    reward_centering_coeff: float | None,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    """Bradley-Terry loss with an optional reward-centering penalty.

    When `reward_centering_coeff` is set, we add `coeff * mean((r_chosen + r_rejected) ** 2)` to the loss.
    This term pulls each pair toward mean-zero rewards.
    Setting `reward_centering_coeff = None` leaves the Bradley-Terry loss unchanged.

    Returns:
        The loss, pairwise accuracy, and the unweighted centering penalty (to track during training).
    """
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    centering_penalty = torch.mean((chosen_rewards + rejected_rewards) ** 2)
    if reward_centering_coeff is not None:
        loss = loss + reward_centering_coeff * centering_penalty
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss, accuracy, centering_penalty


class RewardModel(L.LightningModule):
    """Reward model to score proof quality. Follows the standard Bradley-Terry preference learning framework.

    Code modified from https://rlhfbook.com/c/05-reward-models#reward-modeling.
    """

    def __init__(
        self,
        base_lm: PreTrainedModel,
        hidden_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        reward_centering_coeff: float | None = None,
        lora_rank: int = 8,
        lora_scaling_factor: int = 8,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_scaling_factor,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
        )
        if getattr(base_lm, "supports_gradient_checkpointing", False):
            base_lm.gradient_checkpointing_enable()
        self.lm = get_peft_model(base_lm, peft_config)
        self.lm.print_trainable_parameters()
        self.head = nn.Linear(hidden_size, 1)
        # from_pretrained leaves the LM in eval mode
        self.train()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.reward_centering_coeff = reward_centering_coeff

    def adapter_state(self) -> dict:
        """LoRA weights, reward head, and the settings needed to rebuild them."""
        lora = self.lm.peft_config["default"]
        return {
            "adapter": dict(get_peft_model_state_dict(self.lm)),
            "reward_head": dict(self.head.state_dict()),
            "lora_rank": lora.r,
            "lora_scaling_factor": lora.lora_alpha,
            "lora_dropout": lora.lora_dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "reward_centering_coeff": self.reward_centering_coeff,
        }

    @classmethod
    def from_checkpoint(
        cls, base_lm: PreTrainedModel, path: Path | str
    ) -> "RewardModel":
        """Load a trained adapter on top of `base_lm`.

        `base_lm` is the same pretrained backbone used in training, still without
        a LoRA adapter. The checkpoint supplies the adapter and reward head.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            base_lm=base_lm,
            hidden_size=base_lm.config.hidden_size,
            learning_rate=ckpt["learning_rate"],
            weight_decay=ckpt["weight_decay"],
            warmup_ratio=ckpt["warmup_ratio"],
            reward_centering_coeff=ckpt.get("reward_centering_coeff"),
            lora_rank=ckpt["lora_rank"],
            lora_scaling_factor=ckpt["lora_scaling_factor"],
            lora_dropout=ckpt["lora_dropout"],
        )
        set_peft_model_state_dict(model.lm, ckpt["adapter"])
        model.head.load_state_dict(ckpt["reward_head"])
        return model

    def _get_last_token_reprs(
        self,
        token_level_reprs: Float[Tensor, "B N D"],
        attention_mask: Float[Tensor, "B N"],
    ) -> Float[Tensor, "B D"]:
        """Extract the representation of the last non-padding token for each sequence.

        Args:
            token_level_reprs: A (B, N, D) tensor with the last-layer token representations for a batch of sequences.
            attention_mask: A (B, N) tensor with 1s for real tokens and 0s for padding.

        Returns:
            A (B, D) tensor with the representation of the last non-padding token for each sequence.
        """
        last_tkn_indices: Int[Tensor, " B"] = attention_mask.sum(dim=-1).long() - 1
        batch_idx: Int[Tensor, " B"] = torch.arange(
            token_level_reprs.size(0),
            device=token_level_reprs.device,
        )
        return token_level_reprs[batch_idx, last_tkn_indices]

    def forward(
        self, input_ids: Int[Tensor, "B N"], attention_mask: Float[Tensor, "B N"]
    ) -> Float[Tensor, " B"]:
        """Compute rewards for the given batch of (tokenized) sequences.

        Args:
            input_ids: A (B, N) tensor of prompt tokens followed by proof tokens.
            attention_mask: A (B, N) tensor with 1s for real tokens and 0s for padding.

        Returns:
            A (B,) tensor with the rewards for each sequence.
        """
        outputs: BaseModelOutput = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        seq_reprs = self._get_last_token_reprs(
            outputs.last_hidden_state, attention_mask
        )
        rewards = torch.squeeze(self.head(seq_reprs), -1)

        return rewards

    def _compute_loss_and_pairwise_acc(
        self, batch: TokenizedPreferenceBatch
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        chosen_rewards = self.forward(
            batch["chosen_input_ids"], batch["chosen_attention_mask"]
        )
        rejected_rewards = self.forward(
            batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )
        return preference_loss(
            chosen_rewards, rejected_rewards, self.reward_centering_coeff
        )

    def training_step(
        self,
        batch: TokenizedPreferenceBatch,
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        loss, acc, reward_centering_penalty = self._compute_loss_and_pairwise_acc(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log("train/acc", acc, on_epoch=True, on_step=False)
        self.log(
            "train/reward_centering_penalty",
            reward_centering_penalty,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(
        self,
        batch: TokenizedPreferenceBatch,
        batch_idx: int,
    ):
        loss, acc, reward_centering_penalty = self._compute_loss_and_pairwise_acc(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True)
        self.log(
            "val/reward_centering_penalty", reward_centering_penalty, on_epoch=True
        )
        return loss

    def test_step(
        self,
        batch: TokenizedPreferenceBatch,
        batch_idx: int,
    ):
        loss, acc, reward_centering_penalty = self._compute_loss_and_pairwise_acc(batch)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)
        self.log(
            "test/reward_centering_penalty", reward_centering_penalty, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

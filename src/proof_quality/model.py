import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from jaxtyping import Float, Int
from proof_quality.types import TokenizedPreferenceBatch
from peft import get_peft_model, LoraConfig, TaskType


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
        self.lm = get_peft_model(base_lm, peft_config)
        self.lm.print_trainable_parameters()
        self.head = nn.Linear(hidden_size, 1)
        # from_pretrained leaves the LM in eval mode
        self.train()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

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

    def training_step(
        self,
        batch: TokenizedPreferenceBatch,
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        chosen_rewards = self.forward(
            batch["chosen_input_ids"], batch["chosen_attention_mask"]
        )
        rejected_rewards = self.forward(
            batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        acc = (chosen_rewards > rejected_rewards).float().mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log("train/acc", acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(
        self,
        batch: TokenizedPreferenceBatch,
        batch_idx: int,
    ):
        chosen_rewards = self.forward(
            batch["chosen_input_ids"], batch["chosen_attention_mask"]
        )
        rejected_rewards = self.forward(
            batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        acc = (chosen_rewards > rejected_rewards).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True)
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

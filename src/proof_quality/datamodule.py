from __future__ import annotations

import datasets
import json
from pathlib import Path
from typing import Iterator
import lightning as L
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from proof_quality.types import TokenizedPreferenceBatch


ROOT = Path(__file__).resolve().parent / "data" / "proofjudge"
SPLITS = ("dev", "test")


class MathlibProofPairsDataset:
    def __init__(self, split: str, root: Path | str = ROOT):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.path = Path(root) / f"{split}.jsonl"

    def _lazily_load_examples(self) -> Iterator[dict]:
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def to_hf_dataset(self) -> datasets.Dataset:
        return datasets.Dataset.from_generator(self._lazily_load_examples)


class ProofPreferenceDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_length: int,
        root: Path | str = ROOT,
    ):
        """Initialize a `ProofPreferenceDataModule`.

        Args:
            tokenizer: Used to tokenize the text before LM consumption.
            batch_size: Number of samples to include in each batch.
            max_length: Maximum length of the tokenized sequences (we pad to this length).
            root: Root directory of the dataset.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.root = root

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train = MathlibProofPairsDataset("dev", self.root).to_hf_dataset()
        if stage in ("fit", "validate", None):
            self.val = MathlibProofPairsDataset("test", self.root).to_hf_dataset()
        if stage == "test":
            raise NotImplementedError(
                f"Setup for stage '{stage!r}' is not implemented yet"
            )

    def _tokenize_conversations(self, conversations: list[list[dict[str, str]]]):
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_dict=True,
        )

    def _collate_fn(self, rows: list[dict]) -> TokenizedPreferenceBatch:
        chosen = self._tokenize_conversations(
            [
                [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["chosen"]},
                ]
                for row in rows
            ]
        )
        rejected = self._tokenize_conversations(
            [
                [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["rejected"]},
                ]
                for row in rows
            ]
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

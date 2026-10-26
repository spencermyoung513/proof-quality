from __future__ import annotations

import datasets
import json
from pathlib import Path
from typing import Iterator
import lightning as L
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from proof_quality.schema import TokenizedPreferenceBatch


ROOT = Path(__file__).resolve().parents[2] / "data" / "mathlibpr"
SPLITS = ("dev", "test")


def _messages(prompt: str, completion: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def _tokenized_length(
    tokenizer: PreTrainedTokenizerBase, prompt: str, completion: str
) -> int:
    """Token count for one chat, using the same template as batch collation."""
    encoded = tokenizer.apply_chat_template(
        _messages(prompt, completion),
        tokenize=True,
        add_generation_prompt=False,
        padding=False,
        truncation=False,
        return_dict=True,
    )
    return len(encoded["input_ids"])


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
            max_length: Maximum tokenized length. Pairs with a longer chosen or rejected sequence are dropped; remaining batches are truncated to this length.
            root: Root directory of the dataset.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.root = root

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            train = MathlibProofPairsDataset("dev", self.root).to_hf_dataset()
            self.train = self._drop_truncated(train, "dev")
        if stage in ("fit", "validate", None):
            val = MathlibProofPairsDataset("test", self.root).to_hf_dataset()
            self.val = self._drop_truncated(val, "test")
        if stage == "test":
            raise NotImplementedError(
                f"Setup for stage '{stage!r}' is not implemented yet"
            )

    def _drop_truncated(
        self, dataset: datasets.Dataset, split: str
    ) -> datasets.Dataset:
        """Drop pairs whose chosen or rejected sequence would be truncated."""
        keep = [
            i
            for i, row in enumerate(dataset)
            if _tokenized_length(self.tokenizer, row["prompt"], row["chosen"])
            <= self.max_length
            and _tokenized_length(self.tokenizer, row["prompt"], row["rejected"])
            <= self.max_length
        ]
        dropped = len(dataset) - len(keep)
        if not keep:
            raise ValueError(
                f"All {len(dataset)} {split} samples exceed max_length={self.max_length}."
            )
        print(
            f"Dropped {dropped}/{len(dataset)} {split} samples longer than "
            f"max_length={self.max_length}."
        )
        return dataset.select(keep)

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
            [_messages(row["prompt"], row["chosen"]) for row in rows]
        )
        rejected = self._tokenize_conversations(
            [_messages(row["prompt"], row["rejected"]) for row in rows]
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    def preview_batch(self, num_examples: int) -> TokenizedPreferenceBatch:
        """Load a preview batch from the training split.

        Args:
            num_examples: Number of examples to load.

        Returns:
            A preview batch of (tokenized) examples.
        """
        loader = DataLoader(
            dataset=self.train,
            batch_size=num_examples,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
        return next(iter(loader))

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

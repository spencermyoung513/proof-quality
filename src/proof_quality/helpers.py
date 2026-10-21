import random
import numpy as np
import torch
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint


def fix_random_seed(random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


def get_ckpt_callbacks(ckpt_dir: Path) -> list[ModelCheckpoint]:
    return [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/ce-loss",
            filename="best_val_loss",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        )
    ]

"""Build theorem-proof preference pairs from MathlibPR snapshots."""

from proof_quality.dataset_generation.extract import (
    extract_pair,
    main,
    parse_hunks,
    parse_name_status,
)

__all__ = [
    "extract_pair",
    "main",
    "parse_hunks",
    "parse_name_status",
]

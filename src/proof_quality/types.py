from typing import TypedDict
from torch import Tensor
from jaxtyping import Int, Float


class TokenizedPreferenceBatch(TypedDict):
    chosen_input_ids: Int[Tensor, "B N"]
    chosen_attention_mask: Float[Tensor, "B N"]
    rejected_input_ids: Int[Tensor, "B N"]
    rejected_attention_mask: Float[Tensor, "B N"]

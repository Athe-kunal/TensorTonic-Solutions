import torch

def shared_expert(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """
    Returns shared-expert outputs with the same shape as x.
    """
    gate_project = torch.nn.functional.silu(x @ W_gate.T)   # (..., d_ff)
    up_project = x @ W_up.T                                  # (..., d_ff)
    return (gate_project * up_project) @ W_down.T            # (..., d_model)
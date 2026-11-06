import torch

def moe_router(x: torch.Tensor, expert_centroids: torch.Tensor, top_k: int) -> dict:
    """
    Returns selected expert indices and normalized weights in a dictionary.
    """
    # x: [batch_size, seq_len, d_model] @ expert_centroids [n_experts, d_model]
    # router_affinity: [batch_size, seq_len, n_experts]
    router_affinity = torch.sigmoid(x @ expert_centroids.T)
    topk_res = torch.topk(router_affinity, k=top_k, dim=-1)
    # [batch_size, seq_len, top_k], [batch_size, seq_len, top_k]
    topk_indices, topk_values = topk_res.indices, topk_res.values
    return {"indices":topk_indices,"weights":topk_values / topk_values.sum(dim=-1,keepdim=True)}
    
    
import torch as th

class K2A(th.nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, symmetrize=True, verbose=False):
        """ 
        KernelAttentionAggregator (K2A) aggregates multiple kernel matrices using self-attention.
        Args:
            embed_dim (int): Dimensionality of projected kernel embeddings.
            num_heads (int): Number of attention heads.
            symmetrize (bool): If True, symmetrize the final kernel output.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.symmetrize = symmetrize

        # Projects each scalar kernel entry into a d_model-dim embedding
        self.kernel_proj = th.nn.Linear(1, embed_dim)  # applied elementwise
        self.attn = th.nn.MultiheadAttention(embed_dim, num_heads) # , batch_first=True -> TODO: whats this?
        self.attn_score_proj = th.nn.Linear(embed_dim, 1)

        if verbose: print(f"K2A initialized with {embed_dim} embed_dim, {num_heads} heads, {'symmetrized' if symmetrize else ''}")

    def forward(self, kernel_tensor, return_weights=False):
        """
        Args:
            kernel_tensor (Tensor): shape (m, n, n) — m kernels for n datapoints
            return_weights (bool): if True, also return attention weights over patches

        Returns:
            Tensor of shape (a, b): aggregated kernel matrix
            Optional Tensor of shape (p,): attention weights over patches
        """
        p = kernel_tensor.shape[0]  # number of patches

        # Step 1: Embed each kernel (flattened and embedded element-wise)
        x = kernel_tensor.view(p, -1, 1)         # (p, a*b, 1)
        x = self.kernel_proj(x).transpose(0, 1)  # (a*b, p, d_model)

        # Step 2: Self-attention across patch-wise kernels
        attn_out, _ = self.attn(x, x, x)         # (a*b, p, d_model)

        # Step 3: Get attention scores for each patch
        scores = self.attn_score_proj(attn_out).mean(dim=0).squeeze(-1)  # (p,)
        weights = th.nn.functional.softmax(scores, dim=0)  # attention weights over m patches

        # Step 4: Compute weighted sum of kernels
        weighted_kernels = weights.view(p, 1, 1) * kernel_tensor  # (m, n, n)
        K_agg = weighted_kernels.sum(dim=0)                       # (n, n)

        # Step 5: Optional symmetrization
        if self.symmetrize and K_agg.shape[0] == K_agg.shape[1]: K_agg = 0.5 * (K_agg + K_agg.T)

        if return_weights: return K_agg, weights
        return K_agg

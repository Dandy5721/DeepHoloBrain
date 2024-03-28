"""The scattering-transform Mixer layer (Sec. 3.1-3.2, Eq. 2-3, Fig. 1-3).

`ScatteringMixerLayer` builds the block matrix of graph harmonic wavelets P
from the structural-connectivity graph, applies the learnable frequency
scaling g_gamma(lambda) = exp(-gamma * lambda) of Eq. 2, weights it with
node-specific ("location") and frequency-specific attention (visualized in
Fig. 4/5), forms the Supra-FC matrix X = P^T X P, and max-pools it back
down to an N x N SPD matrix (Eq. 3 / Prop. 3.6).
"""

import torch
from torch import nn

from ..utils import fc2vector
from .wavelet import graph_harmonic_basis


def _trace_normalize(x, eps=1e-12):
    """Rescale a batch of SPD matrices to a fixed average eigenvalue (trace/N)."""
    trace = x.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    return x / (trace / x.shape[-1] + eps)


class ScatteringMixerLayer(nn.Module):
    """Row/column graph-scattering mapping P^T X P with location/frequency attention.

    Args:
        num_nodes: number of brain regions N (AAL116 parcellation).

    Forward args:
        input: (N, N) or (B, N, N) FC matrix X.
        sc: (N, N) or (B, N, N) structural-connectivity matrix the harmonic
            wavelet basis is derived from.
        attention_mode: 0 -> output is vectorized (lower triangle) for the
            plain-MLP classifier head (`DeepHoloBrain.classifier2`, the
            "no manifold layers" ablation). 1 -> output stays an SPD matrix
            for the manifold layers (`DeepHoloBrain.layers` + `classifier1`,
            Fig. 1/3's main path). The attention maps returned for Fig. 4/5
            are always softmax-normalized, independent of `attention_mode`.

    Returns `(pooled_output, scale, attention_rows, attention_columns, fc_weight)`.
    `fc_weight` in [0, 1] is how much of the output is the raw FC term versus
    the SC-conditioned scattering term (see below) -- report it alongside
    accuracy: a model that pins it near 1 is not using SC.
    """

    def __init__(self, num_nodes=116):
        super(ScatteringMixerLayer, self).__init__()
        self.num_nodes = num_nodes
        self.scale = nn.Parameter(torch.randn(1))
        self.attention_row = nn.Parameter(torch.randn(1, 1, num_nodes * num_nodes))
        self.attention_column = nn.Parameter(torch.randn(1, num_nodes, 1))
        self.pool = nn.MaxPool2d(kernel_size=num_nodes, stride=num_nodes)
        # Learnable FC/scattering mix, sigmoid(0) = 0.5: unbiased at init.
        self.mix_logit = nn.Parameter(torch.zeros(1))

    def forward(self, input, sc, attention_mode):
        harmonic_basis = graph_harmonic_basis(sc)
        # Normalize to a fixed O(1) dynamic range before the exponential
        # gate: raw harmonic_basis entries are tiny (eigenvalue times a
        # unit-eigenvector outer product), so exp(-gamma*basis) otherwise
        # sits within ~1e-2 of the constant matrix 1 for any `gamma` SGD
        # reaches, making `weighted_matrix` dominated by the location-
        # independent attention parameters rather than the SC-derived
        # spectrum.
        harmonic_basis = harmonic_basis / (
            harmonic_basis.detach().abs().amax(dim=(1, 2), keepdim=True) + 1e-8
        )
        scaled_wavelet = torch.exp(-self.scale * harmonic_basis)

        attention_rows = torch.softmax(self.attention_row, dim=2)
        attention_columns = torch.softmax(self.attention_column, dim=1)

        if attention_mode == 0:
            weighted_matrix = scaled_wavelet * self.attention_row * self.attention_column
            attention_rows = attention_rows.squeeze(0)
            attention_columns = attention_columns.squeeze(0)
        else:
            attention_rows = attention_rows.view(1, 1, -1)
            weighted_matrix = scaled_wavelet * attention_rows * attention_columns
            attention_rows = attention_rows.squeeze(0)
            attention_columns = attention_columns.squeeze(0)

        output = weighted_matrix.transpose(-2, -1) @ input @ weighted_matrix
        if output.dim() == 2:
            output = output.unsqueeze(0)
        scatter_term = self.pool(output)

        # Combine the SC-conditioned scattering term with the raw FC matrix.
        # `weighted_matrix` is built from two softmax-normalized attention
        # tensors, so `scatter_term`'s natural magnitude is orders below
        # FC's -- summing them raw lets FC dominate regardless of whether
        # SC is informative. Trace-normalizing both to the same average
        # eigenvalue first, then combining with a learnable weight (rather
        # than a fixed residual), gives the SC branch a fair, equal-footing
        # chance to matter; how much it ends up contributing is then a
        # genuine, checkable property of the trained model (see
        # `fc_weight` above) rather than an artifact of this layer's
        # arithmetic. Both terms are SPD, so the convex combination is too.
        fc_term = input if input.dim() == 3 else input.unsqueeze(0)
        fc_weight = torch.sigmoid(self.mix_logit)
        pooled_output = fc_weight * _trace_normalize(fc_term) + (1 - fc_weight) * _trace_normalize(scatter_term)

        if attention_mode == 0:
            pooled_output = fc2vector(pooled_output)

        return pooled_output, self.scale, attention_rows, attention_columns, fc_weight

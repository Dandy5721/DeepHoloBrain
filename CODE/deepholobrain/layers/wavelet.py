"""Graph harmonic wavelet basis construction (Sec. 2.3, background to Eq. 2).

Symmetric-normalized graph Laplacian L = I - D^{-1/2} W D^{-1/2} of the
structural-connectivity graph, eigendecomposed into the harmonic (Fourier)
basis {u_k} with eigenvalues {lambda_k}. `graph_harmonic_basis` returns,
per frequency index k, the rank-1 contribution lambda_k * u_k u_k^T;
`ScatteringMixerLayer` (scattering.py) turns this into the frequency-scaled
spectral filter g_gamma(lambda) = exp(-gamma * lambda) of Eq. 2.
"""

import torch

from .riemannian import safe_eigh


def _symmetric_normalized_laplacian(sc):
    """Symmetric-normalized graph Laplacian of a (B, N, N) structural-connectivity graph.

    Disconnected regions (`degree == 0`) are handled by substituting a safe
    placeholder degree before the `1/sqrt(degree)` division, rather than
    masking the result afterward -- avoids ever forming an `inf` that would
    otherwise poison the backward pass with `0 * inf = NaN` even though the
    forward value is masked out correctly.
    """
    node_num = sc.shape[-1]
    degree = sc.sum(dim=2)
    has_degree = degree > 0
    safe_degree = torch.where(has_degree, degree, torch.ones_like(degree))
    inv_sqrt_degree = torch.where(has_degree, 1.0 / torch.sqrt(safe_degree), torch.zeros_like(degree))

    laplacian = -sc.clone()
    laplacian = inv_sqrt_degree.unsqueeze(2) * laplacian * inv_sqrt_degree.unsqueeze(1)
    diag_idx = torch.arange(node_num, device=sc.device)
    laplacian[:, diag_idx, diag_idx] = 1.0
    return (laplacian + laplacian.transpose(1, 2)) / 2


def graph_harmonic_basis(sc, eigh_eps=1e-4):
    """Per-frequency harmonic basis of a structural-connectivity graph (Sec. 2.3).

    For each of the N frequency indices k (ascending eigenvalue order, sign
    of each eigenvector fixed so its first entry is non-negative), returns
    the rank-1 matrix lambda_k * u_k u_k^T, where {u_k, lambda_k} are the
    eigenpairs of the symmetric-normalized graph Laplacian of `sc`. Uses
    `safe_eigh` since real structural-connectivity Laplacians routinely
    have near-degenerate eigenvalues (e.g. bilateral hemisphere symmetry).

    Args:
        sc: (N, N) or (B, N, N) structural connectivity / adjacency matrix.

    Returns:
        (B, N, N * N) tensor: N rank-1 N x N matrices (flattened) per
        batch element, one per frequency.
    """
    if sc.dim() == 2:
        sc = sc.unsqueeze(0)

    sym_laplacian = _symmetric_normalized_laplacian(sc)
    eigvals, eigvecs = safe_eigh(sym_laplacian, eigh_eps)  # eigh: ascending order
    eigvals = torch.clamp(eigvals, min=0)

    signs = torch.sign(eigvecs[:, 0, :])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    eigvecs = eigvecs * signs.unsqueeze(1)

    outer = torch.einsum('bik,bjk->bkij', eigvecs, eigvecs)
    basis = eigvals.unsqueeze(-1).unsqueeze(-1) * outer
    return basis.reshape(basis.size(0), basis.size(1), -1)

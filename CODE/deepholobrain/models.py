"""Models evaluated in the paper: the SPDNet baseline (Sec. 4, Huang & Van Gool 2017)
and DeepHoloBrain, the proposed scattering-transform Mixer (Sec. 3.3, Fig. 1/3).
"""

import torch
from torch import nn

from .layers.scattering import ScatteringMixerLayer
from .layers.spd_layers import Normalize, SPDRectified, SPDTangentSpace, SPDTransform


class SPDNet(nn.Module):
    """SPDNet baseline (Huang & Van Gool, 2017): a stack of Eq. 1 positive mappings.

    `spd` selects the network depth (0-4 use progressively deeper/shallower
    layer stacks; anything else falls back to `layers`/`classifier`),
    matching the `--spd` ablation flag used across the training scripts.
    """

    def __init__(self, num_classes):
        super(SPDNet, self).__init__()
        self.layers = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier = nn.Sequential(nn.Linear(16 * 17 // 2, num_classes))

        self.layers0 = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier0 = nn.Sequential(nn.Linear(32 * 33 // 2, num_classes))

        self.layers1 = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier1 = nn.Sequential(nn.Linear(64 * 65 // 2, num_classes))

        self.layers2 = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDRectified(),
            SPDTransform(16, 8, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier2 = nn.Sequential(nn.Linear(8 * 9 // 2, num_classes))

        self.layers3 = nn.Sequential(
            SPDTransform(116, 16, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier3 = nn.Sequential(nn.Linear(16 * 17 // 2, num_classes))

        self.layers4 = nn.Sequential(
            SPDTransform(116, 32, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier4 = nn.Sequential(nn.Linear(32 * 33 // 2, num_classes))

    def forward(self, x, spd):
        layers, classifier = {
            0: (self.layers0, self.classifier0),
            1: (self.layers1, self.classifier1),
            2: (self.layers2, self.classifier2),
            3: (self.layers3, self.classifier3),
            4: (self.layers4, self.classifier4),
        }.get(spd, (self.layers, self.classifier))
        x = layers(x)
        out = classifier(x)
        return x, out


class DeepHoloBrain(nn.Module):
    """DeepHoloBrain (Sec. 3.3): scattering-transform Mixer for the SPD manifold.

    Forward computes the Supra-FC pooling of Eq. 2-3 via `ScatteringMixerLayer`
    (using `sc`, the structural-connectivity graph, to derive the harmonic
    wavelets), then either:
      - `spd == 1`: feeds the pooled SPD matrix through a small SPDNet-style
        manifold stack (`self.layers`) before `classifier1` (the main path,
        Fig. 1/3), or
      - `spd == 0`: feeds the pooled/vectorized Supra-FC features directly
        to the plain MLP `classifier2` (the "no manifold layers" ablation).

    Returns `(scale, out, attention_rows, attention_columns, fc_weight)`:
    `scale` is the learnable frequency-scaling gamma of Eq. 2 (regularized
    via Eq. 4, see `scripts/common.py`); `attention_rows`/`attention_columns`
    are the node-specific / frequency-specific attention maps visualized in
    Fig. 4/5; `fc_weight` (see `ScatteringMixerLayer`) is how much of the
    pooled representation is the raw FC term versus the SC-conditioned
    scattering term -- worth reporting alongside accuracy, since a model
    that pins it near 1 is not actually using SC.
    """

    def __init__(self, num_classes, num_nodes=116):
        super(DeepHoloBrain, self).__init__()
        self.layers = nn.Sequential(
            SPDTransform(num_nodes, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDTangentSpace(vectorize_all=False),
            Normalize(),
        )
        self.weighted_scaled = ScatteringMixerLayer(num_nodes=num_nodes)
        self.classifier1 = nn.Sequential(nn.Linear(16 * 17 // 2, num_classes))
        self.classifier2 = nn.Sequential(
            nn.Linear(num_nodes * (num_nodes - 1) // 2, 1280),
            nn.Linear(1280, 640),
            nn.Linear(640, 320),
            nn.Linear(320, 16 * 17 // 2),
            nn.Linear(16 * 17 // 2, num_classes),
        )

    def forward(self, x, sc, spd):
        # `weighted_scaled`'s output is a convex combination of two
        # trace-normalized SPD terms (see `ScatteringMixerLayer`), so it
        # already has average eigenvalue 1 -- unlike a raw scattering-only
        # output, whose eigenvalues would otherwise sit below
        # `SPDRectified`'s epsilon floor and collapse `self.layers` to a
        # constant with zero gradient.
        x, scale, attention_rows, attention_columns, fc_weight = self.weighted_scaled(x, sc, spd)

        if spd == 1:
            x = self.layers(x)
            out = self.classifier1(x)
        else:
            out = self.classifier2(x)

        return scale, out, attention_rows, attention_columns, fc_weight

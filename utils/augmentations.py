from enum import IntEnum

import einops as ein
import numpy as np
import numpy.random as npr


class NormalizeMCS:
    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def __call__(self, x):
        self.mins = self.mins
        self.maxs = self.maxs

        x = ein.rearrange(x, "c n l -> n l c")
        x = (x - self.mins) / (self.maxs - self.mins + 1e-6)
        x = ein.rearrange(x, "n l c -> c n l")

        return x


class SpectralCutout:
    def __init__(self, p_apply, num_mask=1, max_size=20):
        self.p_apply = p_apply
        self.num_mask = num_mask
        self.max_size = max_size

    def __call__(self, spec):
        for _ in range(self.num_mask):
            mask = self._make_mask(spec.shape)
            spec = self._apply_mask(spec, mask)
        return spec

    def _make_mask(self, size):
        mask = np.ones(size)
        if npr.uniform(0, 1) > self.p_apply:
            return mask
        start = npr.randint(0, size, size=(3,))
        sizes = npr.randint(0, self.max_size, size=(3,))
        end = start + sizes
        for i in range(3):
            end[i] = min(end[i], size[i])
        for i in range(3):
            mask[start[i] : end[i]] = 0
        return mask

    def _apply_mask(self, spec, mask):
        mask = mask
        return spec * mask


class SpectralAxialCutout:
    class CutoutType(IntEnum):
        CHANNEL = 0  # 10
        FREQUENCY = 1  # 11
        TIME = 2  # 7

    def __init__(self, p_apply, dim_to_cut: CutoutType, max_num_cut: int = 1):
        self.p_apply = p_apply
        self.dim_to_cut = dim_to_cut
        self.max_num_cut = max_num_cut

    def __call__(self, spec):
        if npr.uniform(0, 1) > self.p_apply:
            return spec
        n_cut = npr.randint(1, self.max_num_cut + 1)
        channel_cut = npr.choice(
            np.arange(spec.shape[int(self.dim_to_cut)]), size=n_cut, replace=False
        )
        idx = [slice(None)] * len(spec.shape)
        idx[int(self.dim_to_cut)] = channel_cut
        spec[idx] = 0
        return spec


class SpectralNoise:
    def __init__(self, p_apply, noise_level=0.1):
        self.p_apply = p_apply
        self.noise_level = noise_level

    def __call__(self, spec):
        if npr.uniform(0, 1) > self.p_apply:
            return spec
        noise = npr.normal(0, self.noise_level, spec.shape)
        return spec + noise

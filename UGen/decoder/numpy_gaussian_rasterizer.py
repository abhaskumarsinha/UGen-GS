from dataclasses import dataclass
from typing import Optional, Tuple
from UGen.decoder.base import *

@dataclass
class GaussianRGBRendererConfig:
    width: int = 160
    height: int = 120
    fx: float = 120.0
    fy: float = 120.0
    cx: Optional[float] = None
    cy: Optional[float] = None
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    composite_mode: str = "alpha"
    max_radius_px: Optional[int] = None
    sigma_clip: float = 3.0
    eps_cov: float = 1e-6



import numpy as np

class GaussianRGBRenderer(BaseRasterizer):
    def __init__(self, config: GaussianRendererConfig):
        self.cfg = config
    
    def render(self, gaussians):

        width = self.cfg.width
        height = self.cfg.height
        fx = self.cfg.fx
        fy = self.cfg.fy
        cx = self.cfg.cx
        cy = self.cfg.cy
        background = self.cfg.background
        composite_mode = self.cfg.composite_mode
        max_radius_px = self.cfg.max_radius_px
        sigma_clip = self.cfg.sigma_clip
        eps_cov = self.cfg.eps_cov

        if cx is None:
            cx = width / 2.0
        if cy is None:
            cy = height / 2.0

        img = np.zeros((height, width, 3), dtype=np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)

        gauss_list = [g for g in gaussians if g is not None]
        gauss_list = [g for g in gauss_list if g.mean[2] > 1e-9]
        gauss_list.sort(key=lambda g: g.mean[2])

        for g in gauss_list:

            X, Y, Z = float(g.mean[0]), float(g.mean[1]), float(g.mean[2])
            if Z <= 0:
                continue

            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy

            J = np.array([
                [fx / Z, 0.0, -fx * X / (Z * Z)],
                [0.0, fy / Z, -fy * Y / (Z * Z)]
            ], dtype=np.float64)

            cov3 = np.asarray(g.cov, dtype=np.float64).reshape(3,3)
            cov2 = J @ cov3 @ J.T
            cov2 += np.eye(2) * eps_cov

            eigvals, eigvecs = np.linalg.eigh(cov2)
            eigvals = np.maximum(eigvals, eps_cov)
            sigmas = np.sqrt(eigvals)
            max_sigma = float(np.max(sigmas))

            radius = int(np.ceil(sigma_clip * max_sigma))
            if max_radius_px is not None:
                radius = min(radius, int(max_radius_px))

            if radius <= 0:
                iu = int(round(u))
                iv = int(round(v))
                if 0 <= iu < width and 0 <= iv < height:
                    src_a = float(np.clip(g.opacity, 0.0, 1.0))
                    src_c = np.clip(np.asarray(g.color, dtype=np.float32), 0.0, 1.0)
                    out_a = src_a + alpha[iv, iu] * (1.0 - src_a)
                    if out_a > 0:
                        img[iv, iu, :] = (
                            src_c * src_a +
                            img[iv, iu, :] * alpha[iv, iu] * (1.0 - src_a)
                        ) / out_a
                        alpha[iv, iu] = out_a
                continue

            x0 = max(0, int(np.floor(u - radius)))
            x1 = min(width - 1, int(np.ceil(u + radius)))
            y0 = max(0, int(np.floor(v - radius)))
            y1 = min(height - 1, int(np.ceil(v + radius)))

            if x1 < 0 or x0 >= width or y1 < 0 or y0 >= height:
                continue

            xs = np.arange(x0, x1 + 1, dtype=np.float64)
            ys = np.arange(y0, y1 + 1, dtype=np.float64)
            xv, yv = np.meshgrid(xs, ys)

            du = (xv - u).astype(np.float64)
            dv = (yv - v).astype(np.float64)
            d = np.stack([du.ravel(), dv.ravel()], axis=0)

            try:
                inv_cov2 = np.linalg.inv(cov2)
            except np.linalg.LinAlgError:
                inv_cov2 = np.linalg.pinv(cov2)

            tmp = inv_cov2 @ d
            m = np.sum(d * tmp, axis=0)

            local_alpha = (
                np.clip(g.opacity, 0.0, 1.0) *
                np.exp(-0.5 * m)
            ).astype(np.float32)

            local_alpha = local_alpha.reshape(yv.shape)

            src_c = np.clip(
                np.asarray(g.color, dtype=np.float32).reshape((1,1,3)),
                0.0, 1.0
            )

            dst_rgb = img[y0:y1+1, x0:x1+1, :]
            dst_a = alpha[y0:y1+1, x0:x1+1]

            a = local_alpha[..., None]
            out_a = a[...,0] + dst_a * (1.0 - a[...,0])

            out_rgb = np.where(
                out_a[...,None] > 0,
                (src_c * a + dst_rgb * dst_a[...,None] * (1.0 - a)) /
                np.maximum(out_a[...,None], 1e-12),
                0.0
            )

            img[y0:y1+1, x0:x1+1, :] = out_rgb
            alpha[y0:y1+1, x0:x1+1] = out_a

        bg = np.asarray(background, dtype=np.float32).reshape((1,1,3))
        final = img * alpha[...,None] + bg * (1.0 - alpha[...,None])
        final = np.clip(final, 0.0, 1.0)

        return final.astype(np.float32)

import numpy as np
from PIL import Image
import math
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from UGen.encoder.algorithms.BaseAlgorithm import *



# ------------------------------------------------------------
# Configuration Dataclass
# ------------------------------------------------------------

@dataclass
class GreedyGaussianConfig:
    n_splats: int = 150
    sigma_values: tuple = (1, 2, 4, 8, 16)
    angle_steps: int = 8
    alphas: tuple = (0.3, 0.6, 1.0)
    verbose: bool = True


# ------------------------------------------------------------
# Greedy Gaussian Encoder
# ------------------------------------------------------------

class GreedyGaussianEncoder(EncoderAlgorithms):

    def __init__(self, image, config: GreedyGaussianConfig):
        self.I = image.astype(np.float32)
        self.config = config
        self.C_canvas = np.zeros_like(self.I, dtype=np.float32)
        self.splats = []

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    def clamp01(self, x):
        return np.clip(x, 0.0, 1.0)

    def load_image(self, path, size=None):
        img = Image.open(path).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr

    def save_image(self, arr, path):
        img = Image.fromarray((self.clamp01(arr) * 255).astype(np.uint8))
        img.save(path)

    # ------------------------------------------------------------
    # Anisotropic Gaussian Kernel
    # ------------------------------------------------------------

    def anisotropic_gaussian(self, sigma_x, sigma_y, theta):
        """
        Returns kernel (peak=1) and radius
        """
        max_sigma = max(sigma_x, sigma_y)
        radius = int(math.ceil(3 * max_sigma))

        xs = np.arange(-radius, radius+1)
        xx, yy = np.meshgrid(xs, xs)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x_rot =  cos_t * xx + sin_t * yy
        y_rot = -sin_t * xx + cos_t * yy

        g = np.exp(
            -0.5 * (
                (x_rot**2) / (sigma_x**2) +
                (y_rot**2) / (sigma_y**2)
            )
        )

        return g.astype(np.float32), radius

    # ------------------------------------------------------------
    # Greedy Fitting
    # ------------------------------------------------------------

    def fit_gaussians(self):

        I = self.I
        H, W, _ = I.shape
        C_canvas = self.C_canvas

        splats = []

        angles = np.linspace(0, np.pi, self.config.angle_steps, endpoint=False)

        for k in range(self.config.n_splats):

            best_score = -1.0
            best_candidate = None

            R = I - C_canvas

            for sigma_x in self.config.sigma_values:
                for sigma_y in self.config.sigma_values:

                    stride = max(1, int(min(sigma_x, sigma_y)))

                    for theta in angles:

                        kernel, radius = self.anisotropic_gaussian(sigma_x, sigma_y, theta)
                        ksz = kernel.shape[0]
                        kernel_sq_sum = (kernel * kernel).sum()

                        for yc in range(radius, H-radius, stride):
                            for xc in range(radius, W-radius, stride):

                                ys = slice(yc-radius, yc+radius+1)
                                xs = slice(xc-radius, xc+radius+1)

                                C_patch = C_canvas[ys, xs]
                                I_patch = I[ys, xs]

                                for alpha in self.config.alphas:

                                    A = alpha * kernel
                                    A3 = A[..., None]

                                    T = I_patch - (1.0 - A3) * C_patch

                                    num = (kernel[..., None] * T).reshape(-1,3).sum(axis=0)
                                    den = kernel_sq_sum

                                    if den < 1e-8:
                                        continue

                                    score = float((num @ num) / den)

                                    if score > best_score:
                                        best_score = score
                                        best_candidate = {
                                            'yc': yc,
                                            'xc': xc,
                                            'sigma_x': sigma_x,
                                            'sigma_y': sigma_y,
                                            'theta': theta,
                                            'alpha': alpha,
                                            'kernel': kernel,
                                            'radius': radius,
                                            'p': num / den
                                        }

            if best_candidate is None:
                print("Stopped early.")
                break

            yc = best_candidate['yc']
            xc = best_candidate['xc']
            kernel = best_candidate['kernel']
            radius = best_candidate['radius']
            alpha = best_candidate['alpha']
            p = best_candidate['p']

            ys = slice(yc-radius, yc+radius+1)
            xs = slice(xc-radius, xc+radius+1)

            C_patch = C_canvas[ys, xs]
            A = alpha * kernel
            A3 = A[..., None]

            C_canvas[ys, xs] = (
                p[None,None,:] * kernel[...,None]
                + (1.0 - A3) * C_patch
            )

            color = p / alpha if alpha > 0 else np.zeros(3)

            splats.append({
                "center": (yc, xc),
                "sigma_x": best_candidate['sigma_x'],
                "sigma_y": best_candidate['sigma_y'],
                "theta": best_candidate['theta'],
                "alpha": alpha,
                "color": tuple(self.clamp01(color))
            })

            if self.config.verbose:
                residual_energy = float(((I - C_canvas)**2).sum())
                print(f"Splat {k+1}/{self.config.n_splats} | Energy: {residual_energy:.4f}")

        self.C_canvas = C_canvas
        self.splats = splats

        return self.clamp01(C_canvas), splats

    # ------------------------------------------------------------
    # Required Base Method
    # ------------------------------------------------------------

    def render(self):
        """
        Required by base class.
        Runs fitting and returns rendered image.
        """
        rendered, _ = self.fit_gaussians()
        return rendered

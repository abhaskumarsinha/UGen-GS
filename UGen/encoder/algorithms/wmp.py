import numpy as np
from PIL import Image
import math
import os
import random
from dataclasses import dataclass
from UGen.encoder.algorithms.BaseAlgorithm import *



# ------------------------------------------------------------
# Configuration Dataclass
# ------------------------------------------------------------

@dataclass
class WeakMatchingPursuitConfig:
    n_iterations: int
    alpha: float = 0.8
    n_candidates: int = 120
    sigma_choices: tuple = (2,4,8,16)
    angle_steps: int = 8
    alpha_choices: tuple = (1.0,)
    random_seed: int = None
    verbose: bool = True


# ------------------------------------------------------------
# Weak Matching Pursuit Encoder
# ------------------------------------------------------------

class WeakMatchingPursuitEncoder(EncoderAlgorithms):

    def __init__(self, image: np.ndarray, config: WeakMatchingPursuitConfig):
        self.image = image.astype(np.float32)
        self.config = config
        self.selected_atoms = []
        self.residual = None
        self.reconstructed = None

    # -------------------------
    # Utilities
    # -------------------------

    def clamp01(self, x):
        return np.clip(x, 0.0, 1.0)

    def load_image_rgb(self, path, size=None):
        im = Image.open(path).convert("RGB")
        if size is not None:
            im = im.resize(size, Image.LANCZOS)
        arr = np.asarray(im).astype(np.float32) / 255.0
        return arr

    def save_image_rgb(self, arr, path):
        img = Image.fromarray((self.clamp01(arr) * 255.0).astype(np.uint8))
        img.save(path)

    # -------------------------
    # Kernel
    # -------------------------

    def anisotropic_kernel(self, sigma_x, sigma_y, theta, radius=None):
        max_sigma = max(sigma_x, sigma_y)
        if radius is None:
            radius = int(math.ceil(3.0 * max_sigma))
        xs = np.arange(-radius, radius+1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, xs)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_rot = cos_t * xx + sin_t * yy
        y_rot = -sin_t * xx + cos_t * yy
        g = np.exp(-0.5 * ((x_rot**2) / (sigma_x**2) + (y_rot**2) / (sigma_y**2)))
        return g.astype(np.float32), radius

    def place_kernel_full(self, kernel, center_y, center_x, H, W):
        ksz = kernel.shape[0]
        radius = (ksz - 1) // 2
        phi = np.zeros((H, W), dtype=np.float32)

        y0 = center_y - radius
        x0 = center_x - radius
        y1 = y0 + ksz
        x1 = x0 + ksz

        ky0 = 0; kx0 = 0
        ky1 = ksz; kx1 = ksz

        if y0 < 0:
            ky0 = -y0; y0 = 0
        if x0 < 0:
            kx0 = -x0; x0 = 0
        if y1 > H:
            ky1 = ksz - (y1 - H); y1 = H
        if x1 > W:
            kx1 = ksz - (x1 - W); x1 = W

        phi[y0:y1, x0:x1] = kernel[ky0:ky1, kx0:kx1]
        return phi

    # -------------------------
    # Candidate Sampling
    # -------------------------

    def sample_candidate_gaussians(self, residual):
        H, W, C = residual.shape

        angle_choices = np.linspace(0, np.pi, self.config.angle_steps, endpoint=False)

        energy_map = np.sqrt((residual**2).sum(axis=2))
        flat_idx = np.argsort(energy_map.ravel())[::-1]

        peaks = []
        max_peaks_to_check = min(len(flat_idx), max(10, self.config.n_candidates//5))
        for i in range(max_peaks_to_check):
            idx = flat_idx[i]
            y = idx // W
            x = idx % W
            peaks.append((y, x))

        candidates = []

        for (py, px) in peaks:
            for sigma_x in self.config.sigma_choices:
                for sigma_y in self.config.sigma_choices:
                    for theta in angle_choices:
                        for alpha in self.config.alpha_choices:
                            kernel, _ = self.anisotropic_kernel(sigma_x, sigma_y, theta)
                            phi = self.place_kernel_full(kernel, py, px, H, W)
                            den = float((phi * phi).sum())
                            if den <= 1e-8:
                                continue
                            candidates.append(dict(
                                yc=int(py), xc=int(px),
                                sigma_x=float(sigma_x), sigma_y=float(sigma_y),
                                theta=float(theta), alpha=float(alpha),
                                phi=phi, den=den
                            ))

        n_random = max(0, self.config.n_candidates - len(candidates))
        for _ in range(n_random):
            py = random.randint(0, H-1)
            px = random.randint(0, W-1)
            sigma_x = float(random.choice(self.config.sigma_choices))
            sigma_y = float(random.choice(self.config.sigma_choices))
            theta = float(random.choice(angle_choices))
            alpha = float(random.choice(self.config.alpha_choices))
            kernel, _ = self.anisotropic_kernel(sigma_x, sigma_y, theta)
            phi = self.place_kernel_full(kernel, py, px, H, W)
            den = float((phi * phi).sum())
            if den <= 1e-8:
                continue
            candidates.append(dict(
                yc=py, xc=px,
                sigma_x=sigma_x, sigma_y=sigma_y,
                theta=theta, alpha=alpha,
                phi=phi, den=den
            ))

        if len(candidates) > self.config.n_candidates:
            candidates = random.sample(candidates, self.config.n_candidates)

        return candidates

    # -------------------------
    # Scoring
    # -------------------------

    def score_candidate_phi(self, phi, residual):
        num_r = (phi * residual[..., 0]).sum()
        num_g = (phi * residual[..., 1]).sum()
        num_b = (phi * residual[..., 2]).sum()
        num = np.array([num_r, num_g, num_b], dtype=np.float32)
        den = float((phi * phi).sum())
        corr = float(np.linalg.norm(num))
        return num, den, corr

    # -------------------------
    # Weak Matching Pursuit
    # -------------------------

    def run(self):
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        H, W, C = self.image.shape
        residual = self.image.copy()
        selected_atoms = []

        for it in range(self.config.n_iterations):

            candidates = self.sample_candidate_gaussians(residual)

            corrs = []
            scored = []

            for cand in candidates:
                num, den, corr = self.score_candidate_phi(cand['phi'], residual)
                scored.append((cand, num, den, corr))
                corrs.append(corr)

            if len(corrs) == 0:
                if self.config.verbose:
                    print(f"[iter {it}] No candidates scored; breaking.")
                break

            max_corr = max(corrs)
            if max_corr <= 0:
                if self.config.verbose:
                    print(f"[iter {it}] max_corr == 0; breaking.")
                break

            valid = [(cand, num, den, corr)
                     for (cand, num, den, corr) in scored
                     if corr >= self.config.alpha * max_corr]

            if not valid:
                selected = max(scored, key=lambda x: x[3])
                if self.config.verbose:
                    print(f"[iter {it}] No weak candidate found; falling back to best candidate.")
            else:
                selected = random.choice(valid)
                cand, num, den, corr = selected
                if self.config.verbose:
                    print(f"[iter {it}] Selected among {len(valid)} weak candidates (max_corr={max_corr:.4f}, chosen_corr={corr:.4f})")

            cand, num, den, corr = selected

            p = num / den
            phi = cand['phi']
            residual = residual - (phi[:, :, None] * p[None, None, :])

            atom_record = {
                'center': (int(cand['yc']), int(cand['xc'])),
                'sigma_x': float(cand['sigma_x']),
                'sigma_y': float(cand['sigma_y']),
                'theta': float(cand['theta']),
                'alpha': float(cand['alpha']),
                'p': p.copy(),
                'phi': phi,
                'den': den,
                'corr': corr
            }

            selected_atoms.append(atom_record)

            if self.config.verbose and ((it+1) % 10 == 0 or it == 0):
                energy = float((residual**2).sum())
                print(f"  Iter {it+1}/{self.config.n_iterations}  chosen_corr={corr:.4f}  residual_energy={energy:.6f}")

        self.selected_atoms = selected_atoms
        self.residual = residual
        self.reconstructed = self.reconstruct_from_atoms(selected_atoms, H, W)

        return selected_atoms, residual

    # -------------------------
    # Reconstruction
    # -------------------------

    def reconstruct_from_atoms(self, atoms, H, W):
        out = np.zeros((H, W, 3), dtype=np.float32)
        for a in atoms:
            phi = a['phi']
            p = a['p']
            out += phi[:, :, None] * p[None, None, :]
        return self.clamp01(out)

    # -------------------------
    # Required Base Method
    # -------------------------

    def render(self):
        self.run()
        return self.reconstructed

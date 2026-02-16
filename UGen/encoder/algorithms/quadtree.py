import numpy as np
from PIL import Image
import math
import os
import heapq
from typing import List, Tuple
from dataclasses import dataclass
from UGen.encoder.algorithms.BaseAlgorithm import *


# ------------------------------------------------------------
# Configuration Dataclass
# ------------------------------------------------------------

@dataclass
class QuadtreeGaussianConfig:
    max_depth: int = 6
    min_size: int = 8
    err_ratio_thresh: float = 0.06
    max_atoms: int = None
    verbose: bool = True


# ------------------------------------------------------------
# Quadtree Gaussian Encoder
# ------------------------------------------------------------

class QuadtreeGaussianEncoder(EncoderAlgorithms):

    def __init__(self, image: np.ndarray, config: QuadtreeGaussianConfig):
        self.I = image.astype(np.float32)
        self.config = config
        self._region_counter = 0
        self.atoms = []
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
        return np.asarray(im).astype(np.float32) / 255.0

    def save_image_rgb(self, arr, path):
        img = Image.fromarray((self.clamp01(arr) * 255.0).astype(np.uint8))
        img.save(path)

    def luminance_map(self, img):
        return 0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]

    # -------------------------
    # Moments
    # -------------------------

    def compute_region_moments(self, I_patch: np.ndarray, use_luminance: bool = True):
        h, w, _ = I_patch.shape
        if use_luminance:
            weights = self.luminance_map(I_patch)
        else:
            weights = I_patch.mean(axis=2)

        mass = float(weights.sum())
        if mass <= 1e-8:
            mu_y = (h-1)/2.0
            mu_x = (w-1)/2.0
            cov = np.eye(2, dtype=np.float32) * 1e-3
            return mass, mu_y, mu_x, cov

        ys = np.arange(h, dtype=np.float64)[:, None]
        xs = np.arange(w, dtype=np.float64)[None, :]

        mu_y = float((weights * ys).sum() / mass)
        mu_x = float((weights * xs).sum() / mass)

        dy = ys - mu_y
        dx = xs - mu_x

        s_yy = float((weights * (dy*dy)).sum() / mass)
        s_xx = float((weights * (dx*dx)).sum() / mass)
        s_xy = float((weights * (dx*dy)).sum() / mass)

        cov = np.array([[s_xx, s_xy], [s_xy, s_yy]], dtype=np.float64)
        cov += np.eye(2) * 1e-8
        return mass, mu_y, mu_x, cov.astype(np.float32)

    def stabilize_covariance(self, cov: np.ndarray, min_ratio: float = 1e-3, eps_abs: float = 1e-6):
        cov = np.array(cov, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)
        vals, vecs = np.linalg.eigh(cov)
        maxv = float(max(vals.max(), 0.0))
        floor = max(eps_abs, maxv * min_ratio)
        vals_clamped = np.maximum(vals, floor)
        cov_pd = (vecs * vals_clamped) @ vecs.T
        cov_pd += np.eye(2) * 1e-12
        return cov_pd

    # -------------------------
    # Gaussian phi
    # -------------------------

    def gaussian_phi_patch(self, h, w, mu_y, mu_x, cov):
        cov_pd = self.stabilize_covariance(cov, min_ratio=1e-3, eps_abs=1e-6)
        inv_cov = np.linalg.inv(cov_pd)

        ys = np.arange(h, dtype=np.float64)[:, None]
        xs = np.arange(w, dtype=np.float64)[None, :]

        dy = ys - mu_y
        dx = xs - mu_x

        a = inv_cov[0, 0]; b = inv_cov[0, 1]; c = inv_cov[1, 1]
        expv = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)
        expv = np.clip(expv, -50.0, 50.0)
        phi = np.exp(expv).astype(np.float32)
        return phi

    # -------------------------
    # Fit
    # -------------------------

    def fit_gaussian_to_patch(self, I_patch):
        mass, mu_y, mu_x, cov = self.compute_region_moments(I_patch)
        h, w, _ = I_patch.shape

        energy = float((I_patch ** 2).sum())
        if energy <= 1e-12 or mass <= 1e-8:
            return {
                'mass': mass, 'mu_y': mu_y, 'mu_x': mu_x, 'cov': cov,
                'phi': np.zeros((h, w), dtype=np.float32),
                'p': np.zeros(3, dtype=np.float32),
                'recon': np.zeros_like(I_patch),
                'error': float((I_patch**2).sum()),
                'energy': float((I_patch**2).sum())
            }

        phi = self.gaussian_phi_patch(h, w, mu_y, mu_x, cov)
        phi_sq_sum = float((phi * phi).sum())
        if phi_sq_sum <= 1e-12:
            return {
                'mass': mass, 'mu_y': mu_y, 'mu_x': mu_x, 'cov': cov,
                'phi': np.zeros((h, w), dtype=np.float32),
                'p': np.zeros(3, dtype=np.float32),
                'recon': np.zeros_like(I_patch),
                'error': float((I_patch**2).sum()),
                'energy': float((I_patch**2).sum())
            }

        num = (phi[..., None] * I_patch).reshape(-1, 3).sum(axis=0)
        p = (num / phi_sq_sum).astype(np.float32)

        recon = phi[..., None] * p[None, None, :]
        diff = I_patch - recon
        error = float((diff ** 2).sum())

        return {
            'mass': mass, 'mu_y': mu_y, 'mu_x': mu_x,
            'cov': cov, 'phi': phi, 'p': p, 'recon': recon,
            'error': error, 'energy': energy
        }

    # -------------------------
    # Heap helpers
    # -------------------------

    def make_region_entry(self, y0,y1,x0,x1, depth, fit):
        err_ratio = fit['error'] / (fit['energy'] + 1e-12)
        self._region_counter += 1
        data = { 'y0':y0,'y1':y1,'x0':x0,'x1':x1,'depth':depth,'fit':fit }
        return (-err_ratio, self._region_counter, data)

    # -------------------------
    # Priority Quadtree
    # -------------------------

    def priority_quadtree(self):
        I = self.I
        H,W,_ = I.shape
        atoms = []

        root_fit = self.fit_gaussian_to_patch(I)
        heap = []
        heapq.heappush(heap, self.make_region_entry(0,H,0,W,0, root_fit))

        while heap and (self.config.max_atoms is None or len(atoms) < self.config.max_atoms):
            neg_err_ratio, _, region = heapq.heappop(heap)
            err_ratio = -neg_err_ratio

            y0,y1,x0,x1 = region['y0'],region['y1'],region['x0'],region['x1']
            depth = region['depth']
            fit = region['fit']
            h = y1 - y0; w = x1 - x0

            if (h <= 0) or (w <= 0):
                continue

            if (depth >= self.config.max_depth) or \
               (h <= self.config.min_size and w <= self.config.min_size) or \
               (fit['energy'] <= 1e-12) or \
               (err_ratio <= self.config.err_ratio_thresh):

                mu_global_y = y0 + fit['mu_y']
                mu_global_x = x0 + fit['mu_x']

                atom = {
                    'y0':int(y0),'y1':int(y1),'x0':int(x0),'x1':int(x1),
                    'mu_y':float(mu_global_y),'mu_x':float(mu_global_x),
                    'cov': fit['cov'],'phi_patch': fit['phi'],
                    'p': fit['p'],'error':fit['error'],
                    'energy':fit['energy'],'err_ratio':err_ratio
                }

                atoms.append(atom)

                if self.config.verbose:
                    print(f"ATOM #{len(atoms)} depth={depth} bbox=({y0},{y1},{x0},{x1}) err_ratio={err_ratio:.4f} energy={fit['energy']:.4f}")
                continue

            ym = y0 + h // 2
            xm = x0 + w // 2
            children = [
                (y0, ym, x0, xm),
                (y0, ym, xm, x1),
                (ym, y1, x0, xm),
                (ym, y1, xm, x1)
            ]

            for (cy0, cy1, cx0, cx1) in children:
                if cy1 <= cy0 or cx1 <= cx0:
                    continue
                patch = I[cy0:cy1, cx0:cx1, :]
                fit_child = self.fit_gaussian_to_patch(patch)
                entry = self.make_region_entry(cy0, cy1, cx0, cx1, depth+1, fit_child)
                heapq.heappush(heap, entry)

        if self.config.verbose:
            print(f"Finished. Collected atoms: {len(atoms)}")

        self.atoms = atoms
        self.reconstructed = self.render_atoms(atoms, H, W)
        return atoms, self.reconstructed

    # -------------------------
    # Rendering
    # -------------------------

    def render_atoms(self, atoms: List[dict], H: int, W: int):
        atoms_sorted = sorted(atoms, key=lambda a: a['energy'])
        canvas = np.zeros((H,W,3), dtype=np.float32)

        for a in atoms_sorted:
            y0,y1,x0,x1 = a['y0'],a['y1'],a['x0'],a['x1']
            phi = a['phi_patch']; p = a['p']
            A3 = phi[...,None]
            patch = canvas[y0:y1, x0:x1, :]
            new_patch = p[None,None,:] * phi[...,None] + (1.0 - A3) * patch
            canvas[y0:y1, x0:x1, :] = new_patch

        return self.clamp01(canvas)

    # -------------------------
    # Required Base Method
    # -------------------------

    def render(self):
        atoms, recon = self.priority_quadtree()
        return recon

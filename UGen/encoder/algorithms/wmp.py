"""
Weak Matching Pursuit with anisotropic Gaussian atoms (NumPy)

Author: (adapted for you)
"""

import numpy as np
from PIL import Image
import math
import os
import random

# -------------------------
# Utilities
# -------------------------

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def load_image_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def save_image_rgb(arr, path):
    img = Image.fromarray((clamp01(arr) * 255.0).astype(np.uint8))
    img.save(path)

# -------------------------
# Anisotropic Gaussian kernel (peak = 1)
# -------------------------

def anisotropic_kernel(sigma_x, sigma_y, theta, radius=None):
    """
    Returns kernel (2D array) with peak==1 and radius.
    sigma_x, sigma_y in pixels; theta in radians.
    """
    max_sigma = max(sigma_x, sigma_y)
    if radius is None:
        radius = int(math.ceil(3.0 * max_sigma))
    xs = np.arange(-radius, radius+1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # rotate coords: x' = cos*x + sin*y ; y' = -sin*x + cos*y
    x_rot = cos_t * xx + sin_t * yy
    y_rot = -sin_t * xx + cos_t * yy
    g = np.exp(-0.5 * ((x_rot**2) / (sigma_x**2) + (y_rot**2) / (sigma_y**2)))
    return g.astype(np.float32), radius

# -------------------------
# Helpers to create full-image atom phi from kernel + center
# -------------------------

def place_kernel_full(kernel, center_y, center_x, H, W):
    """
    Place kernel (ksz x ksz) at center (center_y, center_x) into an HxW array.
    Kernel assumed to be centered at its center index.
    """
    ksz = kernel.shape[0]
    radius = (ksz - 1) // 2
    phi = np.zeros((H, W), dtype=np.float32)
    y0 = center_y - radius
    x0 = center_x - radius
    y1 = y0 + ksz
    x1 = x0 + ksz
    # clip bounds
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
# Candidate sampling
# -------------------------

def sample_candidate_gaussians(residual, n_samples=100, sigma_choices=(2,4,8,16),
                               n_peak_candidates=20, angle_choices=None, alpha_choices=(1.0,)):
    """
    Sample candidate gaussian atoms from the residual.
    Strategy:
      - find n_peak_candidates highest-residual-energy pixels (peaks) and sample some candidates around them
      - add random samples across the image
    Returns a list of candidate dicts (each contains kernel, center, sigma_x, sigma_y, theta, alpha, den)
    """
    H, W, C = residual.shape
    if angle_choices is None:
        angle_choices = np.linspace(0, np.pi, 8, endpoint=False)  # 8 orientations by default

    # scalar residual energy map (l2 over channels)
    energy_map = np.sqrt((residual**2).sum(axis=2))

    # Flatten and pick peaks
    flat_idx = np.argsort(energy_map.ravel())[::-1]
    peaks = []
    max_peaks_to_check = min(len(flat_idx), n_peak_candidates)
    for i in range(max_peaks_to_check):
        idx = flat_idx[i]
        y = idx // W
        x = idx % W
        peaks.append((y, x))

    candidates = []

    # Around each peak, sample a few sigma/angle/alpha combos
    for (py, px) in peaks:
        for sigma_x in sigma_choices:
            for sigma_y in sigma_choices:
                for theta in angle_choices:
                    for alpha in alpha_choices:
                        kernel, _ = anisotropic_kernel(sigma_x, sigma_y, theta)
                        phi = place_kernel_full(kernel, py, px, H, W)  # full-size phi
                        den = float((phi * phi).sum())  # scalar denominator
                        if den <= 1e-8:
                            continue
                        candidates.append(dict(
                            yc=int(py), xc=int(px),
                            sigma_x=float(sigma_x), sigma_y=float(sigma_y),
                            theta=float(theta), alpha=float(alpha),
                            phi=phi, den=den
                        ))

    # Random samples across image
    n_random = max(0, n_samples - len(candidates))
    for _ in range(n_random):
        py = random.randint(0, H-1)
        px = random.randint(0, W-1)
        sigma_x = float(random.choice(sigma_choices))
        sigma_y = float(random.choice(sigma_choices))
        theta = float(random.choice(angle_choices))
        alpha = float(random.choice(alpha_choices))
        kernel, _ = anisotropic_kernel(sigma_x, sigma_y, theta)
        phi = place_kernel_full(kernel, py, px, H, W)
        den = float((phi * phi).sum())
        if den <= 1e-8:
            continue
        candidates.append(dict(
            yc=py, xc=px,
            sigma_x=sigma_x, sigma_y=sigma_y,
            theta=theta, alpha=alpha,
            phi=phi, den=den
        ))

    # Limit to requested n_samples if too many
    if len(candidates) > n_samples:
        candidates = random.sample(candidates, n_samples)

    return candidates

# -------------------------
# Correlation / scoring
# -------------------------

def score_candidate_phi(phi, residual):
    """
    phi: HxW float kernel (peak 1)
    residual: HxWx3 residual image
    Returns:
      - num: 3-vector numerator = sum_x phi(x) * residual(x)
      - den: scalar = sum_x phi(x)^2
      - corr: scalar correlation magnitude = ||num|| (L2 across channels)
    """
    # compute numerator for each channel
    # num[c] = sum phi * residual[..., c]
    num_r = (phi * residual[..., 0]).sum()
    num_g = (phi * residual[..., 1]).sum()
    num_b = (phi * residual[..., 2]).sum()
    num = np.array([num_r, num_g, num_b], dtype=np.float32)
    den = float((phi * phi).sum())
    corr = float(np.linalg.norm(num))
    return num, den, corr

# -------------------------
# Weak Matching Pursuit main
# -------------------------

def weak_matching_pursuit(image, n_iterations, alpha=0.8, n_candidates=120,
                          sigma_choices=(2,4,8,16), angle_steps=8,
                          alpha_choices=(1.0,), random_seed=None, verbose=True):
    """
    image: HxWx3 float in [0,1]
    n_iterations: number of atoms to pick
    alpha: weakness parameter in (0,1] (higher = stricter)
    n_candidates: how many candidates to sample each iteration
    sigma_choices: tuple/list of sigma values to try (sigma_x and sigma_y chosen from this set)
    angle_steps: how many discrete orientations between [0, pi)
    alpha_choices: list of alpha occlusion values to consider (usually <= 1.0)
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    H, W, C = image.shape
    assert C == 3

    residual = image.astype(np.float32).copy()  # start from original image over black background
    selected_atoms = []

    angle_choices = np.linspace(0, np.pi, angle_steps, endpoint=False)

    for it in range(n_iterations):
        # sample candidates
        candidates = sample_candidate_gaussians(residual,
                                               n_samples=n_candidates,
                                               sigma_choices=sigma_choices,
                                               n_peak_candidates=max(10, n_candidates//5),
                                               angle_choices=angle_choices,
                                               alpha_choices=alpha_choices)

        # score all candidates
        corrs = []
        scored = []
        for cand in candidates:
            num, den, corr = score_candidate_phi(cand['phi'], residual)
            scored.append((cand, num, den, corr))
            corrs.append(corr)

        if len(corrs) == 0:
            if verbose:
                print(f"[iter {it}] No candidates scored; breaking.")
            break

        max_corr = max(corrs)
        if max_corr <= 0:
            if verbose:
                print(f"[iter {it}] max_corr == 0; breaking.")
            break

        # weak selection: accept any candidate with corr >= alpha * max_corr
        valid = [(cand, num, den, corr) for (cand, num, den, corr) in scored if corr >= alpha * max_corr]

        if not valid:
            # fallback: pick the best (strict)
            best = max(scored, key=lambda x: x[3])
            selected = best
            if verbose:
                print(f"[iter {it}] No weak candidate found; falling back to best candidate.")
        else:
            selected = random.choice(valid)
            cand, num, den, corr = selected
            if verbose:
                print(f"[iter {it}] Selected among {len(valid)} weak candidates (max_corr={max_corr:.4f}, chosen_corr={corr:.4f})")

        cand, num, den, corr = selected

        # compute premultiplied color coefficient p (3-vector)
        # p = num / den
        p = num / den  # may be negative (we allow +/- to approximate)
        # update residual: residual -= p * phi (broadcasted over channels)
        phi = cand['phi']
        # subtract p * phi from residual across channels
        residual = residual - (phi[:, :, None] * p[None, None, :])

        # store selected atom with parameters and coefficient p
        atom_record = {
            'center': (int(cand['yc']), int(cand['xc'])),
            'sigma_x': float(cand['sigma_x']),
            'sigma_y': float(cand['sigma_y']),
            'theta': float(cand['theta']),
            'alpha': float(cand['alpha']),
            'p': p.copy(),        # premultiplied RGB that is subtracted: p * phi
            'phi': phi,           # store phi if you want to re-render later (or drop to save memory)
            'den': den,
            'corr': corr
        }
        selected_atoms.append(atom_record)

        # optionally print progress
        if verbose and ((it+1) % 10 == 0 or it == 0):
            energy = float((residual**2).sum())
            print(f"  Iter {it+1}/{n_iterations}  chosen_corr={corr:.4f}  residual_energy={energy:.6f}")

    return selected_atoms, residual

# -------------------------
# Reconstruct image from atoms
# -------------------------

def reconstruct_from_atoms(atoms, H, W):
    out = np.zeros((H, W, 3), dtype=np.float32)
    for a in atoms:
        phi = a['phi']    # HxW
        p = a['p']        # 3-vector premultiplied color
        out += phi[:, :, None] * p[None, None, :]
    out = clamp01(out)
    return out


"""
Quadtree / Hierarchical Gaussian Approximation (NumPy + Pillow)

- Recursively subdivide image regions using a quadtree.
- Fit one anisotropic Gaussian atom per region using image moments.
- Subdivide regions whose L2 reconstruction error (region vs fitted Gaussian) is above a threshold ratio.
- Finally render atoms back-to-front (black background) using premultiplied compositing.

Author: adapted for you
"""

import numpy as np
from PIL import Image
import math
import os
from typing import List, Tuple

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

def luminance_map(img):
    """Compute a scalar intensity map for weighting (use Rec. 601 weights)."""
    # img: H,W,3 in [0,1]
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

# -------------------------
# Moments -> Gaussian params
# -------------------------

def compute_region_moments(I_patch: np.ndarray, use_luminance: bool = True):
    """
    Compute intensity-weighted first and second order moments for patch.
    I_patch: (h, w, 3) float image values in [0,1]
    Returns:
      total_mass (scalar),
      mu_y, mu_x (floats, coordinates in patch-local pixel coords where top-left=0),
      cov (2x2 covariance matrix)
    """
    h, w, _ = I_patch.shape
    if use_luminance:
        weights = luminance_map(I_patch)
    else:
        weights = I_patch.mean(axis=2)
    mass = float(weights.sum())
    if mass <= 0.0:
        return 0.0, 0.0, 0.0, np.eye(2, dtype=np.float32) * 1e-6

    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]

    mu_y = float((weights * ys).sum() / mass)
    mu_x = float((weights * xs).sum() / mass)

    dy = ys - mu_y
    dx = xs - mu_x

    s_yy = float((weights * (dy * dy)).sum() / mass)
    s_xx = float((weights * (dx * dx)).sum() / mass)
    s_xy = float((weights * (dx * dy)).sum() / mass)

    cov = np.array([[s_xx, s_xy], [s_xy, s_yy]], dtype=np.float32)

    # regularize tiny variances so kernels have reasonable support
    eps = 1e-4
    cov[0, 0] = max(cov[0, 0], eps)
    cov[1, 1] = max(cov[1, 1], eps)
    return mass, mu_y, mu_x, cov

def cov_to_ellipse_params(cov: np.ndarray):
    """
    Convert 2x2 covariance to (sigma_x, sigma_y, theta) where theta is rotation
    of principal axis relative to image x-axis (radians).
    sigma_x >= sigma_y (sigma_x along largest eigenvalue).
    """
    vals, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    # eigenvalues could be very small; ensure non-negative
    vals = np.maximum(vals, 1e-8)
    # largest eigenvalue index
    i_max = 1
    i_min = 0
    lambda_max = vals[i_max]
    lambda_min = vals[i_min]
    # Convert variance to sigma (stddev)
    sigma_x = math.sqrt(lambda_max)
    sigma_y = math.sqrt(lambda_min)
    # principal eigenvector (corresponding to lambda_max)
    v = vecs[:, i_max]  # shape (2,)
    # angle: arctan2(v_y, v_x)
    theta = math.atan2(v[1], v[0])
    return sigma_x, sigma_y, theta

# -------------------------
# Create Gaussian patch phi for a given region and parameters
# -------------------------

def gaussian_phi_patch(h, w, mu_y, mu_x, cov):
    """
    Create phi over a patch of shape (h,w), where coordinates are patch-local
    (0..h-1, 0..w-1), and Gaussian center at (mu_y, mu_x) (float).
    phi peak = 1 at center.
    cov is 2x2 covariance (variances).
    Returns phi (h,w)
    """
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]

    dy = ys - mu_y
    dx = xs - mu_x

    # compute inverse covariance
    inv_cov = np.linalg.inv(cov)
    # exponent = -0.5 * [dx dy] * inv_cov * [dx; dy]
    a = inv_cov[0, 0]
    b = inv_cov[0, 1]  # = inv_cov[1,0]
    c = inv_cov[1, 1]

    # exponent at each pixel:
    expv = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)
    phi = np.exp(expv)
    return phi.astype(np.float32)

# -------------------------
# Fit one Gaussian to a patch (closed-form p)
# -------------------------

def fit_gaussian_to_patch(I_patch):
    """
    Given image patch I_patch (h,w,3), compute:
      - mass, mu_y, mu_x, cov (moments)
      - build phi over patch using cov and mu
      - compute p (premultiplied color) = sum(phi * I_patch) / sum(phi^2)
      - compute reconstruction inside patch: recon = phi[...,None] * p
      - compute error energy and energy of region
    Returns:
      dict with entries: 'mass','mu_y','mu_x','cov','phi','p','recon','error','energy'
    """
    mass, mu_y, mu_x, cov = compute_region_moments(I_patch)
    h, w, _ = I_patch.shape
    if mass <= 0.0:
        return {
            'mass': 0.0, 'mu_y': 0.0, 'mu_x': 0.0, 'cov': cov,
            'phi': np.zeros((h, w), dtype=np.float32),
            'p': np.zeros(3, dtype=np.float32),
            'recon': np.zeros_like(I_patch),
            'error': float(((I_patch)**2).sum()),  # if empty, error = energy
            'energy': float(((I_patch)**2).sum())
        }

    phi = gaussian_phi_patch(h, w, mu_y, mu_x, cov)  # h x w
    phi_sq_sum = float((phi * phi).sum())

    # numerator: sum_x phi(x) * I_patch(x) per channel
    num = (phi[..., None] * I_patch).reshape(-1, 3).sum(axis=0)  # 3-vector
    if phi_sq_sum <= 1e-12:
        p = np.zeros(3, dtype=np.float32)
    else:
        p = (num / phi_sq_sum).astype(np.float32)  # premultiplied color

    recon = phi[..., None] * p[None, None, :]  # h,w,3
    diff = I_patch - recon
    error = float((diff ** 2).sum())
    energy = float((I_patch ** 2).sum())

    return {
        'mass': mass,
        'mu_y': mu_y, 'mu_x': mu_x, 'cov': cov,
        'phi': phi, 'p': p, 'recon': recon,
        'error': error, 'energy': energy
    }

# -------------------------
# Quadtree recursion
# -------------------------

def quadtree_subdivide(I, y0, y1, x0, x1, max_depth, min_size,
                       err_ratio_thresh, atoms_list, depth=0, max_atoms=None):
    """
    Recursively subdivide bounding box [y0,y1) x [x0,x1) on image I.
    If region fit error / region energy > err_ratio_thresh, subdivide (if allowed).
    At leaves, append atom dictionaries to atoms_list.
    """
    # region dimensions
    h = y1 - y0
    w = x1 - x0

    # stop if region too small
    if h <= 0 or w <= 0:
        return

    I_patch = I[y0:y1, x0:x1, :]
    fit = fit_gaussian_to_patch(I_patch)

    # if zero energy, skip
    if fit['energy'] <= 1e-12:
        return

    # error ratio (normalized)
    err_ratio = fit['error'] / (fit['energy'] + 1e-12)

    # stopping criteria
    stop = (depth >= max_depth) or (h <= min_size and w <= min_size) or (err_ratio <= err_ratio_thresh)
    # also stop if number of atoms reached
    if (max_atoms is not None) and (len(atoms_list) >= max_atoms):
        stop = True

    if stop:
        # compute absolute center in full image coords:
        # mu coordinates are patch-local (0..h-1, 0..w-1); convert to global coords
        mu_global_y = y0 + fit['mu_y']
        mu_global_x = x0 + fit['mu_x']
        atom = {
            'y0': int(y0), 'y1': int(y1), 'x0': int(x0), 'x1': int(x1),
            'mu_y': float(mu_global_y), 'mu_x': float(mu_global_x),
            'cov': fit['cov'],
            'phi_patch': fit['phi'],    # local phi on the patch
            'p': fit['p'],              # premultiplied color (to composite)
            'error': fit['error'],
            'energy': fit['energy'],
            'err_ratio': err_ratio
        }
        atoms_list.append(atom)
        return
    else:
        # subdivide into 4 quadrants
        ym = y0 + h // 2
        xm = x0 + w // 2
        # ensure subdivisions are non-empty
        quadtree_subdivide(I, y0, ym, x0, xm, max_depth, min_size, err_ratio_thresh, atoms_list, depth+1, max_atoms)
        quadtree_subdivide(I, y0, ym, xm, x1, max_depth, min_size, err_ratio_thresh, atoms_list, depth+1, max_atoms)
        quadtree_subdivide(I, ym, y1, x0, xm, max_depth, min_size, err_ratio_thresh, atoms_list, depth+1, max_atoms)
        quadtree_subdivide(I, ym, y1, xm, x1, max_depth, min_size, err_ratio_thresh, atoms_list, depth+1, max_atoms)
        return

# -------------------------
# Render atoms back-to-front
# -------------------------

def render_atoms(atoms: List[dict], H: int, W: int):
    """
    Composite atoms onto black background in back-to-front order.
    Heuristic ordering: sort atoms by (energy) ascending => dimmer/background first, bright features later.
    """
    # sort: small energy first -> they'll be at back
    atoms_sorted = sorted(atoms, key=lambda a: a['energy'])

    canvas = np.zeros((H, W, 3), dtype=np.float32)  # premultiplied canvas; background black

    for a in atoms_sorted:
        y0, y1, x0, x1 = a['y0'], a['y1'], a['x0'], a['x1']
        phi = a['phi_patch']   # patch local size
        p = a['p']             # premultiplied color
        # Compose: C_out = p * phi + (1 - phi) * C_in  (here alpha=1 => A=phi)
        # phi shape (h,w) and p shape (3,)
        h, w = phi.shape
        # ensure patch fits (it should)
        patch = canvas[y0:y1, x0:x1, :]
        A3 = phi[..., None]   # as alpha (since alpha=1)
        new_patch = (p[None, None, :] * phi[..., None]) + (1.0 - A3) * patch
        canvas[y0:y1, x0:x1, :] = new_patch

    # final visible color is premultiplied canvas (background black)
    return clamp01(canvas)

# -------------------------
# Top-level helper
# -------------------------

def quadtree_gaussian_approximation(I: np.ndarray,
                                    max_depth=6,
                                    min_size=8,
                                    err_ratio_thresh=0.06,
                                    max_atoms=None,
                                    verbose=True):
    """
    I: HxWx3 float image in [0,1]
    Returns: (atoms_list, reconstructed_image)
    """
    H, W, _ = I.shape
    atoms: List[dict] = []
    quadtree_subdivide(I, 0, H, 0, W, max_depth, min_size, err_ratio_thresh, atoms, depth=0, max_atoms=max_atoms)
    if verbose:
        print(f"Collected {len(atoms)} atoms from quadtree.")
    recon = render_atoms(atoms, H, W)
    return atoms, recon


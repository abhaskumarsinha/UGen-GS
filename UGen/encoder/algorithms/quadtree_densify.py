import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Gaussian2D:
    """Represents a 2D Gaussian splat with mean (x,y), covariance (2x2), color (RGB), and opacity."""
    mean: np.ndarray      # shape (2,)
    cov: np.ndarray       # shape (2,2)
    color: np.ndarray     # shape (3,)
    opacity: float = 1.0

def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from disk, optionally resize to target_size (width, height).
    Returns RGB float array in [0,1] of shape (target_height, target_width, 3)
    or original size if target_size is None.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB
    if target_size is not None:
        # target_size = (width, height)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def to_grayscale_with_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RGB image to grayscale and compute edge magnitude using Sobel.
    Returns:
        gray: (H,W) float grayscale in [0,1]
        edges: (H,W) float gradient magnitude (normalised to [0,1])
    """
    # Convert to uint8 for OpenCV, then back to float
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Compute Sobel gradients with depth CV_32F to match float32 input
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    if grad_mag.max() > 0:
        grad_mag /= grad_mag.max()
    return gray, grad_mag

def select_important_points(edge_map: np.ndarray, num_points: int = 1000) -> np.ndarray:
    """
    Select points with probability proportional to edge strength.
    A small epsilon is added to all pixels to guarantee that every pixel has
    a non‑zero probability, enabling sampling without replacement even when
    the number of requested points is large.
    Returns an array of shape (N,2) with (x,y) coordinates (column, row).
    """
    h, w = edge_map.shape
    flat_edges = edge_map.flatten()
    # Add a tiny constant to avoid zero probabilities
    flat_edges = flat_edges + 1e-6
    probs = flat_edges / flat_edges.sum()
    indices = np.random.choice(h * w, size=num_points, replace=False, p=probs)
    ys, xs = np.divmod(indices, w)
    return np.stack([xs, ys], axis=1).astype(np.float32)

def weighted_mean_cov(points: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total_weight = weights.sum()
    if total_weight == 0:
        return np.zeros(2), np.eye(2) * 1e-6
    mean = np.average(points, axis=0, weights=weights)
    centered = points - mean
    cov = np.dot(centered.T, centered * weights[:, None]) / total_weight
    cov += np.eye(2) * 1e-6
    return mean, cov

def fit_gaussian_to_region(points: np.ndarray, colors: np.ndarray, weights: np.ndarray) -> Gaussian2D:
    mean, cov = weighted_mean_cov(points, weights)
    if len(colors) > 0:
        avg_color = np.average(colors, axis=0, weights=weights)
    else:
        avg_color = np.array([0.5, 0.5, 0.5])
    return Gaussian2D(mean=mean, cov=cov, color=avg_color, opacity=1.0)

def build_quadtree(
    image: np.ndarray,
    edge_map: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    bbox: Tuple[int, int, int, int],
    max_points_per_leaf: int = 10,
    min_bbox_size: int = 4
) -> List[Gaussian2D]:
    xmin, xmax, ymin, ymax = bbox
    mask = (points[:, 0] >= xmin) & (points[:, 0] < xmax) & (points[:, 1] >= ymin) & (points[:, 1] < ymax)
    pts_in = points[mask]
    cols_in = colors[mask]
    w_in = weights[mask]

    if len(pts_in) <= max_points_per_leaf or (xmax - xmin <= min_bbox_size and ymax - ymin <= min_bbox_size):
        if len(pts_in) > 0:
            return [fit_gaussian_to_region(pts_in, cols_in, w_in)]
        else:
            return []

    xmid = (xmin + xmax) // 2
    ymid = (ymin + ymax) // 2
    gaussians = []
    gaussians.extend(build_quadtree(image, edge_map, points, colors, weights,
                                    (xmin, xmid, ymin, ymid), max_points_per_leaf, min_bbox_size))
    gaussians.extend(build_quadtree(image, edge_map, points, colors, weights,
                                    (xmid, xmax, ymin, ymid), max_points_per_leaf, min_bbox_size))
    gaussians.extend(build_quadtree(image, edge_map, points, colors, weights,
                                    (xmin, xmid, ymid, ymax), max_points_per_leaf, min_bbox_size))
    gaussians.extend(build_quadtree(image, edge_map, points, colors, weights,
                                    (xmid, xmax, ymid, ymax), max_points_per_leaf, min_bbox_size))
    return gaussians

def render_gaussians(gaussians: List[Gaussian2D], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([xs, ys], axis=-1).astype(np.float32)

    total_weight = np.zeros((h, w, 1), dtype=np.float32)
    color_sum = np.zeros((h, w, 3), dtype=np.float32)

    for g in gaussians:
        delta = coords - g.mean
        try:
            inv_cov = np.linalg.inv(g.cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(g.cov)
        temp = np.einsum('...j,ij->...i', delta, inv_cov)
        mahal = np.sum(temp * delta, axis=-1)
        weight = np.exp(-0.5 * mahal)
        total_weight[:, :, 0] += weight
        color_sum += weight[..., None] * g.color

    total_weight = np.maximum(total_weight, 1e-6)
    rendered = color_sum / total_weight
    return np.clip(rendered, 0, 1)

def compute_error(original: np.ndarray, rendered: np.ndarray) -> float:
    return np.mean((original - rendered) ** 2)

def refine_gaussians(
    gaussians: List[Gaussian2D],
    image: np.ndarray,
    edge_map: np.ndarray,
    error_threshold: float = 0.001,
    max_iterations: int = 7
) -> List[Gaussian2D]:
    refined = []
    for g in gaussians:
        try:
            eigvals, eigvecs = np.linalg.eigh(g.cov)
        except:
            eigvals = [0, 0]
        max_eig = max(eigvals)
        if max_eig > 5.0:
            direction = eigvecs[:, np.argmax(eigvals)]
            shift = np.sqrt(max_eig) * direction
            g1 = Gaussian2D(mean=g.mean + shift, cov=g.cov/2, color=g.color, opacity=g.opacity)
            g2 = Gaussian2D(mean=g.mean - shift, cov=g.cov/2, color=g.color, opacity=g.opacity)
            refined.extend([g1, g2])
        else:
            refined.append(g)
    return refined

def quadtree_densify(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),  # (width, height)
    num_initial_points: int = 2000,
    error_threshold=0.01,
    max_iterations=7
):
    """
    Full pipeline: load image, resize to target_size, detect edges, select points,
    build quadtree Gaussians, render, refine, and return final rendered image.
    """
    # 1. Load and resize image
    img = load_image(image_path, target_size)
    h, w = img.shape[:2]
    print(f"Working resolution: {w} x {h}")

    # 2. Grayscale and edge map
    gray, edges = to_grayscale_with_edges(img)

    # 3. Select important points weighted by edges
    points = select_important_points(edges, num_points=num_initial_points)

    # 4. Sample colors and weights at those points
    pts_int = np.round(points).astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, w-1)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, h-1)
    colors = img[pts_int[:, 1], pts_int[:, 0]]
    weights = edges[pts_int[:, 1], pts_int[:, 0]]

    # 5. Build quadtree and obtain Gaussians
    bbox = (0, w, 0, h)
    gaussians = build_quadtree(img, edges, points, colors, weights, bbox,
                               max_points_per_leaf=5, min_bbox_size=8)
    print(f"Initial number of Gaussians: {len(gaussians)}")

    # 6. Render initial Gaussians
    rendered = render_gaussians(gaussians, (h, w))
    error = compute_error(img, rendered)
    print(f"Initial rendering MSE: {error:.6f}")

    # 7. Simple refinement
    gaussians = refine_gaussians(gaussians, img, edges, error_threshold=error_threshold, max_iterations=max_iterations)
    print(f"After refinement: {len(gaussians)} Gaussians")

    # 8. Render final
    final_rendered = render_gaussians(gaussians, (h, w))
    final_error = compute_error(img, final_rendered)
    print(f"Final rendering MSE: {final_error:.6f}")

    return gaussians, final_rendered

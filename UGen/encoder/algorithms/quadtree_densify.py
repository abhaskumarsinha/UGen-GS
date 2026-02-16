import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
from UGen.encoder.algorithms.BaseAlgorithm import *



# ------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------

@dataclass
class Gaussian2D:
    mean: np.ndarray
    cov: np.ndarray
    color: np.ndarray
    opacity: float = 1.0


@dataclass
class QuadtreeDensifyConfig:
    target_size: Tuple[int, int] = (512, 512)
    num_initial_points: int = 2000
    error_threshold: float = 0.01
    max_iterations: int = 7
    max_points_per_leaf: int = 5
    min_bbox_size: int = 8


# ------------------------------------------------------------
# Encoder
# ------------------------------------------------------------

class QuadtreeDensifyEncoder(EncoderAlgorithms):

    def __init__(self, image_path: str, config: QuadtreeDensifyConfig):
        self.image_path = image_path
        self.config = config
        self.gaussians: List[Gaussian2D] = []
        self.rendered = None

    # ------------------------------------------------------------
    # Original Functions (Unmodified Logic)
    # ------------------------------------------------------------

    def load_image(self, path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def to_grayscale_with_edges(self, image: np.ndarray):
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        if grad_mag.max() > 0:
            grad_mag /= grad_mag.max()
        return gray, grad_mag

    def select_important_points(self, edge_map: np.ndarray, num_points: int = 1000):
        h, w = edge_map.shape
        flat_edges = edge_map.flatten()
        flat_edges = flat_edges + 1e-6
        probs = flat_edges / flat_edges.sum()
        indices = np.random.choice(h * w, size=num_points, replace=False, p=probs)
        ys, xs = np.divmod(indices, w)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def weighted_mean_cov(self, points: np.ndarray, weights: np.ndarray):
        total_weight = weights.sum()
        if total_weight == 0:
            return np.zeros(2), np.eye(2) * 1e-6
        mean = np.average(points, axis=0, weights=weights)
        centered = points - mean
        cov = np.dot(centered.T, centered * weights[:, None]) / total_weight
        cov += np.eye(2) * 1e-6
        return mean, cov

    def fit_gaussian_to_region(self, points, colors, weights):
        mean, cov = self.weighted_mean_cov(points, weights)
        if len(colors) > 0:
            avg_color = np.average(colors, axis=0, weights=weights)
        else:
            avg_color = np.array([0.5, 0.5, 0.5])
        return Gaussian2D(mean=mean, cov=cov, color=avg_color, opacity=1.0)

    def build_quadtree(self, image, edge_map, points, colors, weights, bbox):
        xmin, xmax, ymin, ymax = bbox
        mask = (points[:, 0] >= xmin) & (points[:, 0] < xmax) & \
               (points[:, 1] >= ymin) & (points[:, 1] < ymax)
        pts_in = points[mask]
        cols_in = colors[mask]
        w_in = weights[mask]

        if len(pts_in) <= self.config.max_points_per_leaf or \
           (xmax - xmin <= self.config.min_bbox_size and ymax - ymin <= self.config.min_bbox_size):

            if len(pts_in) > 0:
                return [self.fit_gaussian_to_region(pts_in, cols_in, w_in)]
            else:
                return []

        xmid = (xmin + xmax) // 2
        ymid = (ymin + ymax) // 2
        gaussians = []
        gaussians.extend(self.build_quadtree(image, edge_map, points, colors, weights,
                                             (xmin, xmid, ymin, ymid)))
        gaussians.extend(self.build_quadtree(image, edge_map, points, colors, weights,
                                             (xmid, xmax, ymin, ymid)))
        gaussians.extend(self.build_quadtree(image, edge_map, points, colors, weights,
                                             (xmin, xmid, ymid, ymax)))
        gaussians.extend(self.build_quadtree(image, edge_map, points, colors, weights,
                                             (xmid, xmax, ymid, ymax)))
        return gaussians

    def render_gaussians(self, gaussians: List[Gaussian2D], shape: Tuple[int, int]):
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

    def compute_error(self, original, rendered):
        return np.mean((original - rendered) ** 2)

    def refine_gaussians(self, gaussians, image, edge_map):
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

    # ------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------

    def run_pipeline(self):
        img = self.load_image(self.image_path, self.config.target_size)
        h, w = img.shape[:2]

        gray, edges = self.to_grayscale_with_edges(img)

        points = self.select_important_points(edges, self.config.num_initial_points)

        pts_int = np.round(points).astype(int)
        pts_int[:, 0] = np.clip(pts_int[:, 0], 0, w-1)
        pts_int[:, 1] = np.clip(pts_int[:, 1], 0, h-1)

        colors = img[pts_int[:, 1], pts_int[:, 0]]
        weights = edges[pts_int[:, 1], pts_int[:, 0]]

        bbox = (0, w, 0, h)
        gaussians = self.build_quadtree(img, edges, points, colors, weights, bbox)

        rendered = self.render_gaussians(gaussians, (h, w))
        error = self.compute_error(img, rendered)

        gaussians = self.refine_gaussians(gaussians, img, edges)

        final_rendered = self.render_gaussians(gaussians, (h, w))

        self.gaussians = gaussians
        self.rendered = final_rendered

        return gaussians, final_rendered

    # ------------------------------------------------------------
    # Required Base Method
    # ------------------------------------------------------------

    def render(self):
        _, final = self.run_pipeline()
        return final

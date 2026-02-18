import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
from UGen.encoder.algorithms.BaseAlgorithm import EncoderAlgorithms


@dataclass
class Gaussian2D:
    mean: np.ndarray      # (2,) float, pixel coordinates (x, y)
    cov: np.ndarray       # (2,2) float, covariance matrix
    color: np.ndarray     # (3,) float in [0,1]
    opacity: float = 1.0  # base opacity


@dataclass
class AlphaDensifyConfig:
    target_size: Tuple[int, int] = (512, 512)
    num_initial_points: int = 2000
    error_threshold: float = 0.01          # stop when MSE < this
    max_iterations: int = 10
    points_per_iteration: int = 500        # new points added each iteration
    max_points_per_leaf: int = 5
    min_bbox_size: int = 8
    cov_split_threshold: float = 5.0       # split Gaussians with max eigenvalue > this


class AlphaQuadtreeDensifyEncoder(EncoderAlgorithms):
    """
    Adaptive Gaussian splatting with alpha compositing.
    Uses quadtree decomposition of points sampled from edge and error maps.
    """

    def __init__(self, image_path: str, config: AlphaDensifyConfig):
        self.image_path = image_path
        self.config = config
        self.gaussians: List[Gaussian2D] = []
        self.rendered = None

    # ------------------------------------------------------------
    # Helper functions (adapted from original)
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

    def select_important_points(self, importance_map: np.ndarray, num_points: int) -> np.ndarray:
        """Sample pixel coordinates from a 2D importance map (treated as probability distribution)."""
        h, w = importance_map.shape
        flat = importance_map.flatten()
        flat = flat + 1e-6          # avoid zero probabilities
        probs = flat / flat.sum()
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
        cov += np.eye(2) * 1e-6      # regularize
        return mean, cov

    def fit_gaussian_to_region(self, points: np.ndarray, colors: np.ndarray, weights: np.ndarray) -> Gaussian2D:
        mean, cov = self.weighted_mean_cov(points, weights)
        total_weight = weights.sum()
        if total_weight > 0:
            avg_color = np.average(colors, axis=0, weights=weights)
        else:
            # Fallback to unweighted average if there are points, else default gray
            if len(colors) > 0:
                avg_color = np.mean(colors, axis=0)
            else:
                avg_color = np.array([0.5, 0.5, 0.5])
        return Gaussian2D(mean=mean, cov=cov, color=avg_color, opacity=1.0)

    def build_quadtree(self, image: np.ndarray, edge_map: np.ndarray,
                       points: np.ndarray, colors: np.ndarray, weights: np.ndarray,
                       bbox: Tuple[int, int, int, int]) -> List[Gaussian2D]:
        """Recursively partition the bounding box and fit a Gaussian to each leaf."""
        xmin, xmax, ymin, ymax = bbox
        mask = (points[:, 0] >= xmin) & (points[:, 0] < xmax) & \
               (points[:, 1] >= ymin) & (points[:, 1] < ymax)
        pts_in = points[mask]
        cols_in = colors[mask]
        w_in = weights[mask]

        # Leaf condition: few points or small cell size
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

    def split_large_gaussians(self, gaussians: List[Gaussian2D]) -> List[Gaussian2D]:
        """Split any Gaussian whose covariance is too large (eigenvalue > threshold)."""
        refined = []
        for g in gaussians:
            try:
                eigvals, eigvecs = np.linalg.eigh(g.cov)
            except np.linalg.LinAlgError:
                eigvals = [0, 0]
            max_eig = max(eigvals)
            if max_eig > self.config.cov_split_threshold:
                direction = eigvecs[:, np.argmax(eigvals)]
                shift = np.sqrt(max_eig) * direction
                # Create two children with half the covariance
                g1 = Gaussian2D(mean=g.mean + shift, cov=g.cov / 2,
                                color=g.color, opacity=g.opacity)
                g2 = Gaussian2D(mean=g.mean - shift, cov=g.cov / 2,
                                color=g.color, opacity=g.opacity)
                refined.extend([g1, g2])
            else:
                refined.append(g)
        return refined

    # ------------------------------------------------------------
    # Alpha compositing render (front‑to‑back)
    # ------------------------------------------------------------

    def render_gaussians(self, gaussians: List[Gaussian2D], shape: Tuple[int, int]) -> np.ndarray:
        """
        Render Gaussians using alpha compositing.
        For each Gaussian, compute density = exp(-0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ)).
        Contribution = color * (opacity * density), blended front‑to‑back.
        """
        h, w = shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack([xs, ys], axis=-1).astype(np.float32)   # (h, w, 2)

        image = np.zeros((h, w, 3), dtype=np.float32)
        alpha_acc = np.zeros((h, w), dtype=np.float32)            # accumulated opacity

        # Sort Gaussians by something? Not strictly required, but we can sort by mean x or y for consistency.
        # Here we just iterate in given order.
        for g in gaussians:
            # Compute Mahalanobis distance squared
            delta = coords - g.mean
            try:
                inv_cov = np.linalg.inv(g.cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(g.cov)

            # exponent = ∑_ij Δ_i (Σ⁻¹)_ij Δ_j
            # Using einsum for efficiency
            exponent = np.einsum('...i,ij,...j->...', delta, inv_cov, delta)   # (h, w)
            density = np.exp(-0.5 * exponent)                                  # (h, w)

            alpha = g.opacity * density
            alpha = np.clip(alpha, 0, 1)                                       # per‑pixel alpha

            # Front‑to‑back compositing
            # new_color = old_color + (1 - old_alpha) * alpha * color
            # new_alpha = old_alpha + (1 - old_alpha) * alpha
            for c in range(3):
                image[..., c] += (1.0 - alpha_acc) * alpha * g.color[c]
            alpha_acc += (1.0 - alpha_acc) * alpha
            # optional: clip alpha_acc to avoid numerical issues
            alpha_acc = np.clip(alpha_acc, 0, 1)

        return np.clip(image, 0, 1)

    # ------------------------------------------------------------
    # Error computation and point sampling
    # ------------------------------------------------------------

    def compute_error_map(self, original: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """Per‑pixel squared error averaged over channels."""
        return np.mean((original - rendered) ** 2, axis=-1)   # (h, w)

    def sample_new_points_from_error(self, error_map: np.ndarray, num_points: int) -> np.ndarray:
        """Sample pixel coordinates weighted by error map."""
        h, w = error_map.shape
        flat = error_map.flatten()
        flat = flat + 1e-6          # avoid zero probabilities
        probs = flat / flat.sum()
        indices = np.random.choice(h * w, size=num_points, replace=False, p=probs)
        ys, xs = np.divmod(indices, w)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    # ------------------------------------------------------------
    # Main iterative pipeline
    # ------------------------------------------------------------

    def run_pipeline(self) -> Tuple[List[Gaussian2D], np.ndarray]:
        img = self.load_image(self.image_path, self.config.target_size)
        h, w = img.shape[:2]

        # Prepare edge map (importance for initial sampling)
        _, edges = self.to_grayscale_with_edges(img)

        # Initial points from edge map
        points = self.select_important_points(edges, self.config.num_initial_points)

        # We'll maintain a set of points (unique coordinates) across iterations
        # Convert to integer coordinates for indexing colors/weights
        points_int = np.round(points).astype(int)
        points_int[:, 0] = np.clip(points_int[:, 0], 0, w - 1)
        points_int[:, 1] = np.clip(points_int[:, 1], 0, h - 1)

        # Initial colors and weights from the image and edge map
        colors = img[points_int[:, 1], points_int[:, 0]]
        weights = edges[points_int[:, 1], points_int[:, 0]]

        best_gaussians = []
        best_rendered = None
        best_error = float('inf')

        for iteration in range(self.config.max_iterations):
            # Build quadtree with current points
            bbox = (0, w, 0, h)
            gaussians = self.build_quadtree(img, edges, points, colors, weights, bbox)

            # Optionally split large Gaussians (density refinement)
            gaussians = self.split_large_gaussians(gaussians)

            # Render using alpha compositing
            rendered = self.render_gaussians(gaussians, (h, w))

            # Compute overall error
            error_map = self.compute_error_map(img, rendered)
            mse = np.mean(error_map)

            print(f"Iteration {iteration+1}: MSE = {mse:.6f}, #Gaussians = {len(gaussians)}")

            # Keep best result (lowest MSE)
            if mse < best_error:
                best_error = mse
                best_gaussians = gaussians
                best_rendered = rendered

            # Stop if error below threshold
            if mse < self.config.error_threshold:
                print("Error threshold reached.")
                break

            # Sample new points from error map and add to the set
            new_points = self.sample_new_points_from_error(error_map, self.config.points_per_iteration)
            new_points_int = np.round(new_points).astype(int)
            new_points_int[:, 0] = np.clip(new_points_int[:, 0], 0, w - 1)
            new_points_int[:, 1] = np.clip(new_points_int[:, 1], 0, h - 1)

            # Merge new points with existing ones (avoid duplicates by using a set of tuples)
            # For simplicity, we just concatenate; duplicates are okay but may waste computation.
            points = np.vstack([points, new_points])
            points_int = np.vstack([points_int, new_points_int])

            # Append corresponding colors and weights
            new_colors = img[new_points_int[:, 1], new_points_int[:, 0]]
            new_weights = edges[new_points_int[:, 1], new_points_int[:, 0]]
            colors = np.vstack([colors, new_colors])
            weights = np.hstack([weights, new_weights])

        self.gaussians = best_gaussians
        self.rendered = best_rendered
        return best_gaussians, best_rendered

    # ------------------------------------------------------------
    # Required Base Method
    # ------------------------------------------------------------

    def render(self, return_gaussians=False):
        gaussians, final = self.run_pipeline()
        if return_gaussians:
            return final, gaussians
        else:
            return final


import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
from UGen.encoder.algorithms.BaseAlgorithm import EncoderAlgorithms


@dataclass
class Gaussian2D:
    mean: np.ndarray      # (2,) float, pixel coordinates (x, y)
    cov: np.ndarray       # (2,2) float, covariance matrix
    color: np.ndarray     # (3,) float in [0,1]
    opacity: float = 1.0  # base opacity


@dataclass
class RecursiveDensifyConfig:
    target_size: Tuple[int, int] = (512, 512)
    num_initial_points: int = 5000          # dense sampling from edges
    var_threshold: float = 0.01              # stop when color variance < this
    max_points_per_leaf: int = 5
    min_bbox_size: int = 8
    cov_split_threshold: float = 5.0         # split Gaussians with max eigenvalue > this
    coverage_threshold: float = 0.5 
    num_fill_points: float = 500
    blur_coverage_threshold: float = 0.5
    blur_factor: float = 1.2
    min_contribution_ratio: float = 0.1


class RecursiveQuadtreeDensifyEncoder(EncoderAlgorithms):
    """
    Single‑pass recursive quadtree Gaussian splatting with alpha compositing.
    Subdivision is driven by color variance of points inside each cell.
    """

    def __init__(self, image_path: str, config: RecursiveDensifyConfig):
        self.image_path = image_path
        self.config = config
        self.gaussians: List[Gaussian2D] = []
        self.rendered = None

    def blur_gaussians_to_fill_gaps(self, gaussians: List[Gaussian2D], image: np.ndarray,
                                 coverage_threshold: float = 0.5,
                                 blur_factor: float = 1.2,
                                 min_contribution_ratio: float = 0.1) -> List[Gaussian2D]:
        """
        Increase the covariance of Gaussians that significantly contribute to under‑covered areas.
        
        Parameters
        ----------
        coverage_threshold : float
            Pixels with accumulated alpha below this are considered under‑covered.
        blur_factor : float
            Multiply the covariance matrix of selected Gaussians by this factor.
        min_contribution_ratio : float
            A Gaussian is selected if the fraction of its alpha contribution that falls
            in under‑covered pixels exceeds this ratio.
        """
        h, w = image.shape[:2]
    
        # 1. Render the alpha accumulation map and per‑Gaussian alpha maps (optional)
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack([xs, ys], axis=-1).astype(np.float32)
    
        # We'll store per‑Gaussian alpha contributions to compute ratios later
        gaussian_alphas = []  # list of (h, w) arrays for each Gaussian
        alpha_acc = np.zeros((h, w), dtype=np.float32)
    
        for g in gaussians:
            delta = coords - g.mean
            try:
                inv_cov = np.linalg.inv(g.cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(g.cov)
    
            exponent = np.einsum('...i,ij,...j->...', delta, inv_cov, delta)
            density = np.exp(-0.5 * exponent)
            alpha = g.opacity * density
            alpha = np.clip(alpha, 0, 1)
    
            gaussian_alphas.append(alpha.copy())
            alpha_acc += (1.0 - alpha_acc) * alpha
            alpha_acc = np.clip(alpha_acc, 0, 1)
    
        # 2. Identify under‑covered pixels
        low_mask = alpha_acc < coverage_threshold
    
        # If no under‑covered pixels, return unchanged
        if not np.any(low_mask):
            return gaussians
    
        # 3. For each Gaussian, compute contribution in low‑coverage area
        new_gaussians = []
        for g, g_alpha in zip(gaussians, gaussian_alphas):
            total_contrib = g_alpha.sum()
            low_contrib = g_alpha[low_mask].sum()
    
            # If the Gaussian contributes significantly to under‑covered pixels, blur it
            if total_contrib > 0 and (low_contrib / total_contrib) >= min_contribution_ratio:
                # Increase covariance (blur)
                new_cov = g.cov * blur_factor
                # Ensure it remains positive definite
                new_cov += np.eye(2) * 1e-6
                new_g = Gaussian2D(mean=g.mean.copy(), cov=new_cov,
                                   color=g.color.copy(), opacity=g.opacity)
            else:
                new_g = g
            new_gaussians.append(new_g)
    
        return new_gaussians
    
    def fill_gaps(self, gaussians: List[Gaussian2D], image: np.ndarray,
                  coverage_threshold: float = 0.5, num_fill_points: int = 500) -> List[Gaussian2D]:
        """
        Identify under‑covered pixels (alpha < threshold) and add new Gaussians there.
        """
        h, w = image.shape[:2]
    
        # 1. Render the alpha accumulation map using the existing Gaussians
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack([xs, ys], axis=-1).astype(np.float32)
    
        alpha_acc = np.zeros((h, w), dtype=np.float32)
    
        for g in gaussians:
            delta = coords - g.mean
            try:
                inv_cov = np.linalg.inv(g.cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(g.cov)
    
            exponent = np.einsum('...i,ij,...j->...', delta, inv_cov, delta)
            density = np.exp(-0.5 * exponent)
            alpha = g.opacity * density
            alpha = np.clip(alpha, 0, 1)
    
            alpha_acc += (1.0 - alpha_acc) * alpha
            alpha_acc = np.clip(alpha_acc, 0, 1)
    
        # 2. Find pixels where coverage is low
        low_coverage_mask = alpha_acc < coverage_threshold
        y_idxs, x_idxs = np.where(low_coverage_mask)
    
        if len(x_idxs) == 0:
            return gaussians  # nothing to fill
    
        # 3. Sample points from low‑coverage regions
        # Use a uniform distribution (or weight by error if desired)
        num_samples = min(num_fill_points, len(x_idxs))
        sample_indices = np.random.choice(len(x_idxs), size=num_samples, replace=False)
        fill_points = np.stack([x_idxs[sample_indices], y_idxs[sample_indices]], axis=1).astype(np.float32)
    
        # 4. Create new Gaussians with small isotropic covariance
        new_gaussians = []
        for pt in fill_points:
            mean = pt
            cov = np.eye(2) * 1.0  # 1 pixel standard deviation
            # Colour from the original image at that location
            color = image[int(pt[1]), int(pt[0])]  # (y, x)
            new_g = Gaussian2D(mean=mean, cov=cov, color=color, opacity=1.0)
            new_gaussians.append(new_g)
    
        # 5. Merge and return
        return gaussians + new_gaussians
                      
    # ------------------------------------------------------------
    # Helper functions (adapted from previous)
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

    def select_important_points(self, importance_map: np.ndarray, num_points: int) -> np.ndarray:
        """Sample pixel coordinates from a 2D importance map (treated as probability distribution)."""
        h, w = importance_map.shape
        flat = importance_map.flatten()
        flat = flat + 1e-6
        probs = flat / flat.sum()
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

    def fit_gaussian_to_region(self, points: np.ndarray, colors: np.ndarray, weights: np.ndarray) -> Gaussian2D:
        mean, cov = self.weighted_mean_cov(points, weights)
        total_weight = weights.sum()
        if total_weight > 0:
            avg_color = np.average(colors, axis=0, weights=weights)
        else:
            if len(colors) > 0:
                avg_color = np.mean(colors, axis=0)
            else:
                avg_color = np.array([0.5, 0.5, 0.5])
        return Gaussian2D(mean=mean, cov=cov, color=avg_color, opacity=1.0)

    def split_large_gaussians(self, gaussians: List[Gaussian2D]) -> List[Gaussian2D]:
        """Split any Gaussian whose covariance is too large."""
        refined = []
        for g in gaussians:
            try:
                eigvals, eigvecs = np.linalg.eigh(g.cov)
            except np.linalg.LinAlgError:
                eigvals = [0, 0]
            max_eig = max(eigvals)
            if max_eig > self.config.cov_split_threshold:
                direction = eigvecs[:, np.argmax(eigvals)]
                shift = np.sqrt(max_eig) * direction
                g1 = Gaussian2D(mean=g.mean + shift, cov=g.cov / 2,
                                color=g.color, opacity=g.opacity)
                g2 = Gaussian2D(mean=g.mean - shift, cov=g.cov / 2,
                                color=g.color, opacity=g.opacity)
                refined.extend([g1, g2])
            else:
                refined.append(g)
        return refined

    # ------------------------------------------------------------
    # Recursive quadtree builder (core of this encoder)
    # ------------------------------------------------------------

    def build_quadtree(self, image: np.ndarray, edge_map: np.ndarray,
                       points: np.ndarray, colors: np.ndarray, weights: np.ndarray,
                       bbox: Tuple[int, int, int, int]) -> List[Gaussian2D]:
        """
        Recursively build quadtree. At each node:
        - If the cell is small, has few points, or color variance is low, emit a Gaussian.
        - Otherwise, split into four children and recurse.
        """
        xmin, xmax, ymin, ymax = bbox
        mask = (points[:, 0] >= xmin) & (points[:, 0] < xmax) & \
               (points[:, 1] >= ymin) & (points[:, 1] < ymax)
        pts_in = points[mask]
        cols_in = colors[mask]
        w_in = weights[mask]

        # Leaf conditions
        if len(pts_in) <= self.config.max_points_per_leaf or \
           (xmax - xmin <= self.config.min_bbox_size and ymax - ymin <= self.config.min_bbox_size):
            if len(pts_in) > 0:
                g = self.fit_gaussian_to_region(pts_in, cols_in, w_in)
                # Optionally split if still too large
                return self.split_large_gaussians([g])
            else:
                return []

        # Compute color variance to decide subdivision
        if len(pts_in) > 0:
            color_var = np.var(cols_in, axis=0).mean()
            if color_var < self.config.var_threshold:
                g = self.fit_gaussian_to_region(pts_in, cols_in, w_in)
                return self.split_large_gaussians([g])
        # else variance high → subdivide

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

    # ------------------------------------------------------------
    # Alpha compositing render (same as before)
    # ------------------------------------------------------------

    def render_gaussians(self, gaussians: List[Gaussian2D], shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack([xs, ys], axis=-1).astype(np.float32)

        image = np.zeros((h, w, 3), dtype=np.float32)
        alpha_acc = np.zeros((h, w), dtype=np.float32)

        for g in gaussians:
            delta = coords - g.mean
            try:
                inv_cov = np.linalg.inv(g.cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(g.cov)

            exponent = np.einsum('...i,ij,...j->...', delta, inv_cov, delta)
            density = np.exp(-0.5 * exponent)
            alpha = g.opacity * density
            alpha = np.clip(alpha, 0, 1)

            for c in range(3):
                image[..., c] += (1.0 - alpha_acc) * alpha * g.color[c]
            alpha_acc += (1.0 - alpha_acc) * alpha
            alpha_acc = np.clip(alpha_acc, 0, 1)

        return np.clip(image, 0, 1)

    # ------------------------------------------------------------
    # Single‑pass pipeline
    # ------------------------------------------------------------
    
    def run_pipeline(self) -> Tuple[List[Gaussian2D], np.ndarray]:
        img = self.load_image(self.image_path, self.config.target_size)
        h, w = img.shape[:2]
    
        _, edges = self.to_grayscale_with_edges(img)
    
        points = self.select_important_points(edges, self.config.num_initial_points)
        points_int = np.round(points).astype(int)
        points_int[:, 0] = np.clip(points_int[:, 0], 0, w - 1)
        points_int[:, 1] = np.clip(points_int[:, 1], 0, h - 1)
    
        colors = img[points_int[:, 1], points_int[:, 0]]
        weights = edges[points_int[:, 1], points_int[:, 0]]
    
        bbox = (0, w, 0, h)
        gaussians = self.build_quadtree(img, edges, points, colors, weights, bbox)

        gaussians = self.blur_gaussians_to_fill_gaps(gaussians, img,
                                     self.config.blur_coverage_threshold,
                                     self.config.blur_factor,
                                     self.config.min_contribution_ratio)
    
        # --- Gap filling ---
        gaussians = self.fill_gaps(gaussians, img, self.config.coverage_threshold, self.config.num_fill_points)
    
        rendered = self.render_gaussians(gaussians, (h, w))
    
        self.gaussians = gaussians
        self.rendered = rendered
        return gaussians, rendered

    # ------------------------------------------------------------
    # Required Base Method
    # ------------------------------------------------------------

    def render(self, return_gaussians=False):
        gaussians, final = self.run_pipeline()
        if return_gaussians:
            return final, gaussians
        else:
            return final

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ------------------------------------------------------------
# Assume Gaussian2D is defined elsewhere, e.g.:
class Gaussian2D:
    def __init__(self, mean, cov, color, opacity):
       self.mean = mean
       self.cov = cov
       self.color = color
       self.opacity = opacity
# ------------------------------------------------------------

@dataclass
class RecursiveDensifyConfig:
    # Basic image and sampling parameters
    target_size: Tuple[int, int] = (512, 512)
    num_initial_points: int = 5000

    # Quadtree subdivision control
    var_threshold: float = 0.01
    max_points_per_leaf: int = 5
    min_bbox_size: int = 8

    # Gaussian splitting (after fitting)
    cov_split_threshold: float = 5.0

    # Gap‑filling (add new Gaussians)
    coverage_threshold: float = 0.5
    num_fill_points: int = 500

    # Gap‑filling by blurring existing Gaussians
    blur_coverage_threshold: float = 0.5
    blur_factor: float = 1.2
    min_contribution_ratio: float = 0.1


class RecursiveQuadtreeDensifyEncoder:
    """
    Modified encoder that records leaf cells during quadtree construction.
    """
    def __init__(self, image_path: str, config: RecursiveDensifyConfig):
        self.image_path = image_path
        self.config = config
        self.gaussians: List[Gaussian2D] = []
        self.rendered = None
        self.leaf_cells: List[Tuple[Tuple[int, int, int, int], Gaussian2D]] = []

    # ------------------------------------------------------------------
    # All helper methods (load_image, to_grayscale_with_edges, etc.)
    # are identical to the previous implementation. For brevity they are
    # omitted here, but must be included in the actual code.
    # See the previous answer for the full implementation.
    # ------------------------------------------------------------------

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
                  coverage_threshold: float = 0.5, num_fill_points: int = 500,
                  importance_map: Optional[np.ndarray] = None) -> List[Gaussian2D]:
        """
        Identify under‑covered pixels (alpha < threshold) and add new Gaussians there.
        If an importance_map is given, sampling is weighted by its values (e.g., edge magnitude),
        so that new Gaussians are placed preferentially in important regions.
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
        num_samples = min(num_fill_points, len(x_idxs))
    
        if importance_map is not None:
            # Get importance values at the low‑coverage pixels
            imp_values = importance_map[low_coverage_mask]
            # Avoid zero probabilities (add a tiny constant)
            imp_values = imp_values + 1e-6
            probs = imp_values / imp_values.sum()
            # Importance‑weighted sampling without replacement
            sample_indices = np.random.choice(len(x_idxs), size=num_samples, replace=False, p=probs)
        else:
            # Uniform random sampling (original behaviour)
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
        cov += np.eye(2) * 1e-6
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

    # ------------------------------------------------------------------
    # Modified quadtree builder – records leaf cells
    # ------------------------------------------------------------------
    def build_quadtree(self, image, edge_map, points, colors, weights, bbox):
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
                self.leaf_cells.append((bbox, g))
                return self.split_large_gaussians([g])
            return []

        # Color variance check
        if len(pts_in) > 0:
            color_var = np.var(cols_in, axis=0).mean()
            if color_var < self.config.var_threshold:
                g = self.fit_gaussian_to_region(pts_in, cols_in, w_in)
                self.leaf_cells.append((bbox, g))
                return self.split_large_gaussians([g])

        # Subdivide
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

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------
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
        self.leaf_cells = []  # reset
        gaussians = self.build_quadtree(img, edges, points, colors, weights, bbox)

        gaussians = self.blur_gaussians_to_fill_gaps(gaussians, img,
                                     self.config.blur_coverage_threshold,
                                     self.config.blur_factor,
                                     self.config.min_contribution_ratio)

        gaussians = self.fill_gaps(gaussians, img,
                                   self.config.coverage_threshold,
                                   self.config.num_fill_points,
                                   importance_map=edges)

        rendered = self.render_gaussians(gaussians, (h, w))

        self.gaussians = gaussians
        self.rendered = rendered
        return gaussians, rendered

    def render(self, return_gaussians=False):
        gaussians, final = self.run_pipeline()
        if return_gaussians:
            return final, gaussians
        return final


# ------------------------------------------------------------
# Separate Visualizer Class
# ------------------------------------------------------------
class RecursiveQuadtreeVisualizer:
    """
    Visualizer for the recursive quadtree densification algorithm.
    Runs the pipeline once and stores intermediate data for plotting.
    """
    def __init__(self, image_path: str, config: RecursiveDensifyConfig):
        self.config = config
        self.encoder = RecursiveQuadtreeDensifyEncoder(image_path, config)
        # Run the pipeline to populate encoder.gaussians, encoder.leaf_cells, etc.
        self.gaussians, self.rendered = self.encoder.run_pipeline()
        # Load original image (already resized)
        self.image = self.encoder.load_image(image_path, config.target_size)
        self.h, self.w = self.image.shape[:2]
        # Extract initial samples and edge map
        _, self.edges = self.encoder.to_grayscale_with_edges(self.image)
        self.initial_points = self.encoder.select_important_points(
            self.edges, config.num_initial_points
        )

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_initial_samples(self, ax, color='red', s=1, alpha=0.5):
        """Scatter plot of the initial Sobel‑sampled points."""
        ax.scatter(self.initial_points[:, 0], self.initial_points[:, 1],
                   c=color, s=s, alpha=alpha, label='Initial samples')

    def plot_quadtree_cells(self, ax, edgecolor='blue', linewidth=1, alpha=0.3):
        """Draw rectangles for each quadtree leaf cell."""
        for (xmin, xmax, ymin, ymax), _ in self.encoder.leaf_cells:
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=linewidth, edgecolor=edgecolor,
                             facecolor='none', alpha=alpha)
            ax.add_patch(rect)

    def plot_gaussians(self, ax, which='final',
                       facecolor_from_color=True, edgecolor='none', alpha=0.6):
        """
        Plot Gaussians as ellipses.
        which : 'leaf' for the Gaussians stored with leaf cells (before splitting),
                'final' for the final Gaussians after post‑processing.
        """
        if which == 'leaf':
            gaussians = [g for (_, g) in self.encoder.leaf_cells]
        elif which == 'final':
            gaussians = self.gaussians
        else:
            raise ValueError("which must be 'leaf' or 'final'")

        for g in gaussians:
            width, height, angle = self._cov_to_ellipse_params(g.cov)
            color = g.color if facecolor_from_color else 'gray'
            ellipse = Ellipse(xy=(g.mean[0], g.mean[1]),
                              width=width, height=height, angle=angle,
                              facecolor=color, edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(ellipse)

    def _cov_to_ellipse_params(self, cov):
        """Convert 2x2 covariance to ellipse width, height, angle (degrees)."""
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            eigvals = [1.0, 1.0]
            eigvecs = np.eye(2)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        width = 2.0 * np.sqrt(eigvals[0])
        height = 2.0 * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        return width, height, angle

    def plot_rendered(self, ax):
        """Show the final rendered image."""
        ax.imshow(self.rendered)

    def plot_original(self, ax):
        """Show the original image."""
        ax.imshow(self.image)

    # ------------------------------------------------------------------
    # Multi‑panel figure
    # ------------------------------------------------------------------
    def create_figure(self, rows=2, cols=3, figsize=(15, 10),
                      show_original=True, show_initial=True,
                      show_quad=True, show_leaf_gaussians=False,
                      show_final_gaussians=True, show_rendered=True):
        """
        Create a figure with selected subplots.
        Returns the figure and a list of axes.
        """
        # Determine which panels to show
        panels = []
        if show_original:
            panels.append('original')
        if show_initial:
            panels.append('initial')
        if show_quad:
            panels.append('quad')
        if show_leaf_gaussians:
            panels.append('leaf_gauss')
        if show_final_gaussians:
            panels.append('final_gauss')
        if show_rendered:
            panels.append('rendered')

        n = len(panels)
        if n == 0:
            raise ValueError("No panels selected")

        # Create grid and flatten axes
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Hide unused axes
        for i in range(n, len(axes)):
            axes[i].axis('off')

        # Plot each panel
        for i, name in enumerate(panels):
            ax = axes[i]
            if name == 'original':
                self.plot_original(ax)
                ax.set_title('Original Image')
            elif name == 'initial':
                self.plot_original(ax)      # background
                self.plot_initial_samples(ax)
                ax.set_title('Initial Samples (Sobel)')
            elif name == 'quad':
                self.plot_original(ax)
                self.plot_quadtree_cells(ax)
                ax.set_title('Quadtree Leaf Cells')
            elif name == 'leaf_gauss':
                self.plot_original(ax)
                self.plot_gaussians(ax, which='leaf')
                ax.set_title('Leaf Gaussians')
            elif name == 'final_gauss':
                self.plot_original(ax)
                self.plot_gaussians(ax, which='final')
                ax.set_title('Final Gaussians')
            elif name == 'rendered':
                self.plot_rendered(ax)
                ax.set_title('Rendered Image')

            # Set common limits and aspect
            ax.set_xlim(0, self.w)
            ax.set_ylim(self.h, 0)
            ax.set_aspect('equal')

        plt.tight_layout()
        return fig, axes

    def show(self, **kwargs):
        """Create and display the figure with the given keyword arguments."""
        fig, _ = self.create_figure(**kwargs)
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import warnings

# ----------------------------------------------------------------------
# Metric calculation (all numpy, no deep learning)
# ----------------------------------------------------------------------
class Metrics:
    @staticmethod
    def mse(gt, pred):
        """Mean squared error"""
        return np.mean((gt - pred) ** 2)

    @staticmethod
    def rmse(gt, pred):
        """Root mean squared error"""
        return np.sqrt(Metrics.mse(gt, pred))

    @staticmethod
    def psnr(gt, pred, data_range=None):
        """Peak signal‑to‑noise ratio. If data_range not given, use max(gt) - min(gt)."""
        if data_range is None:
            data_range = gt.max() - gt.min()
        return psnr(gt, pred, data_range=data_range)

    @staticmethod
    def mae(gt, pred):
        """Mean absolute error"""
        return np.mean(np.abs(gt - pred))

    @staticmethod
    def max_error(gt, pred):
        """Maximum absolute error"""
        return np.max(np.abs(gt - pred))

    @staticmethod
    def relative_error(gt, pred, epsilon=1e-10):
        """Mean relative error (avoid division by zero)"""
        return np.mean(np.abs(gt - pred) / (np.abs(gt) + epsilon))

    @staticmethod
    def ssim(gt, pred, data_range=None, multichannel=True):
        """Structural Similarity Index (uses skimage)"""
        if data_range is None:
            data_range = gt.max() - gt.min()
        # Handle grayscale vs RGB
        if gt.ndim == 3 and gt.shape[2] == 3:
            channel_axis = 2
        else:
            multichannel = False
            channel_axis = None
        return ssim(gt, pred, data_range=data_range, channel_axis=channel_axis)

    @staticmethod
    def gradient_magnitude_similarity(gt, pred):
        """Gradient magnitude similarity (simple version)"""
        from scipy.ndimage import sobel
        # Compute gradients
        if gt.ndim == 3:
            # Convert to luminance for simplicity
            gt_gray = 0.299 * gt[...,0] + 0.587 * gt[...,1] + 0.114 * gt[...,2]
            pred_gray = 0.299 * pred[...,0] + 0.587 * pred[...,1] + 0.114 * pred[...,2]
        else:
            gt_gray = gt
            pred_gray = pred
        grad_gt = np.hypot(sobel(gt_gray, axis=0), sobel(gt_gray, axis=1))
        grad_pred = np.hypot(sobel(pred_gray, axis=0), sobel(pred_gray, axis=1))
        return Metrics.ssim(grad_gt, grad_pred, data_range=None)

    @staticmethod
    def all_metrics(gt, pred):
        """Compute all metrics and return a dictionary"""
        # Ensure images are float in [0,1] or similar
        gt = gt.astype(np.float32)
        pred = pred.astype(np.float32)
        data_range = gt.max() - gt.min()
        return {
            'MSE': Metrics.mse(gt, pred),
            'RMSE': Metrics.rmse(gt, pred),
            'PSNR': Metrics.psnr(gt, pred, data_range),
            'MAE': Metrics.mae(gt, pred),
            'Max Error': Metrics.max_error(gt, pred),
            'Relative Error': Metrics.relative_error(gt, pred),
            'SSIM': Metrics.ssim(gt, pred, data_range),
            'Gradient Similarity': Metrics.gradient_magnitude_similarity(gt, pred)
        }

# ----------------------------------------------------------------------
# Visualisation tools
# ----------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def side_by_side(gt, pred, title_left='Ground Truth', title_right='Rendered', figsize=(10,5)):
        """Show ground truth and rendered image side by side."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(gt, cmap='gray' if gt.ndim==2 else None)
        axes[0].set_title(title_left)
        axes[0].axis('off')
        axes[1].imshow(pred, cmap='gray' if pred.ndim==2 else None)
        axes[1].set_title(title_right)
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def overlay(gt, pred, alpha=0.6, cmap='viridis', figsize=(6,6)):
        """Overlay the rendered image on top of ground truth using a heatmap (difference) or blend."""
        diff = np.abs(gt - pred)
        # For RGB, convert difference to luminance for overlay
        if diff.ndim == 3:
            diff = np.mean(diff, axis=2)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(diff, cmap=cmap, alpha=alpha)
        plt.colorbar(im, ax=ax, label='Absolute difference')
        ax.set_title('Overlay (absolute error)')
        ax.axis('off')
        plt.show()

    @staticmethod
    def error_maps(gt, pred, figsize=(15,4)):
        """Show absolute, squared, and relative error maps side by side."""
        gt = gt.astype(np.float32)
        pred = pred.astype(np.float32)
        abs_err = np.abs(gt - pred)
        sq_err = (gt - pred) ** 2
        rel_err = np.abs(gt - pred) / (np.abs(gt) + 1e-10)

        # For RGB, take mean across channels for a single map
        if gt.ndim == 3:
            abs_err = np.mean(abs_err, axis=2)
            sq_err = np.mean(sq_err, axis=2)
            rel_err = np.mean(rel_err, axis=2)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        im0 = axes[0].imshow(abs_err, cmap='hot')
        axes[0].set_title('Absolute error')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(sq_err, cmap='hot')
        axes[1].set_title('Squared error')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(rel_err, cmap='hot', vmin=0, vmax=0.5)  # clamp for visibility
        axes[2].set_title('Relative error')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def error_histogram(gt, pred, bins=50, figsize=(6,4)):
        """Histogram of pixel‑wise absolute errors."""
        abs_err = np.abs(gt - pred).ravel()
        plt.figure(figsize=figsize)
        plt.hist(abs_err, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Absolute error')
        plt.ylabel('Frequency')
        plt.title('Error histogram')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def metric_comparison(metrics_dict, figsize=(10,5)):
        """
        metrics_dict: dict {algorithm_name: {metric_name: value}}
        Creates a grouped bar chart comparing metrics across algorithms.
        """
        algos = list(metrics_dict.keys())
        metric_names = list(metrics_dict[algos[0]].keys())

        # Normalise metrics for better visualisation (optional)
        # For now, just plot raw values (they have different scales)
        x = np.arange(len(metric_names))
        width = 0.8 / len(algos)

        fig, ax = plt.subplots(figsize=figsize)
        for i, algo in enumerate(algos):
            values = [metrics_dict[algo][m] for m in metric_names]
            ax.bar(x + i*width - 0.4 + width/2, values, width, label=algo)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_ylabel('Metric value')
        ax.set_title('Metric comparison across algorithms')
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def loss_drop(iterations_data, figsize=(8,5)):
        """
        iterations_data: list of dicts [{'name': algo, 'iterations': list, 'loss': list}]
        Plots loss vs iteration for multiple runs.
        """
        plt.figure(figsize=figsize)
        for data in iterations_data:
            plt.plot(data['iterations'], data['loss'], marker='o', label=data['name'])
        plt.xlabel('Iteration / number of splats')
        plt.ylabel('Loss (e.g., MSE)')
        plt.title('Loss drop over iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ----------------------------------------------------------------------
# Main evaluator class
# ----------------------------------------------------------------------
class GaussianSplattingEvaluator:
    def __init__(self, ground_truth):
        """
        ground_truth: numpy array (H, W) or (H, W, 3) representing the reference image.
                      Values are assumed to be in [0,1] or [0,255] (will be normalised).
        """
        self.gt = self._normalize(ground_truth)
        self.results = {}  # algo_name -> {'rendered': image, 'metrics': dict}

    def _normalize(self, img):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        return img

    def add_algorithm_result(self, name, rendered_image):
        """
        Add a rendered image produced by an algorithm.
        rendered_image: numpy array, same shape as ground truth.
        """
        rendered = self._normalize(rendered_image)
        metrics = Metrics.all_metrics(self.gt, rendered)
        self.results[name] = {
            'rendered': rendered,
            'metrics': metrics
        }
        print(f"Added result for '{name}'")
        return metrics

    def print_statistics(self, algo_name=None):
        """Print detailed statistics for one or all algorithms."""
        if algo_name is not None:
            names = [algo_name]
        else:
            names = self.results.keys()

        for name in names:
            if name not in self.results:
                print(f"Warning: {name} not found.")
                continue
            data = self.results[name]
            print(f"\n--- {name} ---")
            for k, v in data['metrics'].items():
                print(f"{k:20}: {v:.6f}")
            # Additional pixel error stats
            err = np.abs(self.gt - data['rendered']).ravel()
            print(f"{'Mean abs err':20}: {np.mean(err):.6f}")
            print(f"{'Std abs err':20}: {np.std(err):.6f}")
            print(f"{'Min abs err':20}: {np.min(err):.6f}")
            print(f"{'Max abs err':20}: {np.max(err):.6f}")

    def compare_algorithms(self):
        """Bar chart comparing metrics across all added algorithms."""
        if not self.results:
            print("No results to compare.")
            return
        metrics_dict = {name: data['metrics'] for name, data in self.results.items()}
        Visualizer.metric_comparison(metrics_dict)

    def visualize(self, algo_name, show=['side', 'overlay', 'errormap', 'hist']):
        """
        Generate various visualisations for a specific algorithm.
        show: list containing any of 'side', 'overlay', 'errormap', 'hist'
        """
        if algo_name not in self.results:
            print(f"Algorithm '{algo_name}' not found.")
            return
        rendered = self.results[algo_name]['rendered']
        if 'side' in show:
            Visualizer.side_by_side(self.gt, rendered, title_right=algo_name)
        if 'overlay' in show:
            Visualizer.overlay(self.gt, rendered)
        if 'errormap' in show:
            Visualizer.error_maps(self.gt, rendered)
        if 'hist' in show:
            Visualizer.error_histogram(self.gt, rendered)

    def loss_drop_plot(self, iterations_data):
        """
        iterations_data: list of dicts with keys 'name', 'iterations', 'loss'
        This method does not depend on stored results; it's a convenience wrapper.
        """
        Visualizer.loss_drop(iterations_data)

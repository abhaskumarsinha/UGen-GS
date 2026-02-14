from UGen.encoder.evaluate_2D_enc import *

class BatchEvaluator:
    """
    Evaluate multiple algorithms on a set of ground‑truth images.
    """
    def __init__(self, ground_truths, image_names=None):
        """
        ground_truths : list of numpy arrays (H,W) or (H,W,3)
        image_names   : list of strings, optional
        """
        self.gt_list = [self._normalize(gt) for gt in ground_truths]
        self.num_images = len(self.gt_list)
        if image_names is None:
            self.image_names = [f"img_{i}" for i in range(self.num_images)]
        else:
            self.image_names = image_names
        # results[algo] = list of rendered arrays (same order as gt_list)
        self.results = {}  # algo -> list of rendered images
        self.metrics = {}  # will be computed lazily

    def _normalize(self, img):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        return img

    def add_algorithm(self, name, rendered_list):
        """
        rendered_list : list of numpy arrays, same length as ground_truths and same order.
        """
        if len(rendered_list) != self.num_images:
            raise ValueError(f"Number of rendered images ({len(rendered_list)}) does not match number of ground truths ({self.num_images})")
        self.results[name] = [self._normalize(r) for r in rendered_list]
        # Invalidate previously computed metrics
        self.metrics.pop(name, None)

    def _compute_metrics_for_algo(self, algo):
        """Compute metrics for one algorithm across all images."""
        metrics_list = []
        for i in range(self.num_images):
            gt = self.gt_list[i]
            pred = self.results[algo][i]
            m = Metrics.all_metrics(gt, pred)
            metrics_list.append(m)
        return metrics_list

    def compute_all_metrics(self):
        """Ensure metrics are computed for all algorithms."""
        for algo in self.results:
            if algo not in self.metrics:
                self.metrics[algo] = self._compute_metrics_for_algo(algo)

    def get_metrics_dataframe(self):
        """
        Return a pandas DataFrame with multi-index (image_name, algorithm) and metric columns.
        """
        import pandas as pd
        self.compute_all_metrics()
        rows = []
        for algo, mlist in self.metrics.items():
            for img_name, m in zip(self.image_names, mlist):
                row = {'image': img_name, 'algorithm': algo}
                row.update(m)
                rows.append(row)
        df = pd.DataFrame(rows)
        df.set_index(['image', 'algorithm'], inplace=True)
        return df

    def summary_statistics(self):
        """
        For each algorithm, compute mean, std, min, max of each metric across images.
        Returns a dict: {algo: {metric: {'mean':..., 'std':...}}}
        """
        self.compute_all_metrics()
        summary = {}
        for algo, mlist in self.metrics.items():
            # mlist is list of dicts (one per image)
            # collect values per metric
            metric_names = mlist[0].keys()
            stats = {}
            for met in metric_names:
                vals = [m[met] for m in mlist]
                stats[met] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'min': np.min(vals),
                    'max': np.max(vals)
                }
            summary[algo] = stats
        return summary

    def print_summary(self):
        """Pretty print summary statistics."""
        summary = self.summary_statistics()
        for algo, stats in summary.items():
            print(f"\n--- {algo} ---")
            for met, vals in stats.items():
                print(f"{met:20}: mean={vals['mean']:.6f} ± {vals['std']:.6f} "
                      f"[{vals['min']:.6f}, {vals['max']:.6f}]")

    def plot_summary_bars(self, metric='PSNR', figsize=(10,5)):
        """
        Bar chart of the mean value of a chosen metric for each algorithm, with error bars (std).
        """
        summary = self.summary_statistics()
        algos = list(summary.keys())
        means = [summary[a][metric]['mean'] for a in algos]
        stds  = [summary[a][metric]['std'] for a in algos]

        plt.figure(figsize=figsize)
        x = np.arange(len(algos))
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xticks(x, algos)
        plt.ylabel(metric)
        plt.title(f'Average {metric} across images (with std)')
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    def plot_boxplots(self, metrics=None, figsize=(12,6)):
        """
        Box plots comparing algorithms for one or more metrics.
        metrics: list of metric names, or None for all.
        """
        self.compute_all_metrics()
        if metrics is None:
            # take first algo's keys
            first_algo = next(iter(self.metrics))
            metrics = list(self.metrics[first_algo][0].keys())
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize, squeeze=False)
        for idx, met in enumerate(metrics):
            data = []
            algos = list(self.metrics.keys())
            for algo in algos:
                vals = [m[met] for m in self.metrics[algo]]
                data.append(vals)
            axes[0, idx].boxplot(data, labels=algos)
            axes[0, idx].set_title(met)
            axes[0, idx].set_ylabel(met)
            axes[0, idx].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_image(self, image_index, algorithm_names=None, show=['side', 'overlay', 'errormap', 'hist']):
        """
        For a specific image, show visualizations comparing ground truth and one or more algorithms.
        algorithm_names: list of algorithm names; if None, show all.
        """
        gt = self.gt_list[image_index]
        if algorithm_names is None:
            algos = list(self.results.keys())
        else:
            algos = [a for a in algorithm_names if a in self.results]
        for algo in algos:
            pred = self.results[algo][image_index]
            print(f"\n--- Visualizing {algo} for image {self.image_names[image_index]} ---")
            # Reuse existing Visualizer functions
            if 'side' in show:
                Visualizer.side_by_side(gt, pred, title_right=f"{algo} - {self.image_names[image_index]}")
            if 'overlay' in show:
                Visualizer.overlay(gt, pred)
            if 'errormap' in show:
                Visualizer.error_maps(gt, pred)
            if 'hist' in show:
                Visualizer.error_histogram(gt, pred)

    def loss_drop_plot(self, iterations_data):
        """
        iterations_data: list of dicts with keys 'name', 'iterations', 'loss'
        (Same as before, just a pass‑through to Visualizer)
        """
        Visualizer.loss_drop(iterations_data)

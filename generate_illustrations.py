#!/usr/bin/env python3

import argparse
import os
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

# =========================
# UGen Imports
# =========================
from UGen.encoder import *
from UGen.encoder.algorithms import *
from UGen.encoder.algorithms.quadtree_visualizer import *
from UGen.visualizer import *
from UGen.utils.point_cloud import *
from UGen.decoder import *
from UGen.utils.unproj_gaussians import *
from UGen.utils.colmap_cameras import *
from UGen.renderer import *


# =========================
# Argument Parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Recursive Quadtree Gaussian Splatting experiments"
    )

    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image")

    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save outputs")

    parser.add_argument("--n_runs", type=int, default=20,
                        help="Number of experiment runs")

    parser.add_argument("--width", type=int, default=160,
                        help="Resize width")

    parser.add_argument("--height", type=int, default=120,
                        help="Resize height")

    return parser.parse_args()


# =========================
# Experiment Runner
# =========================
def run_multiple_experiments(
    n_runs,
    input_image_path,
    quadtreeden_cfg,
    y0
):
    all_results = []
    metrics_storage = defaultdict(list)

    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")

        enc = RecursiveQuadtreeDensifyEncoder(
            input_image_path,
            quadtreeden_cfg
        )

        # ✅ UPDATED LINE
        rendered, gaussians = enc.render(return_gaussians=True)

        gaussian_count = len(gaussians)

        gs_eval = GaussianSplattingEvaluator(y0)

        expt_name = str(uuid.uuid4())[:8]
        result = gs_eval.add_algorithm_result(expt_name, rendered)

        # ✅ Add Gaussian count to result dictionary
        result["gaussian_count"] = gaussian_count

        all_results.append(result)

        # Store metrics
        for k, v in result.items():
            metrics_storage[k].append(float(v))

    # ---- Compute mean + std ----
    summary = {}

    for metric, values in metrics_storage.items():
        values = np.array(values)
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    return summary, all_results


# =========================
# Save CSV
# =========================
def save_csv(summary, all_results, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    df_runs = pd.DataFrame(all_results)
    df_runs.to_csv(os.path.join(output_folder, "per_run_results.csv"), index=False)

    summary_rows = []
    for metric, stats in summary.items():
        summary_rows.append({
            "Metric": metric,
            "Mean": stats["mean"],
            "Std": stats["std"]
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_folder, "summary.csv"), index=False)

    print("CSV files saved.")


# =========================
# Main
# =========================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load & preprocess image
    y0 = Image.open(args.input_image).convert("RGB")
    y0 = y0.resize((args.width, args.height), Image.LANCZOS)
    y0 = np.asarray(y0).astype(np.float32) / 255.0

    # Quadtree config
    quadtreeden_cfg = RecursiveDensifyConfig(
        (args.width, args.height),
        10000,
        min_bbox_size=2,
        var_threshold=0.0021,
        max_points_per_leaf=7,
        cov_split_threshold=4.05,
        coverage_threshold=0.412,
        num_fill_points=1209,
        blur_coverage_threshold=0.892,
        blur_factor=7,
        min_contribution_ratio=0.268
    )

    # Visualization
    print("Saving visualization outputs...")
    y0_viz = RecursiveQuadtreeVisualizer(args.input_image, quadtreeden_cfg)

    y0_viz.save_all(
        base_name="my_image",
        output_dir=args.output_dir,
        width_pixels=2070,
        dpi=600,
        sobel_point_color='red',
        sobel_point_size=1,
        cell_color='blue',
        cell_linewidth=1,
        gaussians_fill=False,
        gaussians_edgecolor='white',
        gaussians_alpha=1.0
    )

    print(f"Saved visualization outputs to: {args.output_dir}")

    # Run experiments
    summary, all_results = run_multiple_experiments(
        n_runs=args.n_runs,
        input_image_path=args.input_image,
        quadtreeden_cfg=quadtreeden_cfg,
        y0=y0
    )

    # Save CSV
    save_csv(summary, all_results, args.output_dir)

    print("\n===== Summary =====")
    for metric, stats in summary.items():
        print(f"{metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import os
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from UGen.encoder import *
from UGen.encoder.algorithms import *
from UGen.encoder.algorithms.quadtree_visualizer import *
from UGen.visualizer import *
from UGen.decoder import *
from UGen.renderer import *


# =========================
# LoD Configurations
# =========================
def get_lod_configs(width, height):

    return {

        "low": {
            "target_points": 800,
            "min_bbox_size": 6,
            "var_threshold": 2.94e-05,
            "max_points_per_leaf": 9,
            "cov_split_threshold": 3.753023113,
            "coverage_threshold": 0.48833341,
            "num_fill_points": 200,
            "blur_coverage_threshold": 0.964467743,
            "blur_factor": 10,
            "min_contribution_ratio": 0.185161798
        },

        "medium": {
            "target_points": 2500,
            "min_bbox_size": 2,
            "var_threshold": 0.001600925,
            "max_points_per_leaf": 7,
            "cov_split_threshold": 4.163973149,
            "coverage_threshold": 0.418701411,
            "num_fill_points": 500,
            "blur_coverage_threshold": 0.569876154,
            "blur_factor": 9,
            "min_contribution_ratio": 0.265283058
        },

        "high": {
            "target_points": 5000,
            "min_bbox_size": 2,
            "var_threshold": 0.002130755,
            "max_points_per_leaf": 7,
            "cov_split_threshold": 4.054542572,
            "coverage_threshold": 0.412878369,
            "num_fill_points": 609,
            "blur_coverage_threshold": 0.892973704,
            "blur_factor": 9,
            "min_contribution_ratio": 0.268658287
        }
    }


# =========================
# Experiment Runner
# =========================
def run_trials(lod_name, cfg_dict, n_trials, input_image_path, y0, width, height):

    print(f"\n===== Running {lod_name.upper()} LoD =====")

    quad_cfg = RecursiveDensifyConfig(
        (width, height),
        cfg_dict["target_points"],
        min_bbox_size=cfg_dict["min_bbox_size"],
        var_threshold=cfg_dict["var_threshold"],
        max_points_per_leaf=cfg_dict["max_points_per_leaf"],
        cov_split_threshold=cfg_dict["cov_split_threshold"],
        coverage_threshold=cfg_dict["coverage_threshold"],
        num_fill_points=cfg_dict["num_fill_points"],
        blur_coverage_threshold=cfg_dict["blur_coverage_threshold"],
        blur_factor=cfg_dict["blur_factor"],
        min_contribution_ratio=cfg_dict["min_contribution_ratio"]
    )

    all_results = []
    metrics_storage = defaultdict(list)

    for i in range(n_trials):
        print(f"{lod_name} Run {i+1}/{n_trials}")

        enc = RecursiveQuadtreeDensifyEncoder(
            input_image_path,
            quad_cfg
        )

        rendered, gaussians = enc.render(return_gaussians=True)
        gaussian_count = len(gaussians)

        evaluator = GaussianSplattingEvaluator(y0)
        expt_name = f"{lod_name}_{str(uuid.uuid4())[:8]}"

        result = evaluator.add_algorithm_result(expt_name, rendered)
        result["gaussian_count"] = gaussian_count
        result["lod"] = lod_name

        all_results.append(result)

        for k, v in result.items():
            if isinstance(v, (int, float, np.number)):
                metrics_storage[k].append(float(v))

    summary = {}
    for metric, values in metrics_storage.items():
        values = np.array(values)
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

    return summary, all_results


# =========================
# Main
# =========================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_dir", default="./lod_results")
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load image
    y0 = Image.open(args.input_image).convert("RGB")
    y0 = y0.resize((args.width, args.height), Image.LANCZOS)
    y0 = np.asarray(y0).astype(np.float32) / 255.0

    lod_configs = get_lod_configs(args.width, args.height)

    combined_results = []
    combined_summary_rows = []

    for lod_name, cfg in lod_configs.items():

        summary, results = run_trials(
            lod_name,
            cfg,
            args.n_trials,
            args.input_image,
            y0,
            args.width,
            args.height
        )

        # Save per-LoD CSV
        df_runs = pd.DataFrame(results)
        df_runs.to_csv(
            os.path.join(args.output_dir, f"{lod_name}_per_run.csv"),
            index=False
        )

        # Save summary
        for metric, stats in summary.items():
            combined_summary_rows.append({
                "lod": lod_name,
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"]
            })

        combined_results.extend(results)

    # Save combined summary
    df_summary = pd.DataFrame(combined_summary_rows)
    df_summary.to_csv(
        os.path.join(args.output_dir, "lod_summary.csv"),
        index=False
    )

    print("\n===== FINAL SUMMARY =====")
    print(df_summary)


if __name__ == "__main__":
    main()

# optuna_gs_opt.py

import os
import json
import csv
import uuid
import traceback
import datetime
from datetime import timezone
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import optuna

from UGen.encoder import *
from UGen.encoder.algorithms import *

from UGen.encoder import *
from UGen.encoder.algorithms import *
from UGen.sfm import *
from UGen.visualizer import *
from UGen.BA import *
from UGen.utils.unproject_gaussians import *
from UGen.utils.point_cloud import *
from UGen.decoder import *

import uuid

# ---------------------------------------------------------
# CLI PARSER
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for Gaussian Splatting Pipeline",
        add_help=True
    )

    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="optuna_gs_logs")
    parser.add_argument("--study_name", type=str, default="gaussian_splatting_mse")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_gs.db")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_images", action="store_true", default=True)

    # 🔥 KEY FIX: ignore unknown args injected by Jupyter/Colab
    args, unknown = parser.parse_known_args()

    return args


# ---------------------------------------------------------
# DATA + FIXED ENCODER CONFIG (UNCHANGED)
# ---------------------------------------------------------

IMG_PATHS = [
    "/content/UGen-GS/BatchedImages/0.png",
    "/content/UGen-GS/BatchedImages/1.png",
    "/content/UGen-GS/BatchedImages/2.png",
    "/content/UGen-GS/BatchedImages/3.png",
]

quadtreeden_cfg = RecursiveDensifyConfig((160, 120),
                                         5000,
                                         min_bbox_size=2,
                                         var_threshold=0.0021,
                                         max_points_per_leaf = 7,
                                         cov_split_threshold = 4.05,
                                         coverage_threshold = 0.412,
                                         num_fill_points = 609,
                                         blur_coverage_threshold = 0.89,
                                         blur_factor = 9,
                                         min_contribution_ratio = 0.268)


def load_and_preprocess_image(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((160, 120), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    args = parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)

    TRIAL_JSON_DIR = OUT_DIR / "trials_json"
    TRIAL_JSON_DIR.mkdir(exist_ok=True)

    TRIAL_IMG_DIR = OUT_DIR / "trial_renders"
    TRIAL_IMG_DIR.mkdir(exist_ok=True)

    SUMMARY_CSV = OUT_DIR / "summary.csv"

    GT_IMAGES = [load_and_preprocess_image(p) for p in IMG_PATHS]

    def append_summary_row(row_dict):
        write_header = not SUMMARY_CSV.exists()
        with open(SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row_dict)
    
    # lol
    def make_json_serializable(obj):
        """
        Recursively convert numpy types and other non-json-native objects
        into Python primitives (list, dict, int, float, str).
        """
        # numpy scalar
        if isinstance(obj, (np.generic,)):
            return obj.item()
        # numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # basic python types (safe)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # dict -> convert keys and values
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                # ensure keys are strings
                new[str(k)] = make_json_serializable(v)
            return new
        # lists/tuples/sets -> list
        if isinstance(obj, (list, tuple, set)):
            return [make_json_serializable(v) for v in obj]
        # objects with __dict__: try to convert
        if hasattr(obj, "__dict__"):
            return make_json_serializable(vars(obj))
        # fallback: string representation
        return str(obj)

    # ---------------------------------------------------------
    # OPTUNA OBJECTIVE
    # ---------------------------------------------------------

    def objective(trial):

        trial_id = trial.number
        ts = datetime.datetime.now(timezone.utc).isoformat()
        trial_uid = f"{trial_id}_{str(uuid.uuid4())[:8]}"

        # Hyperparameters to optimize
        ratio_thresh = trial.suggest_float("ratio_thresh", 0.0, 100.0)
        ransac_thresh = trial.suggest_float("ransac_thresh", 0.0, 100.0)
        n_neighbors = trial.suggest_int("n_neighbors", 0, 100)
        position_weight = trial.suggest_float("position_weight", 0.0, 1.0)

        k_neighbors = trial.suggest_int("k_neighbors", 1, 20)
        opacity_threshold_factor = trial.suggest_float("opacity_threshold_factor", 0.0, 10.0)

        eps = 1e-6

        try:
            # Encoders
            encoders = [
                RecursiveQuadtreeDensifyEncoder(p, quadtreeden_cfg)
                for p in IMG_PATHS
            ]

            renders_init = [enc.render(return_gaussians=True) for enc in encoders]
            y_preds = [x[0] for x in renders_init]
            gaussians = [x[1] for x in renders_init]

            cfg = SfMConfig(
                image_size=(160, 120),
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh,
                n_neighbors=n_neighbors,
                enabled_features=['position', 'color'],
                feature_weights={'position': position_weight,
                                 'color': 1.0 - position_weight}
            )

            sfm = GaussianIncrementalSfM(cfg)

            results = [
                sfm.match_gaussians(gaussians[i], gaussians[i+1])
                for i in range(3)
            ]

            tracks = sfm.build_tracks(gaussians, results)

            cam_poses, points_3d, track_to_point = sfm.incremental_sfm(
                gaussians, results, tracks
            )

            K = results[0]['K']

            point_for_observation = {}
            for track_id, point_id in enumerate(track_to_point):
                if point_id != -1:
                    track = tracks[track_id]
                    for view, idx in track:
                        point_for_observation[(view, idx)] = point_id

            dense_gaussians_3d = lift_gaussians_to_3d(
                gaussians,
                cam_poses,
                K,
                points_3d,
                point_for_observation,
                k_neighbors=k_neighbors,
                opacity_threshold_factor=opacity_threshold_factor,
                eps=eps
            )

            Rasterizer_cfg = GaussianRGBRendererConfig(
                fx=int(K[0, 0]),
                fy=int(K[1, 1]),
                cx=int(K[0, 2]),
                cy=int(K[1, 2])
            )

            renderer = GaussianRGBRenderer(Rasterizer_cfg)

            renders = []
            for i in range(4):
                R, t = cam_poses[i]
                transformed = []
                for g in dense_gaussians_3d:
                    mean_cam = R @ g.mean + t.reshape(3,)
                    cov_cam = R @ g.cov @ R.T
                    transformed.append(
                        type(g)(mean=mean_cam,
                                cov=cov_cam,
                                color=g.color,
                                opacity=g.opacity)
                    )
                renders.append(renderer.render(transformed))

            evaluators = [GaussianSplattingEvaluator(gt) for gt in GT_IMAGES]
            results_eval = [
                evaluators[i].add_algorithm_result(str(uuid.uuid4())[:8], renders[i])
                for i in range(4)
            ]

            avg_mse = float(np.mean([r['MSE'] for r in results_eval]))

            # Save renders if requested
            saved_images = []
            if args.save_images:
                for vi, img in enumerate(renders):
                    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    fname = TRIAL_IMG_DIR / f"trial{trial_id}_view{vi}.png"
                    Image.fromarray(arr).save(fname)
                    saved_images.append(str(fname))

            # Log trial JSON
            trial_log = {
                "trial_number": trial_id,
                "timestamp": ts,
                "hyperparams": trial.params,
                "avg_mse": avg_mse,
                "results_per_view": results_eval,
                "saved_images": saved_images
            }

            json_path = TRIAL_JSON_DIR / f"trial_{trial_id}.json"
            with open(json_path, "w") as f:
                json.dump(make_json_serializable(trial_log), f, indent=2)

            append_summary_row(make_json_serializable({
                **trial.params,
                "trial_number": trial_id,
                "avg_mse": avg_mse
            }))

            return avg_mse

        except Exception as e:
            print("Trial failed:", e)
            return float("inf")

    # ---------------------------------------------------------
    # STUDY SETUP
    # ---------------------------------------------------------

    if args.resume:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize",
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize"
        )

    study.optimize(objective,
                   n_trials=args.n_trials,
                   n_jobs=args.n_jobs)

    print("\nOptimization complete.")
    print("Best MSE:", study.best_value)
    print("Best Params:", study.best_params)


if __name__ == "__main__":
    main()

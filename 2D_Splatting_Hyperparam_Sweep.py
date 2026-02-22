# optuna_ugens_sweep.py
import os
import json
import uuid
import argparse
import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
import optuna
import pandas as pd

# Import your UGen modules (assumes these are on PYTHONPATH / installed)
from UGen.encoder import *
from UGen.encoder.algorithms import *
from UGen.sfm import *
from UGen.visualizer import *
from UGen.BA import *
from UGen.utils.unproject_gaussians import *
from UGen.utils.point_cloud import *
from UGen.decoder import *

# ---------- Configurable ----------
IMAGES_LIST_TXT = "/content/UGen-GS/BatchedImages/images.txt"
IMAGES_FOLDER = "/content/UGen-GS/BatchedImages"  # images are assumed relative to this
IMAGE_SIZE = (160, 120)  # (width, height)
SQLITE_DB = "sqlite:///optuna_ugens_study.db"
STUDY_NAME = "ugens_gauss_splatting"
# ----------------------------------

def load_image(path: Path, size=(160, 120)):
    im = Image.open(str(path)).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def save_image_array(arr: np.ndarray, path: Path):
    # arr expected floating [0,1] or [0,255], shape HxWx3
    arr = np.clip(arr, 0.0, 1.0)
    img = (arr * 255.0).astype(np.uint8)
    Image.fromarray(img).save(str(path))

def read_image_list(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    lines = [l for l in lines if l]
    return lines

def make_out_dirs(base_out: Path):
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "trials").mkdir(exist_ok=True)
    (base_out / "images").mkdir(exist_ok=True)

def objective_factory(image_paths: List[Path], out_base: Path):
    """
    Return objective function closing over image_paths and out_base.
    """

    def objective(trial: optuna.trial.Trial):
        # ----------------------
        # Sample hyperparameters
        # ----------------------
        # The second arg in your config (5000) looks like a choice; use categorical
        target_points = trial.suggest_categorical("target_points", [1000, 2500, 5000])

        min_bbox_size = trial.suggest_int("min_bbox_size", 1, 8)                # int
        # var_threshold in [0, 1.0] but can't sample exactly 0; sample in small positive range
        var_threshold = trial.suggest_float("var_threshold", 1e-8, 1.0, log=True)
        max_points_per_leaf = trial.suggest_int("max_points_per_leaf", 5, 10)   # int
        cov_split_threshold = trial.suggest_float("cov_split_threshold", 0.0, 10.0)
        coverage_threshold = trial.suggest_float("coverage_threshold", 0.0, 5.0)
        num_fill_points = trial.suggest_int("num_fill_points", 0, 1000)
        blur_coverage_threshold = trial.suggest_float("blur_coverage_threshold", 0.0, 1.0)
        blur_factor = trial.suggest_int("blur_factor", 0, 10)
        min_contribution_ratio = trial.suggest_float("min_contribution_ratio", 0.0, 1.0)

        # Create a trial-specific folder
        trial_id = f"trial_{trial.number:04d}_{str(uuid.uuid4())[:8]}"
        trial_out = out_base / "trials" / trial_id
        trial_out.mkdir(parents=True, exist_ok=True)

        # Save sampled hyperparameters
        params = {
            "target_points": int(target_points),
            "min_bbox_size": int(min_bbox_size),
            "var_threshold": float(var_threshold),
            "max_points_per_leaf": int(max_points_per_leaf),
            "cov_split_threshold": float(cov_split_threshold),
            "coverage_threshold": float(coverage_threshold),
            "num_fill_points": int(num_fill_points),
            "blur_coverage_threshold": float(blur_coverage_threshold),
            "blur_factor": int(blur_factor),
            "min_contribution_ratio": float(min_contribution_ratio),
        }
        with open(trial_out / "params.json", "w") as f:
            json.dump(params, f, indent=2)

        per_image_results = []
        mse_list = []
        # iterate over images
        for img_path in image_paths:
            img_name = img_path.name
            try:
                # load image & evaluator (one eval per image)
                img_arr = load_image(img_path, size=IMAGE_SIZE)
                gs_eval = GaussianSplattingEvaluator(img_arr)

                # Build config and encoder
                cfg = RecursiveDensifyConfig(
                    IMAGE_SIZE,
                    params["target_points"],
                    min_bbox_size=params["min_bbox_size"],
                    var_threshold=params["var_threshold"],
                    max_points_per_leaf=params["max_points_per_leaf"],
                    cov_split_threshold=params["cov_split_threshold"],
                    coverage_threshold=params["coverage_threshold"],
                    num_fill_points=params["num_fill_points"],
                    blur_coverage_threshold=params["blur_coverage_threshold"],
                    blur_factor=params["blur_factor"],
                    min_contribution_ratio=params["min_contribution_ratio"],
                )

                # instantiate encoder (path arg preserved as original usage)
                quadtreeden_enc = RecursiveQuadtreeDensifyEncoder(str(img_path), cfg)

                # render (return_gaussians True not needed here; we only need image)
                rendered, gauss = quadtreeden_enc.render(return_gaussians=True)

                # Evaluate and log metrics
                uid = str(uuid.uuid4())[:8]
                trial_name = f"{trial_id}_{img_name}_{uid}"
                result = gs_eval.add_algorithm_result(trial_name, rendered)

                # Save rendered image
                img_save_path = out_base / "images" / f"{trial_id}__{img_name}"
                save_image_array(rendered, img_save_path)

                per_image_results.append({
                    "image": img_name,
                    "success": True,
                    "result": {k: float(v) for k, v in result.items()}
                })
                mse_list.append(float(result.get("MSE", np.nan)))

            except Exception as e:
                # graceful skip for any image error, but log it
                per_image_results.append({
                    "image": img_name,
                    "success": False,
                    "error": str(e)
                })
                # optional: record a large penalty MSE so trial is disfavored
                mse_list.append(float("nan"))

        # Convert MSEs: use only finite values
        mse_arr = np.array([m for m in mse_list if np.isfinite(m)])
        if mse_arr.size == 0:
            # nothing succeeded: give a large penalty
            aggregated_mse_mean = float("inf")
            aggregated_mse_std = None
        else:
            aggregated_mse_mean = float(mse_arr.mean())
            aggregated_mse_std = float(mse_arr.std())

        trial_result = {
            "trial": trial.number,
            "trial_id": trial_id,
            "params": params,
            "aggregated": {
                "mse_mean": aggregated_mse_mean,
                "mse_std": aggregated_mse_std,
                "n_images": int(len(image_paths)),
                "n_success": int(np.sum([1 for p in per_image_results if p["success"]])),
            },
            "per_image": per_image_results,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

        # Save per-trial JSON
        with open(trial_out / "summary.json", "w") as f:
            json.dump(trial_result, f, indent=2)

        # Attach some attributes to Optuna trial (makes dashboard easier)
        trial.set_user_attr("trial_id", trial_id)
        trial.set_user_attr("mse_mean", aggregated_mse_mean)
        trial.set_user_attr("mse_std", aggregated_mse_std)
        for k, v in params.items():
            trial.set_user_attr(k, v)

        # Return the objective to minimize
        # (use aggregated_mse_mean; if inf return large number)
        if aggregated_mse_mean == float("inf"):
            return 1e6
        return aggregated_mse_mean

    return objective

def run_study(image_txt: str,
              images_folder: str,
              out_dir: str,
              n_trials: int = 50,
              n_jobs: int = 1,
              sampler=None,
              pruner=None,
              resume: bool = True):
    image_list = read_image_list(image_txt)
    image_paths = [Path(images_folder) / p for p in image_list]

    out_base = Path(out_dir)
    make_out_dirs(out_base)

    # Create / resume Optuna study backed by SQLite file
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=SQLITE_DB,
        direction="minimize",
        load_if_exists=resume,
        sampler=sampler,
        pruner=pruner
    )

    objective = objective_factory(image_paths, out_base)

    # Optimize
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Export results to CSV summary
    trials = []
    for t in study.trials:
        entry = {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "trial_id": t.user_attrs.get("trial_id", ""),
            "mse_mean": t.user_attrs.get("mse_mean", None),
            "mse_std": t.user_attrs.get("mse_std", None),
        }
        # include params (flatten)
        for k, v in t.user_attrs.items():
            if k not in entry:
                entry[k] = v
        trials.append(entry)

    df = pd.DataFrame(trials)
    df.to_csv(out_base / "study_summary.csv", index=False)

    # Save best trial info
    best = study.best_trial
    with open(out_base / "best_trial.json", "w") as f:
        json.dump({
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs
        }, f, indent=2)

    print(f"Study finished. Best trial #{best.number} value={best.value:.6g}")
    print(f"Results saved in {out_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_txt", default=IMAGES_LIST_TXT)
    parser.add_argument("--images_folder", default=IMAGES_FOLDER)
    parser.add_argument("--out", default="./optuna_ugens_out")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)

    # 👇 THIS FIXES COLAB
    args, unknown = parser.parse_known_args()

    run_study(
        image_txt=args.images_txt,
        images_folder=args.images_folder,
        out_dir=args.out,
        n_trials=args.trials,
        n_jobs=args.jobs,
    )

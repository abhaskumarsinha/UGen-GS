from UGen.encoder import *
from UGen.encoder.algorithms import *
from UGen.encoder.algorithms.quadtree_visualizer import *
from UGen.visualizer import *
from UGen.utils.point_cloud import *
from UGen.decoder import *

from UGen.utils.unproj_gaussians import *
from UGen.utils.colmap_cameras import *

from UGen.renderer import *

import uuid
import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ======== ARGPARSE ========
parser = argparse.ArgumentParser(description="Generate quadtree illustrations for a single image.")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
parser.add_argument("--output", type=str, required=True, help="Output directory to save illustrations")

args = parser.parse_args()

input_image_path = args.input
output_folder = args.output
# ==========================


y0 = Image.open(input_image_path).convert("RGB")

y0 = y0.resize((160, 120), Image.LANCZOS)  # (width, height)

y0 = np.asarray(y0).astype(np.float32) / 255.0


quadtreeden_cfg = RecursiveDensifyConfig((160, 120),
                                         10000,
                                         min_bbox_size=2,
                                         var_threshold=0.0021,
                                         max_points_per_leaf = 7,
                                         cov_split_threshold = 4.05,
                                         coverage_threshold = 0.412,
                                         num_fill_points = 1209,
                                         blur_coverage_threshold = 0.892,
                                         blur_factor = 7,
                                         min_contribution_ratio = 0.268)


y0_viz = RecursiveQuadtreeVisualizer(input_image_path, quadtreeden_cfg)


# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)


# Save all illustrations
y0_viz.save_all(base_name="my_image",
                output_dir=output_folder,
                width_pixels=2070,
                dpi=600,
                sobel_point_color='red',
                sobel_point_size=1,
                cell_color='blue',
                cell_linewidth=1,
                gaussians_fill=False,
                gaussians_edgecolor='white',
                gaussians_alpha=1.0)

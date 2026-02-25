import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os

# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------
@dataclass
class Gaussian2D:
    mean: np.ndarray      # (2,)
    cov: np.ndarray       # (2,2)
    color: np.ndarray     # (3,)
    opacity: float = 1.0

@dataclass
class Gaussian3D:
    mean: np.ndarray      # (3,)
    cov: np.ndarray       # (3,3)
    color: np.ndarray     # (3,)
    opacity: float

# ------------------------------------------------------------
# COLMAP text readers (pinhole-only extraction)
# ------------------------------------------------------------
def read_cameras_text_pinhole(path: str) -> Dict[int, Tuple[float, float, float, float, int, int]]:
    """
    Extract pinhole parameters from any COLMAP camera model.
    Returns: dict camera_id -> (fx, fy, cx, cy, width, height)
    """
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]

            # Extract pinhole intrinsics (ignore distortion)
            if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                # params: f, cx, cy, ...
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
                # params: fx, fy, cx, cy, ...
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                # Fallback: assume first 4 are fx, fy, cx, cy
                if len(params) >= 4:
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                elif len(params) == 3:
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                else:
                    raise ValueError(f"Cannot extract pinhole from {model}: {params}")
            cameras[camera_id] = (fx, fy, cx, cy, width, height)
    return cameras

def read_images_text(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Read images.txt. Returns dict: image_name -> (quat, trans, camera_id)"""
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        quat = np.array([qw, qx, qy, qz])
        trans = np.array([tx, ty, tz])
        images[name] = (quat, trans, camera_id)
        i += 1  # skip the second line (2D points)
    return images

def read_points3d_text(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Read points3D.txt. Returns dict: point_id -> (XYZ, RGB) with RGB in [0,1]."""
    points = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points[point_id] = (np.array([x, y, z]), np.array([r, g, b]) / 255.0)
    return points

# ------------------------------------------------------------
# Corrected quaternion → rotation matrix
# ------------------------------------------------------------
def quaternion_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w    ],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w    ],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])

# ------------------------------------------------------------
# Main conversion function
# ------------------------------------------------------------
def colmap_2d_to_3d_gaussians(
    colmap_path: str,
    view_gaussians: List[List[Gaussian2D]],
    image_names: List[str],
    distance_threshold: float = 3.0,
    target_resolution: Tuple[int, int] = None,
    k: int = 3,                       # number of neighbors to consider
    verbose: bool = False
) -> List[Gaussian3D]:
    """
    Convert per-view 2D Gaussians to 3D Gaussians using COLMAP sparse points.
    All camera models are treated as pinhole (distortion ignored).

    Args:
        colmap_path: Folder containing cameras.txt, images.txt, points3D.txt.
        view_gaussians: List of lists; view_gaussians[i] = 2D Gaussians for view i.
        image_names: List of image filenames corresponding to views.
        distance_threshold: Max pixel distance (at target resolution) to keep a Gaussian.
        target_resolution: If provided, (new_width, new_height) for rendering.
                           The intrinsics from COLMAP are scaled accordingly.
                           The 2D Gaussians must be defined on this resolution.
        k: Number of nearest neighbors to query for depth averaging.
        verbose: If True, print statistics per view.

    Returns:
        List of Gaussian3D objects.
    """
    # Load COLMAP data
    cameras = read_cameras_text_pinhole(os.path.join(colmap_path, "cameras.txt"))
    images = read_images_text(os.path.join(colmap_path, "images.txt"))
    points3D = read_points3d_text(os.path.join(colmap_path, "points3D.txt"))

    all_3d_gaussians = []

    for view_idx, gaussians in enumerate(view_gaussians):
        img_name = image_names[view_idx]
        if img_name not in images:
            raise ValueError(f"Image {img_name} not found in COLMAP model.")

        quat, trans, cam_id = images[img_name]
        fx, fy, cx, cy, orig_width, orig_height = cameras[cam_id]

        # Scale intrinsics if requested
        if target_resolution is not None:
            target_w, target_h = target_resolution
            sx = target_w / orig_width
            sy = target_h / orig_height
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy
            width, height = target_w, target_h
        else:
            width, height = orig_width, orig_height

        R_w2c = quaternion_to_rotation_matrix(quat)
        t_w2c = trans

        # Project all 3D points into this image
        proj_points = []  # (u, v, Z_cam, point_id)
        for pt_id, (xyz, _) in points3D.items():
            P_cam = R_w2c @ xyz + t_w2c
            if P_cam[2] <= 0:
                continue
            u = fx * P_cam[0] / P_cam[2] + cx
            v = fy * P_cam[1] / P_cam[2] + cy
            if 0 <= u < width and 0 <= v < height:
                proj_points.append((u, v, P_cam[2], pt_id))

        if not proj_points:
            if verbose:
                print(f"View {img_name}: no projected 3D points, skipping.")
            continue

        proj_array = np.array([[p[0], p[1]] for p in proj_points])
        z_vals = np.array([p[2] for p in proj_points])
        pt_ids = [p[3] for p in proj_points]
        tree = cKDTree(proj_array)

        kept = 0
        for g2d in gaussians:
            # Find k nearest projected points
            distances, indices = tree.query(g2d.mean.reshape(1, -1), k=k)

            # Flatten in case k=1 returns 2D arrays
            distances = distances.flatten()
            indices = indices.flatten()

            # Filter neighbors within threshold
            valid_mask = distances <= distance_threshold
            if not np.any(valid_mask):
                continue

            # Use the minimum distance for opacity fade (closest point)
            min_dist = np.min(distances[valid_mask])

            # Average depth from all valid neighbors
            avg_Z_cam = np.mean(z_vals[indices[valid_mask]])

            # Unproject Gaussian mean using average depth
            u, v = g2d.mean
            X_cam = (u - cx) * avg_Z_cam / fx
            Y_cam = (v - cy) * avg_Z_cam / fy
            P_cam_mean = np.array([X_cam, Y_cam, avg_Z_cam])

            # Transform to world coordinates
            mean_world = R_w2c.T @ (P_cam_mean - t_w2c)

            # Build 3D covariance that projects to g2d.cov
            J_inv = np.diag([avg_Z_cam / fx, avg_Z_cam / fy])
            cov_xy = J_inv @ g2d.cov @ J_inv.T

            cov_cam = np.zeros((3, 3))
            cov_cam[:2, :2] = cov_xy
            cov_world = R_w2c.T @ cov_cam @ R_w2c

            # Opacity: linear fade with the minimum distance
            opacity = g2d.opacity * max(0.0, 1.0 - min_dist / distance_threshold)

            all_3d_gaussians.append(Gaussian3D(
                mean=mean_world,
                cov=cov_world,
                color=g2d.color,
                opacity=opacity
            ))
            kept += 1

        if verbose:
            print(f"View {img_name}: {len(gaussians)} Gaussians, {kept} kept (threshold={distance_threshold}px, k={k})")

    return all_3d_gaussians

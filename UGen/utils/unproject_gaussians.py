import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Dict
from dataclasses import dataclass

from UGen.sfm import *

@dataclass
class Gaussian3D:
    mean: np.ndarray       # (3,)
    cov: np.ndarray        # (3,3)
    color: np.ndarray      # (3,) RGB
    opacity: float

def lift_gaussians_to_3d(
    gaussians_per_view: List[List[Gaussian2D]],
    camera_poses: List[Tuple[np.ndarray, np.ndarray]],  # (R, t) each, world->cam, t is (3,1) column
    K: np.ndarray,
    point_cloud: List[np.ndarray],                      # 3D points from SfM (each as (3,) or (3,1))
    point_for_observation: Dict[Tuple[int, int], int],  # (view, idx) -> point_id
    k_neighbors: int = 5,
    opacity_threshold_factor: float = 2.0,
    eps: float = 1e-6
) -> List[Gaussian3D]:
    """
    For every 2D Gaussian in every view, create a 3D Gaussian that exactly
    reproduces that 2D Gaussian when projected into its view.

    Parameters
    ----------
    gaussians_per_view : list of lists of Gaussian2D
        All 2D Gaussians.
    camera_poses : list of (R, t)
        Camera poses: world → camera: X_cam = R @ X_world + t.
        R is (3,3), t is (3,1) column.
    K : 3x3 intrinsic matrix.
    point_cloud : list of (3,) or (3,1) arrays
        Triangulated 3D points from SfM.
    point_for_observation : dict
        Maps (view, idx) to point_id (index in point_cloud).
    k_neighbors : int
        Number of neighbours to use for depth interpolation.
    opacity_threshold_factor : float
        Multiplier for the median neighbour distance to set opacity threshold.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    list of Gaussian3D
        One 3D Gaussian per 2D observation.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    all_3d_gaussians = []

    # For each view, precompute known points (those with a point_id)
    for view_idx, (R, t) in enumerate(camera_poses):
        # Ensure t is column vector (3,1)
        if t.shape == (3,):
            t = t.reshape(3, 1)

        gaussians = gaussians_per_view[view_idx]
        n = len(gaussians)

        # Separate known and unknown indices
        known_mask = np.zeros(n, dtype=bool)
        known_depths = np.full(n, np.nan)          # depth (Z in camera frame) for known points
        known_world_pts = [None] * n                # corresponding world point (original SfM point)
        known_ids = []                              # list of indices that are known

        for i, g in enumerate(gaussians):
            key = (view_idx, i)
            if key in point_for_observation:
                point_id = point_for_observation[key]
                # Safety check for out-of-range point_id
                if point_id >= len(point_cloud):
                    print(f"Warning: point_id {point_id} out of range (point_cloud size {len(point_cloud)}). Skipping observation in view {view_idx}, idx {i}.")
                    continue
                pt_world_sfm = point_cloud[point_id]
                # Ensure column vector
                if pt_world_sfm.shape == (3,):
                    pt_world_sfm = pt_world_sfm.reshape(3, 1)
                known_mask[i] = True
                known_ids.append(i)
                known_world_pts[i] = pt_world_sfm
                # Compute camera coordinates
                pt_cam = R @ pt_world_sfm + t
                # Extract scalar depth (Z coordinate)
                known_depths[i] = pt_cam[2, 0]   # Z coordinate as scalar

        if len(known_ids) == 0:
            # No known points in this view – skip this view
            print(f"Warning: View {view_idx} has no known 3D points. Skipping view.")
            continue

        # Build KDTree for known 2D positions (for depth interpolation)
        known_pts_2d = np.array([gaussians[i].mean for i in known_ids])
        tree = NearestNeighbors(n_neighbors=min(k_neighbors, len(known_ids)), algorithm='auto').fit(known_pts_2d)

        # Precompute ray directions for all Gaussians in this view (camera frame, normalized)
        ray_dirs = []   # list of unit direction vectors in camera frame (3,)
        for g in gaussians:
            u, v = g.mean
            # direction in camera coordinates (not normalized)
            dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            # normalize
            norm = np.linalg.norm(dir_cam)
            if norm < eps:
                dir_cam = np.array([0, 0, 1])  # fallback
            else:
                dir_cam = dir_cam / norm
            ray_dirs.append(dir_cam)

        # Process each Gaussian in this view
        for i, g in enumerate(gaussians):
            if known_mask[i]:
                # ------------------------------------------------------------
                # Case 1: Known observation (has a corresponding 3D point)
                # ------------------------------------------------------------
                pt_world_sfm = known_world_pts[i]   # (3,1)
                # Compute camera coordinates of the SfM point
                pt_cam_sfm = R @ pt_world_sfm + t   # (3,1)
                # Get the ray direction for this 2D point
                d = ray_dirs[i]                      # (3,)
                # Project the SfM point onto the ray to get depth along ray
                # (λ = dot(pt_cam_sfm, d) because d is unit)
                lam = np.dot(pt_cam_sfm.flatten(), d)   # scalar
                # New camera point on the ray
                new_pt_cam = lam * d                  # (3,)
                # Convert back to world
                new_pt_world = (R.T @ (new_pt_cam.reshape(3,1) - t)).flatten()  # (3,)
                mean_world = new_pt_world
                # Opacity: 1 for known points
                opacity = 1.0
            else:
                # ------------------------------------------------------------
                # Case 2: Unknown observation – estimate depth from neighbours
                # ------------------------------------------------------------
                # Find nearest known 2D points
                distances, indices = tree.kneighbors([g.mean], n_neighbors=k_neighbors)
                indices = indices[0]
                dists = distances[0]
                # If the closest neighbour is too far, we may want to skip this point
                # We'll still estimate but set opacity low.
                # Get depths of neighbours (Z in camera frame)
                neighbour_ids = [known_ids[idx] for idx in indices]
                neighbour_depths = [known_depths[nid] for nid in neighbour_ids]
                # Inverse distance weighting
                weights = 1.0 / (dists + eps)
                weights /= weights.sum()
                depth_est = np.sum(weights * neighbour_depths)   # scalar

                # Compute 3D point in camera frame using the estimated depth
                d = ray_dirs[i]
                new_pt_cam = depth_est * d            # (3,)
                # World point
                new_pt_world = (R.T @ (new_pt_cam.reshape(3,1) - t)).flatten()
                mean_world = new_pt_world

                # Opacity: based on distance to nearest known 3D point (in world)
                # Compute world coordinates of the nearest known point
                nearest_known_idx = known_ids[indices[0]]
                nearest_world = known_world_pts[nearest_known_idx].flatten()   # (3,)
                dist_to_nearest = np.linalg.norm(mean_world - nearest_world)
                # Compute threshold as factor * median of distances between known points in this view
                if len(known_ids) > 1:
                    # Compute pairwise distances among known points in world
                    known_worlds = np.array([known_world_pts[idx].flatten() for idx in known_ids])
                    diffs = known_worlds[:, None, :] - known_worlds[None, :, :]
                    pair_dists = np.linalg.norm(diffs, axis=-1)
                    # Use upper triangle to avoid zeros
                    median_dist = np.median(pair_dists[pair_dists > 0])
                else:
                    median_dist = 1.0   # fallback
                threshold = opacity_threshold_factor * median_dist
                if dist_to_nearest > threshold:
                    opacity = 0.0
                else:
                    # Linear falloff
                    opacity = 1.0 - (dist_to_nearest / threshold)
                opacity = np.clip(opacity, 0.0, 1.0)

            # ------------------------------------------------------------
            # Compute 3D covariance such that projection matches observed 2D covariance
            # ------------------------------------------------------------
            # Observed 2D covariance
            Sigma_img = g.cov  # 2x2

            # Camera coordinates of the chosen mean
            pt_cam = R @ mean_world.reshape(3,1) + t
            # Extract scalars correctly
            X = pt_cam[0, 0]
            Y = pt_cam[1, 0]
            Z = pt_cam[2, 0]
            Z = max(Z, eps)   # avoid division by zero

            # Jacobian of projection (pixel coordinates) w.r.t. camera point
            J = np.array([
                [fx / Z, 0,      -fx * X / (Z*Z)],
                [0,      fy / Z, -fy * Y / (Z*Z)]
            ])   # shape (2,3)

            # Ray direction (unit) in camera frame
            d = ray_dirs[i]   # (3,)

            # Find two orthonormal vectors orthogonal to d
            if abs(d[0]) < 0.9:
                a = np.array([1, 0, 0])
            else:
                a = np.array([0, 1, 0])
            u = np.cross(d, a)
            u = u / (np.linalg.norm(u) + eps)
            v = np.cross(d, u)   # already orthogonal and unit if d and u are orthonormal
            v = v / (np.linalg.norm(v) + eps)

            # Matrix of tangent basis (3x2): [u, v]
            B = np.column_stack((u, v))

            # Jacobian restricted to tangent plane: J_plane = J @ B  (2x2)
            J_plane = J @ B

            # We need Σ_plane such that J_plane @ Σ_plane @ J_plane.T = Sigma_img
            try:
                Jinv = np.linalg.inv(J_plane)
                Sigma_plane = Jinv @ Sigma_img @ Jinv.T
                Sigma_plane = (Sigma_plane + Sigma_plane.T) / 2.0
            except np.linalg.LinAlgError:
                # If singular, fallback to diagonal
                Sigma_plane = np.eye(2) * 1e-6

            # Construct 3D covariance in camera frame: Σ_cam = B @ Σ_plane @ B.T
            Sigma_cam = B @ Sigma_plane @ B.T
            Sigma_cam = (Sigma_cam + Sigma_cam.T) / 2.0

            # Transform to world frame: Σ_world = R.T @ Σ_cam @ R
            Sigma_world = R.T @ Sigma_cam @ R

            # Create 3D Gaussian
            g3d = Gaussian3D(
                mean=mean_world,
                cov=Sigma_world,
                color=g.color,
                opacity=opacity
            )
            all_3d_gaussians.append(g3d)

    return all_3d_gaussians

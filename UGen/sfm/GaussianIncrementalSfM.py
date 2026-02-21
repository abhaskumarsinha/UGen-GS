import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Gaussian2D:
    mean: np.ndarray
    cov: np.ndarray
    color: np.ndarray
    opacity: float = 1.0


from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, List
import numpy as np


@dataclass
class SfMConfig:
    # ---------------------------
    # Matching parameters
    # ---------------------------
    K: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    enabled_features: Optional[Set[str]] = None
    feature_weights: Optional[Dict[str, float]] = None
    ratio_thresh: float = 0.75
    ransac_thresh: float = 1.0
    visualize: bool = True
    filter_invalid: bool = True
    n_neighbors: int = 50

    # ---------------------------
    # PnP parameters
    # ---------------------------
    pnp_iterations: int = 100
    pnp_reprojection_error: float = 8.0
    pnp_confidence: float = 0.99
    pnp_flags: int = 0  # will be set to cv2.SOLVEPNP_ITERATIVE after cv2 import



import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class GaussianIncrementalSfM:

    def __init__(self, config: SfMConfig):
        self.cfg = config

        if self.cfg.enabled_features is None:
            self.cfg.enabled_features = {'color', 'eigenvalues', 'angle'}

        if self.cfg.feature_weights is None and self.cfg.enabled_features is not None:
            self.cfg.feature_weights = {
                feat: 1.0 for feat in self.cfg.enabled_features
            }

        if self.cfg.pnp_flags == 0:
            self.cfg.pnp_flags = cv2.SOLVEPNP_ITERATIVE


    # ============================================================
    # 1. Gaussian Matching (UNMODIFIED LOGIC)
    # ============================================================
    def match_gaussians(self, gauss1, gauss2):

        enabled_features = self.cfg.enabled_features
        feature_weights = self.cfg.feature_weights
        image_size = self.cfg.image_size
        K = self.cfg.K
        ratio_thresh = self.cfg.ratio_thresh
        ransac_thresh = self.cfg.ransac_thresh
        visualize = self.cfg.visualize
        filter_invalid = self.cfg.filter_invalid
        n_neighbors = self.cfg.n_neighbors

        if 'position' in enabled_features and image_size is None:
            raise ValueError("image_size must be provided when 'position' is enabled.")

        if K is None:
            if image_size is None:
                raise ValueError("Either K or image_size must be provided.")
            w, h = image_size
            f = 1.2 * max(w, h)
            K = np.array([[f, 0, w/2],
                          [0, f, h/2],
                          [0, 0, 1]], dtype=np.float32)

        def compute_feature(g):
            components = []

            if 'position' in enabled_features:
                x, y = g.mean
                w, h = image_size
                norm_pos = np.array([x / w, y / h], dtype=np.float32)
                components.append(feature_weights['position'] * norm_pos)

            if 'color' in enabled_features:
                components.append(feature_weights['color'] * g.color)

            if 'eigenvalues' in enabled_features or 'angle' in enabled_features:
                eigvals, eigvecs = np.linalg.eigh(g.cov)
                eigvals = np.sort(eigvals)[::-1]
                eigvals = np.maximum(eigvals, 1e-6)

                if 'eigenvalues' in enabled_features:
                    log_eigvals = np.log(eigvals)
                    components.append(feature_weights['eigenvalues'] * log_eigvals)

                if 'angle' in enabled_features:
                    major_vec = eigvecs[:, 0] if eigvals[0] >= eigvals[1] else eigvecs[:, 1]
                    angle = np.arctan2(major_vec[1], major_vec[0]) * 180 / np.pi
                    angle_norm = (angle % 360) / 360.0
                    components.append(np.array([feature_weights['angle'] * angle_norm]))

            return np.concatenate(components)

        features1_raw = np.array([compute_feature(g) for g in gauss1])
        features2_raw = np.array([compute_feature(g) for g in gauss2])

        valid1 = np.all(np.isfinite(features1_raw), axis=1)
        valid2 = np.all(np.isfinite(features2_raw), axis=1)

        if filter_invalid:
            gauss1_filt = [g for i, g in enumerate(gauss1) if valid1[i]]
            gauss2_filt = [g for i, g in enumerate(gauss2) if valid2[i]]
            features1 = features1_raw[valid1]
            features2 = features2_raw[valid2]
            orig_idx1 = np.where(valid1)[0]
            orig_idx2 = np.where(valid2)[0]
        else:
            gauss1_filt = gauss1
            gauss2_filt = gauss2
            features1 = features1_raw
            features2 = features2_raw
            orig_idx1 = np.arange(len(gauss1))
            orig_idx2 = np.arange(len(gauss2))

        nbrs1 = NearestNeighbors(n_neighbors=n_neighbors).fit(features1)
        nbrs2 = NearestNeighbors(n_neighbors=n_neighbors).fit(features2)

        distances2, indices2 = nbrs2.kneighbors(features1, n_neighbors=n_neighbors)
        distances1, indices1 = nbrs1.kneighbors(features2, n_neighbors=n_neighbors)

        matches_filt = []
        for i1_filt in range(len(gauss1_filt)):
            if distances2[i1_filt, 0] < ratio_thresh * distances2[i1_filt, 1]:
                i2_filt = indices2[i1_filt, 0]
                if indices1[i2_filt, 0] == i1_filt and distances1[i2_filt, 0] < ratio_thresh * distances1[i2_filt, 1]:
                    matches_filt.append((i1_filt, i2_filt))

        matches = [(orig_idx1[i1], orig_idx2[i2]) for i1, i2 in matches_filt]

        pts1 = np.array([gauss1[i1].mean for i1, _ in matches], dtype=np.float32)
        pts2 = np.array([gauss2[i2].mean for _, i2 in matches], dtype=np.float32)

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_thresh
        )

        mask = mask.ravel().astype(bool)
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

        _, R, t, _ = cv2.recoverPose(E, pts1[mask], pts2[mask], K)

        pts1_norm = cv2.undistortPoints(np.expand_dims(pts1[mask], axis=1), K, None)
        pts2_norm = cv2.undistortPoints(np.expand_dims(pts2[mask], axis=1), K, None)

        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))

        points_4d = cv2.triangulatePoints(
            P1, P2,
            pts1_norm.squeeze(1).T,
            pts2_norm.squeeze(1).T
        )

        points_3d = (points_4d[:3] / points_4d[3]).T

        return {
            'matches': matches,
            'inlier_matches': inlier_matches,
            'R': R,
            't': t,
            'K': K,
            'points_3d': points_3d,
            'mask': mask
        }


    # ============================================================
    # 2. Track Builder (UNMODIFIED)
    # ============================================================
    def build_tracks(self, gaussians_per_view, results):

        n_views = len(gaussians_per_view)

        forward_maps = []
        for res in results:
            fwd = {}
            for i1, i2 in res['inlier_matches']:
                fwd[i1] = i2
            forward_maps.append(fwd)

        backward_maps = []
        for res in results:
            bwd = {}
            for i1, i2 in res['inlier_matches']:
                bwd[i2] = i1
            backward_maps.append(bwd)

        tracks = []
        used_in_later = set()

        for idx1 in range(len(gaussians_per_view[0])):
            if any(idx1 in fwd for fwd in forward_maps):
                track = [(0, idx1)]
                cur_view = 0
                cur_idx = idx1
                while cur_view < n_views - 1 and cur_idx in forward_maps[cur_view]:
                    next_idx = forward_maps[cur_view][cur_idx]
                    track.append((cur_view+1, next_idx))
                    used_in_later.add((cur_view+1, next_idx))
                    cur_view += 1
                    cur_idx = next_idx
                tracks.append(track)
            else:
                tracks.append([(0, idx1)])

        for view in range(1, n_views):
            for idx in range(len(gaussians_per_view[view])):
                if (view, idx) in used_in_later:
                    continue
                if view > 0 and idx in backward_maps[view-1]:
                    continue
                track = [(view, idx)]
                cur_view = view
                cur_idx = idx
                while cur_view < n_views - 1 and cur_idx in forward_maps[cur_view]:
                    next_idx = forward_maps[cur_view][cur_idx]
                    track.append((cur_view+1, next_idx))
                    used_in_later.add((cur_view+1, next_idx))
                    cur_view += 1
                    cur_idx = next_idx
                tracks.append(track)

        return tracks


    # ============================================================
    # 3. Incremental SfM (UNMODIFIED)
    # ============================================================
    def incremental_sfm(self, gaussians_per_view, results, tracks):

        K = results[0]['K']
        n_views = len(gaussians_per_view)

        cam_poses = []
        cam_poses.append((np.eye(3), np.zeros((3,1))))

        R1 = results[0]['R']
        t1 = results[0]['t']
        cam_poses.append((R1, t1))

        points_3d = []
        track_to_point = [-1] * len(tracks)

        # === Initial triangulation ===
        for tid, track in enumerate(tracks):
            obs0 = next(((v,idx) for v,idx in track if v==0), None)
            obs1 = next(((v,idx) for v,idx in track if v==1), None)
            if obs0 and obs1:
                pt0 = gaussians_per_view[0][obs0[1]].mean
                pt1 = gaussians_per_view[1][obs1[1]].mean

                pt0_n = cv2.undistortPoints(np.array([[pt0]], dtype=np.float32), K, None)
                pt1_n = cv2.undistortPoints(np.array([[pt1]], dtype=np.float32), K, None)

                P0 = np.hstack((np.eye(3), np.zeros((3,1))))
                P1 = np.hstack((R1, t1))

                pts4d = cv2.triangulatePoints(P0, P1, pt0_n[0].T, pt1_n[0].T)
                pt3d = (pts4d[:3] / pts4d[3]).flatten()

                points_3d.append(pt3d)
                track_to_point[tid] = len(points_3d)-1

        # === Add new views ===
        for i in range(2, n_views):

            obj_pts = []
            img_pts = []
            track_ids = []

            for tid, track in enumerate(tracks):
                if track_to_point[tid] == -1:
                    continue
                obs_i = next(((v,idx) for v,idx in track if v==i), None)
                if obs_i:
                    pt3d = points_3d[track_to_point[tid]]
                    pt2d = gaussians_per_view[i][obs_i[1]].mean
                    obj_pts.append(pt3d)
                    img_pts.append(pt2d)
                    track_ids.append(tid)

            if len(obj_pts) < 5:
                raise RuntimeError(f"Not enough correspondences for view {i}")

            obj_pts = np.array(obj_pts, dtype=np.float32)
            img_pts = np.array(img_pts, dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts,
                img_pts,
                K,
                None,
                iterationsCount=self.cfg.pnp_iterations,
                reprojectionError=self.cfg.pnp_reprojection_error,
                confidence=self.cfg.pnp_confidence,
                flags=self.cfg.pnp_flags
            )

            if not success:
                raise RuntimeError(f"PnP failed for view {i}")

            R_i, _ = cv2.Rodrigues(rvec)
            cam_poses.append((R_i, tvec))

        return cam_poses, points_3d, track_to_point

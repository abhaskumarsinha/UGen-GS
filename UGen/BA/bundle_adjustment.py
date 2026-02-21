import numpy as np
import cv2
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================
# Config Dataclass
# ============================================================

@dataclass
class BundleAdjustmentConfig:
    fix_first_camera: bool = True

    # SciPy least_squares parameters
    method: str = 'lm'
    verbose: int = 2
    x_scale: str = 'jac'
    max_nfev: int = 20
    ftol: float = 1e-4
    gtol: float = 1e-6


# ============================================================
# Bundle Adjustment Class
# ============================================================

class BundleAdjustmentScipy:

    def __init__(self, config: BundleAdjustmentConfig):
        self.cfg = config

    def run(self,
            cam_poses: List[Tuple[np.ndarray, np.ndarray]],
            points_3d: List[np.ndarray],
            tracks,
            track_to_point,
            gaussians_per_view,
            K: np.ndarray):
        """
        Refine cameras and 3D points using SciPy's Levenberg-Marquardt.
        """

        cfg = self.cfg

        n_cameras = len(gaussians_per_view)
        n_points = len(points_3d)

        # --------------------------------------------------
        # Build observation list
        # --------------------------------------------------
        observations = []
        for point_id in range(n_points):
            for track_id, pid in enumerate(track_to_point):
                if pid == point_id:
                    for view, idx in tracks[track_id]:
                        pt2d = gaussians_per_view[view][idx].mean
                        observations.append((view, point_id, pt2d[0], pt2d[1]))

        obs = np.array(observations, dtype=np.float64)
        n_obs = obs.shape[0]

        start_cam = 1 if cfg.fix_first_camera else 0
        n_cam_params = (n_cameras - start_cam) * 6
        n_point_params = n_points * 3
        n_params = n_cam_params + n_point_params

        # --------------------------------------------------
        # Pack initial parameters
        # --------------------------------------------------
        x0 = []

        for i in range(start_cam, n_cameras):
            R, t = cam_poses[i]
            rvec, _ = cv2.Rodrigues(R)
            x0.extend(rvec.flatten())
            x0.extend(t.flatten())

        for p in points_3d:
            x0.extend(p)

        x0 = np.array(x0, dtype=np.float64)

        # --------------------------------------------------
        # Residual function
        # --------------------------------------------------
        def residuals(params):

            point_params = params[n_cam_params:].reshape(-1, 3)

            if cfg.fix_first_camera:
                R_fixed, t_fixed = cam_poses[0]
                rvec_fixed, _ = cv2.Rodrigues(R_fixed)
                t_fixed_flat = t_fixed.flatten()

            errors = []

            for obs_idx in range(n_obs):

                cam_idx = int(obs[obs_idx, 0])
                pt_idx = int(obs[obs_idx, 1])
                x_obs = obs[obs_idx, 2]
                y_obs = obs[obs_idx, 3]

                if cfg.fix_first_camera and cam_idx == 0:
                    rvec = rvec_fixed
                    t = t_fixed_flat
                else:
                    cam_offset = (cam_idx - start_cam) * 6
                    rvec = params[cam_offset:cam_offset+3]
                    t = params[cam_offset+3:cam_offset+6]

                X = point_params[pt_idx]

                pt2d_proj, _ = cv2.projectPoints(
                    X.reshape(1,1,3), rvec, t, K, None
                )

                proj_x, proj_y = pt2d_proj[0][0]

                errors.append(proj_x - x_obs)
                errors.append(proj_y - y_obs)

            return np.array(errors)

        # --------------------------------------------------
        # Optimization
        # --------------------------------------------------
        result = least_squares(
            residuals,
            x0,
            method=cfg.method,
            verbose=cfg.verbose,
            x_scale=cfg.x_scale,
            max_nfev=cfg.max_nfev,
            ftol=cfg.ftol,
            gtol=cfg.gtol
        )

        # --------------------------------------------------
        # Unpack refined parameters
        # --------------------------------------------------
        refined_cam_poses = cam_poses.copy()

        for i in range(start_cam, n_cameras):
            cam_offset = (i - start_cam) * 6
            rvec = result.x[cam_offset:cam_offset+3]
            t = result.x[cam_offset+3:cam_offset+6].reshape(3,1)
            R, _ = cv2.Rodrigues(rvec)
            refined_cam_poses[i] = (R, t)

        refined_points = result.x[n_cam_params:].reshape(-1, 3)

        return refined_cam_poses, refined_points

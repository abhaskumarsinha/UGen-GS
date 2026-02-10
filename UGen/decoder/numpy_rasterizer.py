from dataclasses import dataclass
from UGen.decoder.base import *


@dataclass
class PointRasterizerConfig:
    return_depth: bool = False


import numpy as np
import math


class NumpyPointRasterizer(BaseRasterizer):
    """
    Sparse (mean-only) Gaussian rasterizer using NumPy.
    Non-trainable, visualization / debugging rasterizer.
    """

    def __init__(self, width: int, height: int, config: PointRasterizerConfig):
        super().__init__(width, height)
        self.config = config

    # ---------------------------------------------------------
    # Helpers (unchanged logic)
    # ---------------------------------------------------------

    def quat_to_rot_matrix(self, q):
        qw, qx, qy, qz = q
        n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n

        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float32)

    def world_to_camera(self, points, R, t):
        return (R @ points.T).T + t

    def project(self, points_cam, fx, fy, cx, cy):
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]

        valid = Z > 1e-6

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        return u, v, Z, valid

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------

    def render(self, cam, gaussian):
        """
        cam      : CameraLayer(...).call()
        gaussian : GaussianLayer(...).call()
        """

        # -------- Torch → NumPy (EXPLICIT) --------
        cam_np = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
            for k, v in cam.items()
        }

        gaussian_np = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
            for k, v in gaussian.items()
        }

        # -------- Camera --------
        H, W = cam_np["height"], cam_np["width"]

        R = self.quat_to_rot_matrix(cam_np["rotation_quaternion"])
        t = np.asarray(cam_np["translation"], dtype=np.float32)

        # -------- Points --------
        points_world = gaussian_np["position"].astype(np.float32)
        colors = np.clip(gaussian_np["sh"], 0.0, 1.0).astype(np.float32)

        # -------- Buffers --------
        image = np.zeros((H, W, 3), dtype=np.float32)
        depth = np.full((H, W), np.inf, dtype=np.float32)

        # -------- World → Camera --------
        points_cam = self.world_to_camera(points_world, R, t)

        # -------- Projection --------
        u, v, z, valid = self.project(
            points_cam,
            cam_np["fx"], cam_np["fy"],
            cam_np["cx"], cam_np["cy"]
        )

        # -------- Rasterize --------
        for i in range(points_world.shape[0]):
            if not valid[i]:
                continue

            x = int(round(u[i]))
            y = int(round(v[i]))

            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            if z[i] < depth[y, x]:
                depth[y, x] = z[i]
                image[y, x] = colors[i]

        if self.config.return_depth:
            return image, depth

        return image

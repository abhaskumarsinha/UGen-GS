from dataclasses import dataclass, field
import numpy as np

@dataclass
class GaussianRendererConfig:
    # use default_factory to avoid mutable-default error
    background_color: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    return_uint8: bool = True


import numpy as np
from typing import List, Dict


class GaussianRenderer:

    def __init__(self, config: GaussianRendererConfig):
        self.cfg = config

    def render(
        self,
        gaussians: List,           # each has .mean (3,), .cov (3,3), .color (3,), .opacity (float)
        cameras: Dict[str, dict],  # output of import_colmap_cameras()
    ) -> Dict[str, np.ndarray]:

        background_color = self.cfg.background_color
        return_uint8 = self.cfg.return_uint8

        results = {}

        for img_name, cam in cameras.items():
            width = cam['width']
            height = cam['height']
            fx = cam['fx']
            fy = cam['fy']
            cx = cam['cx']
            cy = cam['cy']
            R = cam['rotation']
            t = cam['translation']

            visible_gaussians = []

            for g in gaussians:
                mean_cam = R @ g.mean + t
                z = mean_cam[2]
                if z <= 0:
                    continue

                x_cam, y_cam = mean_cam[0], mean_cam[1]
                u = fx * x_cam / z + cx
                v = fy * y_cam / z + cy
                if u < 0 or u >= width or v < 0 or v >= height:
                    continue

                cov_cam = R @ g.cov @ R.T

                J = np.array([
                    [fx / z, 0,      -fx * x_cam / (z * z)],
                    [0,      fy / z, -fy * y_cam / (z * z)]
                ])

                cov2d = J @ cov_cam @ J.T
                cov2d[0, 0] += 1e-6
                cov2d[1, 1] += 1e-6

                visible_gaussians.append((z, np.array([u, v]), cov2d, g.color, g.opacity))

            visible_gaussians.sort(key=lambda x: x[0])

            color_buffer = np.full((height, width, 3), background_color, dtype=np.float32)
            alpha_buffer = np.zeros((height, width), dtype=np.float32)

            for z, mean2d, cov2d, col, op in visible_gaussians:
                try:
                    inv_cov2d = np.linalg.inv(cov2d)
                except np.linalg.LinAlgError:
                    continue

                eigvals, eigvecs = np.linalg.eigh(cov2d)
                major_axis_length = 3.0 * np.sqrt(eigvals[1])
                minor_axis_length = 3.0 * np.sqrt(eigvals[0])

                u, v = mean2d
                bbox_u_min = int(np.floor(u - major_axis_length))
                bbox_u_max = int(np.ceil(u + major_axis_length))
                bbox_v_min = int(np.floor(v - major_axis_length))
                bbox_v_max = int(np.ceil(v + major_axis_length))

                bbox_u_min = max(0, bbox_u_min)
                bbox_u_max = min(width, bbox_u_max)
                bbox_v_min = max(0, bbox_v_min)
                bbox_v_max = min(height, bbox_v_max)

                if bbox_u_min >= bbox_u_max or bbox_v_min >= bbox_v_max:
                    continue

                for v_pix in range(bbox_v_min, bbox_v_max):
                    dy = v_pix - v
                    for u_pix in range(bbox_u_min, bbox_u_max):
                        dx = u_pix - u

                        power = -0.5 * (
                            dx*dx*inv_cov2d[0,0] +
                            2*dx*dy*inv_cov2d[0,1] +
                            dy*dy*inv_cov2d[1,1]
                        )

                        if power < -10.0:
                            continue

                        weight = op * np.exp(power)

                        one_minus_alpha = 1.0 - alpha_buffer[v_pix, u_pix]
                        contribution = weight * one_minus_alpha

                        color_buffer[v_pix, u_pix] += contribution * col
                        alpha_buffer[v_pix, u_pix] += contribution

            color_buffer = np.clip(color_buffer, 0.0, 1.0)

            if return_uint8:
                results[img_name] = (color_buffer * 255).astype(np.uint8)
            else:
                results[img_name] = color_buffer

        return results

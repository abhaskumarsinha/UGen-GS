import numpy as np
import plotly.graph_objects as go
import cv2
from dataclasses import dataclass
from typing import Optional


# ============================================================
# Config Dataclass
# ============================================================

@dataclass
class SfMVisualizerConfig:
    point_size: int = 2
    camera_scale: Optional[float] = None

    # Layout parameters
    width: int = 800
    height: int = 600
    title: str = "SfM Reconstruction"

    # Axis display
    show_camera_axes: bool = True
    show_camera_centers: bool = True


# ============================================================
# Visualizer Class
# ============================================================

class SfMVisualizer:

    def __init__(self, config: SfMVisualizerConfig):
        self.cfg = config

    def visualize(self,
                  cam_poses,
                  points_3d,
                  tracks,
                  track_to_point,
                  gaussians_per_view=None):

        cfg = self.cfg

        # --------------------------------------------------
        # Convert points to numpy
        # --------------------------------------------------
        if isinstance(points_3d, list):
            points_3d = np.array(points_3d)

        if points_3d.size == 0:
            print("No 3D points to visualize.")
            return

        # --------------------------------------------------
        # Scene scaling
        # --------------------------------------------------
        scene_center = points_3d.mean(axis=0)
        scene_extent = np.max(points_3d, axis=0) - np.min(points_3d, axis=0)
        scene_diag = np.linalg.norm(scene_extent)

        camera_scale = cfg.camera_scale
        if camera_scale is None:
            camera_scale = 0.05 * scene_diag

        # --------------------------------------------------
        # Prepare point colors
        # --------------------------------------------------
        if gaussians_per_view is not None and len(points_3d) > 0:

            point_colors = np.zeros((len(points_3d), 3))
            point_counts = np.zeros(len(points_3d))

            for track_id, point_id in enumerate(track_to_point):
                if point_id == -1:
                    continue
                for view, idx in tracks[track_id]:
                    color = gaussians_per_view[view][idx].color
                    point_colors[point_id] += color
                    point_counts[point_id] += 1

            valid = point_counts > 0
            point_colors[valid] /= point_counts[valid, np.newaxis]
            point_colors[~valid] = 0.5
            point_colors = np.clip(point_colors, 0, 1)

        else:
            point_colors = np.zeros((len(points_3d), 3))
            for track_id, point_id in enumerate(track_to_point):
                if point_id != -1:
                    length = len(tracks[track_id])
                    hue = (length % 10) / 10.0
                    import matplotlib.cm as cm
                    rgb = cm.tab10(hue)[:3]
                    point_colors[point_id] = rgb

            if np.sum(point_colors) == 0:
                point_colors = None

        # --------------------------------------------------
        # Create figure
        # --------------------------------------------------
        fig = go.Figure()

        # --------------------------------------------------
        # Add 3D points
        # --------------------------------------------------
        if point_colors is not None:
            color_strs = [
                f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
                for c in point_colors
            ]
            fig.add_trace(go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1],
                z=points_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=cfg.point_size,
                    color=color_strs,
                    opacity=0.8
                ),
                name='3D points'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1],
                z=points_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=cfg.point_size,
                    color='blue',
                    opacity=0.8
                ),
                name='3D points'
            ))

        # --------------------------------------------------
        # Camera axes
        # --------------------------------------------------
        if cfg.show_camera_axes:

            x_lines = []
            y_lines = []
            z_lines = []

            for R, t in cam_poses:

                C = -R.T @ t
                C = C.flatten()

                right = R[0, :]
                down = R[1, :]
                forward = R[2, :]

                tip_x = C + camera_scale * right
                tip_y = C + camera_scale * down
                tip_z = C + camera_scale * forward

                x_lines.extend([C, tip_x, [None, None, None]])
                y_lines.extend([C, tip_y, [None, None, None]])
                z_lines.extend([C, tip_z, [None, None, None]])

            if x_lines:
                x_lines = np.array(x_lines)
                fig.add_trace(go.Scatter3d(
                    x=x_lines[:, 0],
                    y=x_lines[:, 1],
                    z=x_lines[:, 2],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False
                ))

            if y_lines:
                y_lines = np.array(y_lines)
                fig.add_trace(go.Scatter3d(
                    x=y_lines[:, 0],
                    y=y_lines[:, 1],
                    z=y_lines[:, 2],
                    mode='lines',
                    line=dict(color='green', width=2),
                    showlegend=False
                ))

            if z_lines:
                z_lines = np.array(z_lines)
                fig.add_trace(go.Scatter3d(
                    x=z_lines[:, 0],
                    y=z_lines[:, 1],
                    z=z_lines[:, 2],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))

        # --------------------------------------------------
        # Camera centers
        # --------------------------------------------------
        if cfg.show_camera_centers:
            centers = np.array([-R.T @ t for R, t in cam_poses]).reshape(-1, 3)

            fig.add_trace(go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode='markers',
                marker=dict(size=4, color='black'),
                name='Cameras'
            ))

        # --------------------------------------------------
        # Layout
        # --------------------------------------------------
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=cfg.width,
            height=cfg.height,
            margin=dict(l=0, r=0, b=0, t=30),
            title=cfg.title
        )

        fig.show()

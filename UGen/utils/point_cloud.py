from dataclasses import dataclass
from typing import Optional

@dataclass
class PlotlyVisualizerConfig:
    point_size: float = 2.0
    camera_size: Optional[float] = None
    show_cameras: bool = True
    title: str = "3D Gaussians and Cameras"

import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple


class PlotlyGaussianVisualizer:

    def __init__(self, config: PlotlyVisualizerConfig):
        self.cfg = config

    def visualize(
        self,
        gaussians_3d,
        camera_poses
    ) -> go.Figure:
        """
        Create an interactive Plotly 3D visualization of Gaussian points and camera poses.
        """

        point_size = self.cfg.point_size
        camera_size = self.cfg.camera_size
        show_cameras = self.cfg.show_cameras
        title = self.cfg.title

        # ------------------------------------------------------------
        # Extract point coordinates, colors, opacities
        # ------------------------------------------------------------
        points = []
        colors_rgb = []
        opacities = []

        for g in gaussians_3d:
            points.append(g.mean)

            col = np.array(g.color, dtype=float)
            if col.max() > 1.0:
                col = col / 255.0

            colors_rgb.append(col)
            opacities.append(g.opacity)

        points = np.array(points)
        colors_rgb = np.array(colors_rgb)
        opacities = np.array(opacities)

        # ------------------------------------------------------------
        # Compute scene bounds for auto-scaling
        # ------------------------------------------------------------
        if len(points) > 0:
            min_xyz = points.min(axis=0)
            max_xyz = points.max(axis=0)
            center = (min_xyz + max_xyz) / 2
            diameter = np.linalg.norm(max_xyz - min_xyz)
            if camera_size is None:
                camera_size = diameter * 0.05
        else:
            center = np.zeros(3)
            diameter = 1.0
            camera_size = 0.1

        # ------------------------------------------------------------
        # Build point trace
        # ------------------------------------------------------------
        point_trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=[
                    'rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255))
                    for r, g, b in colors_rgb
                ],
                opacity=opacities.mean() if len(opacities) > 0 else 1.0,
                line=dict(width=0)
            ),
            name='Gaussians'
        )

        traces = [point_trace]

        # ------------------------------------------------------------
        # Cameras
        # ------------------------------------------------------------
        if show_cameras and camera_poses:

            cam_centers = []
            cam_x_lines = []
            cam_y_lines = []
            cam_z_lines = []

            for R, t in camera_poses:

                if t.shape == (3,):
                    t = t.reshape(3, 1)

                C = -R.T @ t
                C = C.flatten()
                cam_centers.append(C)

                x_axis = R.T[:, 0]
                y_axis = R.T[:, 1]
                z_axis = R.T[:, 2]

                cam_x_lines.append((C, C + camera_size * x_axis))
                cam_y_lines.append((C, C + camera_size * y_axis))
                cam_z_lines.append((C, C + camera_size * z_axis))

            cam_center_trace = go.Scatter3d(
                x=[c[0] for c in cam_centers],
                y=[c[1] for c in cam_centers],
                z=[c[2] for c in cam_centers],
                mode='markers',
                marker=dict(size=point_size*3, color='black'),
                name='Cameras'
            )
            traces.append(cam_center_trace)

            def create_axis_trace(segments, color, name):
                xs, ys, zs = [], [], []
                for start, end in segments:
                    xs.extend([start[0], end[0], None])
                    ys.extend([start[1], end[1], None])
                    zs.extend([start[2], end[2], None])

                return go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=name,
                    showlegend=False
                )

            traces.append(create_axis_trace(cam_x_lines, 'red', 'X'))
            traces.append(create_axis_trace(cam_y_lines, 'green', 'Y'))
            traces.append(create_axis_trace(cam_z_lines, 'blue', 'Z'))

        # ------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------
        fig = go.Figure(data=traces)

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True
        )

        return fig

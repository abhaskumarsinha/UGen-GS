import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

def write_gaussians_to_ply(gaussians, output_path, verify=True):
    """
    Convert a list of Gaussian3D objects to a binary PLY file.
    Follows the format of the official 3D Gaussian Splatting code.

    Args:
        gaussians: List of objects with attributes:
            .mean (3-element array-like)
            .cov (3x3 covariance matrix)
            .color (3-element array-like, values in [0,1])
            .opacity (float in [0,1])
        output_path: Path where the .ply file will be saved.
        verify: If True, read back the file and print the vertex count.
    """
    num = len(gaussians)
    if num == 0:
        print("Warning: Input list is empty. No vertices will be written.")
        return

    print(f"Converting {num} Gaussians to PLY format...")

    # Define the 62 fields in exact order
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    for i in range(45):
        dtype_list.append((f'f_rest_{i}', 'f4'))
    dtype_list.append(('opacity', 'f4'))
    dtype_list.append(('scale_0', 'f4'))
    dtype_list.append(('scale_1', 'f4'))
    dtype_list.append(('scale_2', 'f4'))
    dtype_list.append(('rot_0', 'f4'))
    dtype_list.append(('rot_1', 'f4'))
    dtype_list.append(('rot_2', 'f4'))
    dtype_list.append(('rot_3', 'f4'))

    data = np.zeros(num, dtype=dtype_list)

    factor = 2.0 * np.sqrt(np.pi)   # ≈ 3.5449077

    for i, g in enumerate(gaussians):
        # ---- Position ----
        data['x'][i], data['y'][i], data['z'][i] = g.mean

        # ---- Normals (unused, set to 0) ----
        data['nx'][i] = data['ny'][i] = data['nz'][i] = 0.0

        # ---- SH DC coefficients from RGB ----
        color = np.asarray(g.color, dtype=np.float32)
        f_dc = (color - 0.5) * factor
        data['f_dc_0'][i], data['f_dc_1'][i], data['f_dc_2'][i] = f_dc

        # ---- Higher SH bands (all zero) ----
        # (already initialized to zero)

        # ---- Opacity (convert to logit) ----
        opacity = np.clip(g.opacity, 1e-6, 1.0 - 1e-6)   # avoid division by zero
        logit_opacity = np.log(opacity / (1.0 - opacity)).astype(np.float32)
        data['opacity'][i] = logit_opacity

        # ---- Covariance decomposition using SVD for stability ----
        cov = np.asarray(g.cov, dtype=np.float64)
        # Ensure symmetry
        cov = (cov + cov.T) / 2.0

        # SVD: cov = U @ diag(S) @ Vt, but since cov is symmetric, U = V
        U, S, _ = np.linalg.svd(cov)
        # S are eigenvalues (variances)
        S = np.clip(S, 1e-10, None)          # avoid negative due to numerical issues
        scales = np.sqrt(S).astype(np.float64)          # standard deviations
        log_scales = np.log(scales).astype(np.float32)

        # Rotation matrix = U, ensure determinant +1 (proper rotation)
        R = U
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1   # flip last column

        # Convert rotation matrix to quaternion in (w, x, y, z) order
        quat_xyzw = Rotation.from_matrix(R).as_quat()   # returns (x, y, z, w)
        quat_wxyz = np.roll(quat_xyzw, shift=1)         # now (w, x, y, z)

        data['scale_0'][i], data['scale_1'][i], data['scale_2'][i] = log_scales
        data['rot_0'][i], data['rot_1'][i], data['rot_2'][i], data['rot_3'][i] = quat_wxyz

    # ---- Write binary PLY ----
    vertex_element = PlyElement.describe(data, 'vertex')
    ply_data = PlyData([vertex_element], text=False, byte_order='<')
    ply_data.write(output_path)
    print(f"File written: {output_path}")

    if verify:
        # Read back and check vertex count
        try:
            ply_read = PlyData.read(output_path)
            print(f"Verification: PLY header shows {ply_read['vertex'].count} vertices.")
        except Exception as e:
            print(f"Verification failed: {e}")


import numpy as np
from plyfile import PlyData
import os

class Gaussian3D:
    """Simple container for a 3D Gaussian."""
    def __init__(self, mean, cov, color, opacity):
        self.mean = mean      # (3,) numpy array
        self.cov = cov        # (3,3) numpy array
        self.color = color    # (3,) numpy array in [0,1]
        self.opacity = opacity # float in [0,1]

def load_ply_to_gaussians(ply_path):
    """
    Load a PLY file (binary little‑endian) exported by the official
    3D Gaussian Splatting code and return a list of Gaussian3D objects.

    The PLY must contain the following properties per vertex:
        x, y, z                     (position)
        nx, ny, nz                   (normals, ignored)
        f_dc_0, f_dc_1, f_dc_2       (SH DC coefficients)
        f_rest_0 ... f_rest_44        (higher SH coefficients, ignored here)
        opacity                       (logit of opacity)
        scale_0, scale_1, scale_2     (logarithm of scales)
        rot_0, rot_1, rot_2, rot_3    (rotation quaternion in w,x,y,z order)
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    num_vertices = len(vertex)
    gaussians = []

    # Pre‑compute constants
    FACTOR = 2.0 * np.sqrt(np.pi)          # used for SH DC → RGB conversion

    for i in range(num_vertices):
        # ---- Position ----
        mean = np.array([vertex['x'][i], vertex['y'][i], vertex['z'][i]], dtype=np.float64)

        # ---- SH DC → RGB ----
        f_dc = np.array([vertex['f_dc_0'][i], vertex['f_dc_1'][i], vertex['f_dc_2'][i]], dtype=np.float32)
        rgb = (f_dc / FACTOR) + 0.5
        rgb = np.clip(rgb, 0.0, 1.0)        # ensure valid range

        # ---- Opacity (logit → probability) ----
        logit = vertex['opacity'][i]
        opacity = 1.0 / (1.0 + np.exp(-logit))

        # ---- Scales (log → linear) ----
        log_scales = np.array([vertex['scale_0'][i], vertex['scale_1'][i], vertex['scale_2'][i]], dtype=np.float64)
        scales = np.exp(log_scales)

        # ---- Rotation quaternion (w, x, y, z → rotation matrix) ----
        qw = vertex['rot_0'][i]
        qx = vertex['rot_1'][i]
        qy = vertex['rot_2'][i]
        qz = vertex['rot_3'][i]

        # Normalize quaternion to avoid numerical issues
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm < 1e-12:
            # Degenerate quaternion – use identity
            R = np.eye(3)
        else:
            qw /= norm
            qx /= norm
            qy /= norm
            qz /= norm

            # Rotation matrix from quaternion (w, x, y, z)
            R = np.array([
                [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
            ])

        # ---- Covariance matrix: R @ diag(scales**2) @ R.T ----
        S = np.diag(scales ** 2)
        cov = R @ S @ R.T

        # ---- Create Gaussian3D object ----
        gaussians.append(Gaussian3D(mean, cov, rgb, opacity))

    return gaussians


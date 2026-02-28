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

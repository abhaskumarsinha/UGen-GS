import keras
from keras import layers, initializers, ops
import numpy as np


class CameraParameterLayer(layers.Layer):
    """
    Keras layer that owns camera parameters and optionally
    makes subsets trainable. Includes view and projection matrix
    computations for COLMAP-style cameras.
    """

    def __init__(
        self,
        cameras_json: list[dict],
        train_intrinsics: bool = False,
        train_rotation: bool = False,
        train_translation: bool = False,
        name="camera_parameters",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        assert len(cameras_json) > 0, "Empty camera list"

        self.N = len(cameras_json)

        # -------- Metadata --------
        self.widths = [c["width"] for c in cameras_json]
        self.heights = [c["height"] for c in cameras_json]

        # -------- Parse parameters --------
        fx, fy, cx, cy = [], [], [], []
        rotations = []
        translations = []
        images = []

        for c in cameras_json:
            fx.append(c["fx"])
            fy.append(c["fy"])
            cx.append(c["cx"])
            cy.append(c["cy"])

            rotations.append(c["rotation"])        # quat (4,)
            translations.append(c["translation"])  # (3,)
            images.append(c["image"])               # (H, W, 3)

        # -------- Convert to arrays --------
        self.fx_init = np.asarray(fx, dtype=np.float32).reshape(self.N, 1)
        self.fy_init = np.asarray(fy, dtype=np.float32).reshape(self.N, 1)
        self.cx_init = np.asarray(cx, dtype=np.float32).reshape(self.N, 1)
        self.cy_init = np.asarray(cy, dtype=np.float32).reshape(self.N, 1)

        self.rot_init = np.asarray(rotations, dtype=np.float32)      # (N, 4)
        self.trans_init = np.asarray(translations, dtype=np.float32) # (N, 3)

        # Images → constant tensor
        self.images = keras.Variable(
            np.asarray(images, dtype=np.float32),
            name="camera_images",
            trainable=False
        )

        # -------- Trainability flags --------
        self.train_intrinsics = train_intrinsics
        self.train_rotation = train_rotation
        self.train_translation = train_translation

        self.fx = self.add_weight(
            name="fx",
            shape=(self.N, 1),
            initializer=initializers.Constant(self.fx_init),
            trainable=self.train_intrinsics,
        )

        self.fy = self.add_weight(
            name="fy",
            shape=(self.N, 1),
            initializer=initializers.Constant(self.fy_init),
            trainable=self.train_intrinsics,
        )

        self.cx = self.add_weight(
            name="cx",
            shape=(self.N, 1),
            initializer=initializers.Constant(self.cx_init),
            trainable=self.train_intrinsics,
        )

        self.cy = self.add_weight(
            name="cy",
            shape=(self.N, 1),
            initializer=initializers.Constant(self.cy_init),
            trainable=self.train_intrinsics,
        )

        # -------- Extrinsics --------
        self.rotation = self.add_weight(
            name="rotation_quat",
            shape=(self.N, 4),
            initializer=initializers.Constant(self.rot_init),
            trainable=self.train_rotation,
        )

        self.translation = self.add_weight(
            name="translation",
            shape=(self.N, 3),
            initializer=initializers.Constant(self.trans_init),
            trainable=self.train_translation,
        )

    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert quaternion (w, x, y, z) to rotation matrix.
        Assumes quaternion is normalized.
        """
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Compute rotation matrix from quaternion
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        
        # Build rotation matrix columns
        col0 = ops.stack([
            1 - 2 * (yy + zz),
            2 * (xy + wz),
            2 * (xz - wy)
        ], axis=-1)
        
        col1 = ops.stack([
            2 * (xy - wz),
            1 - 2 * (xx + zz),
            2 * (yz + wx)
        ], axis=-1)
        
        col2 = ops.stack([
            2 * (xz + wy),
            2 * (yz - wx),
            1 - 2 * (xx + yy)
        ], axis=-1)
        
        # Stack columns to form (..., 3, 3) matrix
        R = ops.stack([col0, col1, col2], axis=-1)
        return R

    def compute_view_matrix(self):
        """
        Compute view matrix (world to camera) for each camera.
        
        COLMAP uses: [R | t] where R is the rotation matrix (camera to world)
        and t is the translation (camera position in world coordinates).
        
        The view matrix transforms world coordinates to camera coordinates:
        view = [R^T | -R^T * t] (4x4)
        """
        # Get quaternions and translations
        quat = self.rotation  # (N, 4)
        t = self.translation  # (N, 3)
        
        # Convert quaternion to rotation matrix (camera to world)
        R_c2w = self.quaternion_to_rotation_matrix(quat)  # (N, 3, 3)
        
        # Compute world to camera rotation (transpose)
        R_w2c = ops.transpose(R_c2w, axes=(0, 2, 1))  # (N, 3, 3)
        
        # Compute camera position in world coordinates
        t_expanded = ops.expand_dims(t, axis=-1)  # (N, 3, 1)
        camera_translation = -ops.matmul(R_w2c, t_expanded)  # (N, 3, 1)
        camera_translation = ops.squeeze(camera_translation, axis=-1)  # (N, 3)
        
        # Build 4x4 view matrix using concatenation
        # Create the top part [R_w2c | camera_translation]
        camera_translation_expanded = ops.expand_dims(camera_translation, axis=-1)  # (N, 3, 1)
        top = ops.concatenate([R_w2c, camera_translation_expanded], axis=-1)  # (N, 3, 4)
        
        # Create the bottom row [0, 0, 0, 1]
        bottom = ops.zeros((self.N, 1, 3), dtype='float32')
        bottom_one = ops.ones((self.N, 1, 1), dtype='float32')
        bottom_row = ops.concatenate([bottom, bottom_one], axis=-1)  # (N, 1, 4)
        
        # Concatenate top and bottom
        view_matrix = ops.concatenate([top, bottom_row], axis=1)  # (N, 4, 4)
        
        return view_matrix

    def compute_projection_matrix(self, near=0.1, far=100.0):
        """
        Compute projection matrix for each camera.
        
        COLMAP uses pinhole camera model with focal length and principal point.
        For rendering/OpenGL-style projection, we convert to perspective projection.
        """
        # Get intrinsic parameters
        fx = self.fx  # (N, 1)
        fy = self.fy  # (N, 1)
        cx = self.cx  # (N, 1)
        cy = self.cy  # (N, 1)
        
        # Get image dimensions as tensors
        widths = ops.convert_to_tensor(self.widths, dtype='float32')  # (N,)
        heights = ops.convert_to_tensor(self.heights, dtype='float32')  # (N,)
        widths = ops.expand_dims(widths, axis=-1)  # (N, 1)
        heights = ops.expand_dims(heights, axis=-1)  # (N, 1)
        
        # Convert near/far to tensors with batch dimension
        near_tensor = ops.full((self.N, 1), near, dtype='float32')
        far_tensor = ops.full((self.N, 1), far, dtype='float32')
        z_range = far_tensor - near_tensor  # (N, 1)
        
        # Create matrix rows
        row0 = ops.concatenate([
            2.0 * fx / widths,
            ops.zeros_like(fx),
            (widths - 2.0 * cx) / widths,
            ops.zeros_like(fx)
        ], axis=-1)  # (N, 4)
        
        row1 = ops.concatenate([
            ops.zeros_like(fy),
            2.0 * fy / heights,
            (heights - 2.0 * cy) / heights,
            ops.zeros_like(fy)
        ], axis=-1)  # (N, 4)
        
        row2 = ops.concatenate([
            ops.zeros_like(fx),
            ops.zeros_like(fx),
            -(far_tensor + near_tensor) / z_range,
            -2.0 * far_tensor * near_tensor / z_range
        ], axis=-1)  # (N, 4)
        
        row3 = ops.concatenate([
            ops.zeros_like(fx),
            ops.zeros_like(fx),
            -ops.ones_like(fx),
            ops.zeros_like(fx)
        ], axis=-1)  # (N, 4)
        
        # Stack rows to create (N, 4, 4) matrix
        proj_matrix = ops.stack([row0, row1, row2, row3], axis=1)
        
        return proj_matrix

    def compute_camera_center(self):
        """
        Compute camera center in world coordinates.
        
        For COLMAP format: camera center = -R^T * t
        where R is rotation (camera to world), t is translation.
        """
        quat = self.rotation  # (N, 4)
        t = self.translation  # (N, 3)
        
        # Convert quaternion to rotation matrix (camera to world)
        R_c2w = self.quaternion_to_rotation_matrix(quat)  # (N, 3, 3)
        
        # Compute world to camera rotation (transpose)
        R_w2c = ops.transpose(R_c2w, axes=(0, 2, 1))  # (N, 3, 3)
        
        # Camera center in world: c = -R^T * t
        camera_center = -ops.matmul(R_w2c, ops.expand_dims(t, axis=-1))  # (N, 3, 1)
        camera_center = ops.squeeze(camera_center, axis=-1)  # (N, 3)
        
        return camera_center

    def compute_intrinsics_matrix(self):
        """
        Compute 3x3 intrinsic matrix for each camera.
        """
        # Create zeros tensors
        zeros = ops.zeros_like(self.fx)
        ones = ops.ones_like(self.fx)
        
        # Build intrinsic matrix rows
        row0 = ops.concatenate([self.fx, zeros, self.cx], axis=-1)  # (N, 3)
        row1 = ops.concatenate([zeros, self.fy, self.cy], axis=-1)  # (N, 3)
        row2 = ops.concatenate([zeros, zeros, ones], axis=-1)       # (N, 3)
        
        # Stack rows to create (N, 3, 3) matrix
        intrinsics = ops.stack([row0, row1, row2], axis=1)
        
        return intrinsics

    def call(self, inputs=None, near=0.1, far=100.0):
        """
        Acts as a parameter container.
        
        Args:
            inputs: unused, for API compatibility
            near: near clipping plane for projection matrix
            far: far clipping plane for projection matrix
            
        Returns:
            Dictionary containing all camera parameters and matrices
        """
        # Compute matrices
        view_matrix = self.compute_view_matrix()  # (N, 4, 4)
        projection_matrix = self.compute_projection_matrix(near, far)  # (N, 4, 4)
        camera_center = self.compute_camera_center()  # (N, 3)
        intrinsics_matrix = self.compute_intrinsics_matrix()  # (N, 3, 3)
        
        # Convert widths and heights to tensors for consistency
        widths_tensor = ops.convert_to_tensor(self.widths, dtype='int32')
        heights_tensor = ops.convert_to_tensor(self.heights, dtype='int32')
        
        return {
            "fx": self.fx,               # (N, 1)
            "fy": self.fy,               # (N, 1)
            "cx": self.cx,               # (N, 1)
            "cy": self.cy,               # (N, 1)
            "rotation": self.rotation,   # (N, 4)
            "translation": self.translation,  # (N, 3)
            "width": widths_tensor,      # (N,)
            "height": heights_tensor,    # (N,)
            "image": self.images,        # (N, H, W, 3)
            "view_matrix": view_matrix,  # (N, 4, 4)
            "projection_matrix": projection_matrix,  # (N, 4, 4)
            "camera_center": camera_center,  # (N, 3)
            "intrinsics_matrix": intrinsics_matrix,  # (N, 3, 3)
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "N": self.N,
                "train_intrinsics": self.train_intrinsics,
                "train_rotation": self.train_rotation,
                "train_translation": self.train_translation,
            }
        )
        return config
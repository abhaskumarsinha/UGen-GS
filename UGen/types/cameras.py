from keras import layers
import keras
from keras import ops

class CameraLayer(layers.Layer):
    """
    Convention-free camera container.
    Stores pose + image, does NOT define intrinsics or projection.
    """

    def __init__(self, camera_dict, trainable_extrinsics=False, **kwargs):
        super().__init__(**kwargs)

        # ---- Metadata ----
        self.width  = camera_dict["width"]
        self.height = camera_dict["height"]
        self.fx, self.fy = camera_dict['fx'], camera_dict['fy']
        self.cx, self.cy = camera_dict['cx'], camera_dict['cy']

        # ---- Pose ----
        # Quaternion: [x, y, z, w]
        self.rotation = keras.Variable(
            camera_dict["rotation"],
            trainable=trainable_extrinsics,
            dtype='float32',
            name="camera_quaternion"
        )

        self.translation = keras.Variable(
            camera_dict["translation"],
            trainable=trainable_extrinsics,
            dtype='float32',
            name="camera_translation"
        )

        # ---- Image (non-trainable) ----
        self.image = keras.Variable(
            camera_dict["image"],
            dtype='float32',
            name="camera_image",
            trainable=False
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


    def call(self, inputs=None):
        """
        Returns a camera dict without assuming any projection model.
        """
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "rotation_quaternion": keras.ops.convert_to_tensor(self.rotation),
            "rotation_matrix": keras.ops.convert_to_tensor(self.quaternion_to_rotation_matrix(self.rotation)),
            "translation": keras.ops.convert_to_tensor(self.translation),
            "image": keras.ops.convert_to_tensor(self.image),
        }

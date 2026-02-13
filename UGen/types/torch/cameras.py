import torch
import torch.nn as nn


class CameraLayer(nn.Module):
    """
    Convention-free camera container (PyTorch replacement).
    Stores pose + image, does NOT define intrinsics or projection.
    """

    def __init__(self, camera_dict, trainable_extrinsics=False, **kwargs):
        super().__init__()

        # ---- Metadata ----
        self.width  = camera_dict["width"]
        self.height = camera_dict["height"]
        self.fx = camera_dict["fx"]
        self.fy = camera_dict["fy"]
        self.cx = camera_dict["cx"]
        self.cy = camera_dict["cy"]

        # ---- Pose ----
        # Quaternion format assumed: [w, x, y, z]
        rotation = torch.as_tensor(camera_dict["rotation"], dtype=torch.float32)
        translation = torch.as_tensor(camera_dict["translation"], dtype=torch.float32)

        self.rotation = nn.Parameter(
            rotation,
            requires_grad=bool(trainable_extrinsics)
        )

        self.translation = nn.Parameter(
            translation,
            requires_grad=bool(trainable_extrinsics)
        )

        # ---- Image (non-trainable) ----
        image = torch.as_tensor(camera_dict["image"], dtype=torch.float32)
        self.image = nn.Parameter(
            image,
            requires_grad=False
        )

    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert quaternion (w, x, y, z) to rotation matrix.
        Assumes quaternion is normalized.
        Supports shape (..., 4).
        """
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        col0 = torch.stack([
            1 - 2 * (yy + zz),
            2 * (xy + wz),
            2 * (xz - wy)
        ], dim=-1)

        col1 = torch.stack([
            2 * (xy - wz),
            1 - 2 * (xx + zz),
            2 * (yz + wx)
        ], dim=-1)

        col2 = torch.stack([
            2 * (xz + wy),
            2 * (yz - wx),
            1 - 2 * (xx + yy)
        ], dim=-1)

        # Stack columns → (..., 3, 3)
        R = torch.stack([col0, col1, col2], dim=-1)
        return R

    def forward(self, inputs=None):
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
            "rotation_quaternion": self.rotation,
            "rotation_matrix": self.quaternion_to_rotation_matrix(self.rotation),
            "translation": self.translation,
            "image": self.image,
        }

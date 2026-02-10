from dataclasses import dataclass
from UGen.decoder.base import *
import torch


@dataclass
class GaussianRasterizerConfig:
    znear: float = 0.01
    zfar: float = 100.0
    scale_modifier: float = 1.0
    sh_degree: int = 0
    output_format: str = "HWC"  # "HWC" or "CHW"
    double_resolution: bool = True
    background_color: tuple = (0.0, 0.0, 0.0)


import math
import torch
from abc import ABC
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

SH0 = 0.282095


class GaussianRasterizerDecoder(BaseRasterizer):
    """
    Gaussian rasterizer decoder using diff-gaussian-rasterization.

    Takes:
        cam      = CameraLayer(...).call()
        gaussian = GaussianLayer(...).call()

    Returns:
        torch.Tensor (H,W,3) or (3,H,W)
    """

    def __init__(self, width: int, height: int, config: GaussianRasterizerConfig):
        super().__init__(width, height, config.background_color)
        self.config = config

    # ---------------------------------------------------------
    # Math helpers (unchanged logic)
    # ---------------------------------------------------------

    def quat_to_rot_matrix(self, q: torch.Tensor):
        q = q / q.norm(dim=-1, keepdim=True)
        qw, qx, qy, qz = q.unbind(-1)

        R = torch.empty((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)

        R[..., 0, 0] = 1 - 2 * (qy*qy + qz*qz)
        R[..., 0, 1] = 2 * (qx*qy - qz*qw)
        R[..., 0, 2] = 2 * (qx*qz + qy*qw)

        R[..., 1, 0] = 2 * (qx*qy + qz*qw)
        R[..., 1, 1] = 1 - 2 * (qx*qx + qz*qz)
        R[..., 1, 2] = 2 * (qy*qz - qx*qw)

        R[..., 2, 0] = 2 * (qx*qz - qy*qw)
        R[..., 2, 1] = 2 * (qy*qz + qx*qw)
        R[..., 2, 2] = 1 - 2 * (qx*qx + qy*qy)

        return R

    def getWorld2View2(self, R, t, translate=None, scale=1.0):
        device, dtype = R.device, R.dtype

        Rt = torch.eye(4, device=device, dtype=dtype)
        Rt[:3, :3] = R.transpose(0, 1)
        Rt[:3, 3] = t

        C2W = torch.linalg.inv(Rt)
        cam_center = C2W[:3, 3]

        if translate is not None:
            cam_center = (cam_center + translate) * scale

        C2W[:3, 3] = cam_center
        Rt = torch.linalg.inv(C2W)
        return Rt

    def getProjectionMatrix(self, znear, zfar, fovX, fovY, device):
        tanHalfFovY = torch.tan(fovY * 0.5)
        tanHalfFovX = torch.tan(fovX * 0.5)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros((4, 4), device=device, dtype=torch.float32)
        z_sign = 1.0

        P[0, 0] = 2 * znear / (right - left)
        P[1, 1] = 2 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P

    def rgb_to_sh(self, rgb, sh_degree):
        N = rgb.shape[0]
        num_basis = (sh_degree + 1) ** 2

        shs = torch.zeros((N, num_basis, 3),
                          dtype=rgb.dtype,
                          device=rgb.device)

        shs[:, 0, :] = rgb / SH0
        return shs.view(N, -1)

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------
    
    def render(self, cam, gaussian):
        # Clear CUDA cache to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Get device from gaussian data
        device = gaussian["position"].device
        
        # Ensure all camera parameters are on the same device and type
        qvec = cam["rotation_quaternion"].to(device).to(torch.float32)
        tvec = cam["translation"].to(device).to(torch.float32)
        
        # Camera matrix calculations
        R = self.quat_to_rot_matrix(qvec).transpose(0, 1)
        view_mat = self.getWorld2View2(R, tvec)
        
        w, h = cam["width"], cam["height"]
        fx, fy = cam["fx"], cam["fy"]
        
        # Ensure fov calculations are on the correct device
        fovX = 2.0 * torch.atan(torch.tensor((w * 0.5) / fx, device=device))
        fovY = 2.0 * torch.atan(torch.tensor((h * 0.5) / fy, device=device))
        
        proj_mat = self.getProjectionMatrix(
            self.config.znear,
            self.config.zfar,
            fovX,
            fovY,
            device,
        )
        
        world_view_transform = view_mat.t().to(torch.float32)
        full_proj_transform = (world_view_transform @ proj_mat.t()).to(torch.float32)
        camera_center = (-R.T @ tvec).to(torch.float32)
        
        # Gaussian parameters - ensure proper types
        means3D = gaussian["position"].to(torch.float32)
        scales = (gaussian["scale"] * 0.001).to(torch.float32)
        opacities = gaussian["alpha"].squeeze(-1).clamp(0, 1).to(torch.float32)
        
        # Fix SH handling for degree 0
        # For degree 0, we should have shape (N, 3)
        if self.config.sh_degree == 0:
            # Direct RGB values, no conversion needed
            shs = gaussian["sh"].to(torch.float32)
            # Ensure it's the right shape: (N, 3)
            if len(shs.shape) == 1:
                shs = shs.unsqueeze(0)
        else:
            shs = self.rgb_to_sh(
                gaussian["sh"].to(torch.float32),
                gaussian["sh_degree"],
            ).to(torch.float32)
        
        rotations = self.quat_to_rot_matrix(
            gaussian["quat"].to(torch.float32)
        ).transpose(1, 2).to(torch.float32)
        
        # Validate dimensions
        assert means3D.dim() == 2 and means3D.shape[1] == 3, f"means3D shape error: {means3D.shape}"
        assert scales.dim() == 2 and scales.shape[1] == 3, f"scales shape error: {scales.shape}"
        assert opacities.dim() == 1, f"opacities shape error: {opacities.shape}"
        assert rotations.dim() == 3 and rotations.shape[1:] == (3, 3), f"rotations shape error: {rotations.shape}"
        
        # Raster settings
        tanfovx = math.tan(fovX.item() * 0.5)
        tanfovy = math.tan(fovY.item() * 0.5)
        
        H = int(h * 2) if self.config.double_resolution else int(h)
        W = int(w * 2) if self.config.double_resolution else int(w)
        
        # Create background tensor
        bg = torch.tensor(self.background_color, dtype=torch.float32, device=device)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=self.config.scale_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=self.config.sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )
        
        # Sync before creating rasterizer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        rasterizer = GaussianRasterizer(raster_settings)
        
        try:
            rendered_image, _ = rasterizer(
                means3D=means3D,
                means2D=None,
                shs=shs,
                colors_precomp=None,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
        except Exception as e:
            print(f"Rasterization failed: {e}")
            print(f"Shapes - means3D: {means3D.shape}, shs: {shs.shape}")
            print(f"Shapes - scales: {scales.shape}, rotations: {rotations.shape}")
            raise
        
        # Sync after rendering
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        rendered_image = rendered_image.clamp(0, 1)
        
        if self.config.output_format.upper() == "HWC":
            rendered_image = rendered_image.permute(1, 2, 0)
        
        return rendered_image

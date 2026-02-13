import torch
import torch.nn as nn
import math

class GaussianLayer(nn.Module):
    """
    Convention-free Gaussian container (PyTorch replacement for Keras layer).
    Keeps the same attribute names and output dictionary keys as the Keras version.
    """

    def __init__(self, gaussians, sh_degree=None, trainable=True, **kwargs):
        super().__init__()

        self.N = len(gaussians)

        # collect arrays (robust to lists, numpy arrays, or torch tensors)
        positions = torch.as_tensor([g["position"] for g in gaussians], dtype=torch.float32)
        scales    = torch.as_tensor([g["scale"]    for g in gaussians], dtype=torch.float32)
        alphas    = torch.as_tensor([g["alpha"]    for g in gaussians], dtype=torch.float32).unsqueeze(1)
        quats     = torch.as_tensor([g["quat"]     for g in gaussians], dtype=torch.float32)
        sh_coeffs = torch.as_tensor([g["sh"]       for g in gaussians], dtype=torch.float32)

        # ---- Trainable Parameters ----
        self.position = nn.Parameter(positions, requires_grad=bool(trainable))
        self.scale    = nn.Parameter(scales,    requires_grad=bool(trainable))
        self.alpha    = nn.Parameter(alphas,    requires_grad=bool(trainable))
        self.quat     = nn.Parameter(quats,     requires_grad=bool(trainable))
        self.sh       = nn.Parameter(sh_coeffs, requires_grad=bool(trainable))

        # ---- Non-trainable 2D means (required by rasterizer) ----
        # Shape: (N, 2)
        mean2D = torch.zeros(self.N, 2, dtype=torch.float32)
        self.register_buffer("mean2D", mean2D)

        # ---- SH degree ----
        if sh_degree is None:
            self.sh_degree = self._infer_sh_degree(self.sh.shape[-1])
        else:
            self.sh_degree = sh_degree


    @staticmethod
    def _infer_sh_degree(num_coeffs):
        """
        Supports:
          - scalar SH: (L+1)^2
          - RGB SH: 3 * (L+1)^2
        """
        # RGB SH
        if num_coeffs % 3 == 0:
            base = num_coeffs // 3
            L = int(math.sqrt(base) - 1)
            if (L + 1) ** 2 == base:
                return L

        # Scalar SH
        L = int(math.sqrt(num_coeffs) - 1)
        if (L + 1) ** 2 == num_coeffs:
            return L

        raise ValueError(f"Invalid SH coefficient count: {num_coeffs}")


    def forward(self, inputs=None):
        return {
            "position": self.position,
            "scale": self.scale,
            "alpha": self.alpha,
            "quat": self.quat,
            "sh": self.sh,
            "sh_degree": self.sh_degree,
            "count": self.N,
            "mean2D": self.mean2D,  # <- added here
        }

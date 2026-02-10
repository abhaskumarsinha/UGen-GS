import keras
from keras import layers, ops, initializers
import math


class GaussianLayer(layers.Layer):
    """
    Convention-free Gaussian container (Keras 3 safe).
    """

    def __init__(self, gaussians, sh_degree=None, trainable=True, **kwargs):
        super().__init__(**kwargs)

        self.N = len(gaussians)

        positions = ops.array([g["position"] for g in gaussians], dtype="float32")
        scales    = ops.array([g["scale"]    for g in gaussians], dtype="float32")
        alphas    = ops.array([g["alpha"]    for g in gaussians], dtype="float32")[:, None]
        quats     = ops.array([g["quat"]     for g in gaussians], dtype="float32")
        sh_coeffs = ops.array([g["sh"]       for g in gaussians], dtype="float32")

        # ---- Parameters ----
        self.position = self.add_weight(
            shape=(self.N, 3),
            initializer=initializers.Constant(positions),
            trainable=trainable,
            name="position",
        )

        self.scale = self.add_weight(
            shape=(self.N, 3),
            initializer=initializers.Constant(scales),
            trainable=trainable,
            name="scale",
        )

        self.alpha = self.add_weight(
            shape=(self.N, 1),
            initializer=initializers.Constant(alphas),
            trainable=trainable,
            name="alpha",
        )

        self.quat = self.add_weight(
            shape=(self.N, 4),
            initializer=initializers.Constant(quats),
            trainable=trainable,
            name="quat",
        )

        self.sh = self.add_weight(
            shape=(self.N, sh_coeffs.shape[-1]),
            initializer=initializers.Constant(sh_coeffs),
            trainable=trainable,
            name="sh",
        )

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


    def call(self, inputs=None):
        return {
            "position": keras.ops.convert_to_tensor(self.position),
            "scale": keras.ops.convert_to_tensor(self.scale),
            "alpha": keras.ops.convert_to_tensor(self.alpha),
            "quat": keras.ops.convert_to_tensor(self.quat),
            "sh": keras.ops.convert_to_tensor(self.sh),
            "sh_degree": self.sh_degree,
            "count": self.N,
        }

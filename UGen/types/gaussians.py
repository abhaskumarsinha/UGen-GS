from keras import layers, initializers
import numpy as np


def infer_sh_order(sh_len: int) -> int:
    """
    Infer SH order l from flattened SH length.
    sh_len = 3 * (l + 1)^2
    """
    l = int(np.sqrt(sh_len / 3) - 1)
    assert 3 * (l + 1) ** 2 == sh_len, \
        f"Invalid SH length: {sh_len}"
    return l


class GaussianParameterLayer(layers.Layer):
    """
    Keras layer that owns Gaussian parameters and optionally
    initializes them from JSON-like dictionaries.
    """

    def __init__(
        self,
        gaussians_json: list[dict],
        trainable: bool = True,
        name="gaussian_parameters",
        **kwargs,
    ):
        super().__init__(name=name, trainable=trainable, **kwargs)

        assert len(gaussians_json) > 0, "Empty gaussian list"

        self.N = len(gaussians_json)

        # -------- Parse JSONs --------
        means = []
        scales = []
        alphas = []
        quats = []
        sh_coeffs = []

        for g in gaussians_json:
            means.append(g["position"])
            scales.append(g["scale"])
            alphas.append([g["alpha"]])
            quats.append(g["quat"])
            sh_coeffs.append(g["sh"])

        self.means_init = np.asarray(means, dtype=np.float32)      # (N, 3)
        self.scales_init = np.asarray(scales, dtype=np.float32)    # (N, 3)
        self.alphas_init = np.asarray(alphas, dtype=np.float32)    # (N, 1)
        self.quats_init = np.asarray(quats, dtype=np.float32)      # (N, 4)
        self.sh_init = np.asarray(sh_coeffs, dtype=np.float32)     # (N, sh_l)

        self.sh_l = self.sh_init.shape[1]
        self.sh_order = infer_sh_order(self.sh_l)

        self.means = self.add_weight(
            name="means",
            shape=(self.N, 3),
            initializer=initializers.Constant(self.means_init),
            trainable=True,
        )

        self.scales = self.add_weight(
            name="scales",
            shape=(self.N, 3),
            initializer=initializers.Constant(self.scales_init),
            trainable=True,
        )

        self.alphas = self.add_weight(
            name="alphas",
            shape=(self.N, 1),
            initializer=initializers.Constant(self.alphas_init),
            trainable=True,
        )

        self.quats = self.add_weight(
            name="quaternions",
            shape=(self.N, 4),
            initializer=initializers.Constant(self.quats_init),
            trainable=True,
        )

        self.sh = self.add_weight(
            name="sh_coeffs",
            shape=(self.N, self.sh_l),
            initializer=initializers.Constant(self.sh_init),
            trainable=True,
        )

    def call(self, inputs=None):
        """
        Returns all Gaussian parameters.
        This layer ignores inputs and acts as a parameter container.
        """
        return {
            "mean": self.means,        # (N, 3)
            "scale": self.scales,      # (N, 3)
            "alpha": self.alphas,      # (N, 1)
            "quat": self.quats,        # (N, 4)
            "sh": self.sh,             # (N, sh_l)
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "N": self.N,
                "sh_l": self.sh_l,
                "sh_order": self.sh_order,
            }
        )
        return config

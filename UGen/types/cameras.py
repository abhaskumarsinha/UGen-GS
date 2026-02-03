import keras
from keras import layers, initializers
import numpy as np


class CameraParameterLayer(layers.Layer):
    """
    Keras layer that owns camera parameters and optionally
    makes subsets trainable.
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


    def call(self, inputs=None):
        """
        Acts as a parameter container.
        """
        return {
            "fx": self.fx,               # (N, 1)
            "fy": self.fy,               # (N, 1)
            "cx": self.cx,               # (N, 1)
            "cy": self.cy,               # (N, 1)
            "rotation": self.rotation,   # (N, 4)
            "translation": self.translation,  # (N, 3)
            "width": self.widths,        # list[int]
            "height": self.heights,      # list[int]
            "image": self.images,        # (N, H, W, 3)
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

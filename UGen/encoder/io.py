import os
import glob
from PIL import Image

def load_image(filepath, target_size=None, mode='RGB'):
    """
    Load an image from disk, optionally resize, and return as a float numpy array in [0,1].
    If target_size is (H, W), the image is resized (preserving aspect ratio? Usually we force exact size.)
    For simplicity, we force exact size using PIL's resize.
    """
    img = Image.open(filepath).convert(mode)
    if target_size is not None:
        img = img.resize(target_size[::-1], Image.BILINEAR)  # PIL size is (width, height)
    return np.array(img).astype(np.float32) / 255.0

def load_images_from_folder(folder, target_size=None, pattern='*.png', mode='RGB'):
    """
    Load all images matching pattern in folder, resize to target_size, return list of arrays and list of filenames.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    images = []
    names = []
    for p in paths:
        images.append(load_image(p, target_size, mode))
        names.append(os.path.basename(p))
    return images, names

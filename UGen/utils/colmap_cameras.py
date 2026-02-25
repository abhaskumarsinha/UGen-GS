from UGen.utils.unproj_gaussians import *


def import_colmap_cameras(
    colmap_path: str,
    target_resolution: Tuple[int, int] = None,
    add_identity_camera: bool = False
):
    """
    Import all camera parameters from COLMAP text files, optionally scale to a target resolution,
    and optionally add an extra camera with identity pose (world→camera rotation = I, translation = 0).

    Args:
        colmap_path: Path to the sparse folder (containing cameras.txt, images.txt).
        target_resolution: If provided, (new_width, new_height) to scale intrinsics.
        add_identity_camera: If True, add a camera named 'identity_view' with identity extrinsics,
                             using the intrinsics of the first camera in the COLMAP model.

    Returns:
        Dictionary mapping image filename to camera info. Keys include:
            - image_name : str
            - width, height : int (scaled if target_resolution provided)
            - fx, fy, cx, cy : float
            - rotation : 3x3 numpy array (world → camera rotation)
            - translation : 3-element numpy array (world → camera translation)
    """
    cam_intrinsics = read_cameras_text_pinhole(os.path.join(colmap_path, "cameras.txt"))
    images = read_images_text(os.path.join(colmap_path, "images.txt"))

    cameras = {}
    for img_name, (quat, trans, cam_id) in images.items():
        fx, fy, cx, cy, orig_width, orig_height = cam_intrinsics[cam_id]
        R = quaternion_to_rotation_matrix(quat)

        # Apply scaling if target resolution is provided
        if target_resolution is not None:
            target_w, target_h = target_resolution
            sx = target_w / orig_width
            sy = target_h / orig_height
            fx_scaled = fx * sx
            fy_scaled = fy * sy
            cx_scaled = cx * sx
            cy_scaled = cy * sy
            width, height = target_w, target_h
        else:
            fx_scaled, fy_scaled = fx, fy
            cx_scaled, cy_scaled = cx, cy
            width, height = orig_width, orig_height

        cameras[img_name] = {
            'image_name': img_name,
            'width': width,
            'height': height,
            'fx': fx_scaled,
            'fy': fy_scaled,
            'cx': cx_scaled,
            'cy': cy_scaled,
            'rotation': R,
            'translation': trans
        }

    # Add an extra camera with identity pose (if requested)
    if add_identity_camera and cameras:
        # Use the intrinsics of the first camera as a template
        first_cam = next(iter(cameras.values()))
        identity_cam = {
            'image_name': 'identity_view',
            'width': first_cam['width'],
            'height': first_cam['height'],
            'fx': first_cam['fx'],
            'fy': first_cam['fy'],
            'cx': first_cam['cx'],
            'cy': first_cam['cy'],
            'rotation': np.eye(3),               # identity rotation
            'translation': np.zeros(3)           # zero translation
        }
        cameras['identity_view'] = identity_cam

    return cameras

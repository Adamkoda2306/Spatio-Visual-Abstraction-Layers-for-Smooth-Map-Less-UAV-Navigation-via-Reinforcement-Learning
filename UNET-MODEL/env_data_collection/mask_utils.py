"""
Shared helpers: AirSim image capture and segmentation-color -> binary mask
conversion, used by both calibrate_mask_color.py and collect_data.py.
"""

import json
import os

import cv2
import numpy as np
import airsim

import config_collection as cfg


def _get_image_bgr(client, camera_name, image_type):
    """Capture one uncompressed image using the same API as the earlier
    working navigation environment: ``simGetImages([ImageRequest(...)])``.

    AirSim returns uncompressed Scene and Segmentation frames as RGB bytes.
    The collector uses OpenCV, so convert them to BGR before returning.
    """
    responses = client.simGetImages([
        airsim.ImageRequest(str(camera_name), image_type, False, False),
    ])
    if not responses:
        return None

    response = responses[0]
    if response.width <= 0 or response.height <= 0 or not response.image_data_uint8:
        return None

    rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    expected_size = response.height * response.width * 3
    if rgb.size != expected_size:
        raise RuntimeError(
            f"Unexpected image buffer size {rgb.size}; expected {expected_size} "
            f"for a {response.width}x{response.height} RGB image."
        )
    rgb = rgb.reshape(response.height, response.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_scene_bgr(client, vehicle_name=cfg.VEHICLE_NAME, camera_name=cfg.CAMERA_NAME):
    return _get_image_bgr(client, camera_name, airsim.ImageType.Scene)


def get_segmentation_bgr(client, vehicle_name=cfg.VEHICLE_NAME, camera_name=cfg.CAMERA_NAME):
    return _get_image_bgr(client, camera_name, airsim.ImageType.Segmentation)


def dominant_colors(bgr_img, top_k=6):
    """Return [(bgr_tuple, pixel_count), ...] sorted by frequency, descending."""
    pixels = bgr_img.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    order = np.argsort(-counts)
    result = [(tuple(int(c) for c in colors[i]), int(counts[i])) for i in order[:top_k]]
    return result


def save_calibration(obstacle_bgr, background_bgr, path=cfg.CALIBRATION_PATH):
    with open(path, "w") as f:
        json.dump(
            {
                "obstacle_bgr": list(int(c) for c in obstacle_bgr),
                "background_bgr": list(int(c) for c in background_bgr),
            },
            f,
            indent=2,
        )


def load_calibration(path=cfg.CALIBRATION_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No calibration file at {path}. Run calibrate_mask_color.py first "
            "so mask_utils knows which segmentation color means 'obstacle'."
        )
    with open(path) as f:
        data = json.load(f)
    return np.array(data["obstacle_bgr"], dtype=np.int16), np.array(data["background_bgr"], dtype=np.int16)


def seg_to_mask(seg_bgr, obstacle_bgr, tolerance=15):
    """Binary obstacle mask: 255 where the pixel color matches obstacle_bgr
    within `tolerance` per channel, 0 elsewhere."""
    diff = np.abs(seg_bgr.astype(np.int16) - obstacle_bgr.reshape(1, 1, 3))
    match = np.all(diff <= tolerance, axis=-1)
    mask = np.where(match, 255, 0).astype(np.uint8)
    return mask

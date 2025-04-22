
import cv2
import numpy as np
import os
import uuid

from typing import Tuple, List, Any, Dict


import logging, coloredlogs

from cv2 import Mat
from numpy import ndarray

# Create a logger object.
log = logging.getLogger("LiLoc")
coloredlogs.install(logger=log, level=logging.DEBUG)


def read_images(in_imgs: list, max_image_size: int = 0) -> tuple[list[str | Any], dict[str | Any, ndarray | Any]]:
    """
    Reads a list of images and returns a dictionary with unique identifiers as keys and image data as values.

    This function processes a list of image inputs, which can either be image objects or file paths. For image objects,
    it assigns a new UUID as the key. For file paths, it uses the file name (without extension) as the key. If an image
    with the same file name already exists in the dictionary, it skips that image to avoid duplicates. The function
    logs debug and warning messages during the process.

    Args:
        in_imgs (list): A list of image objects or file paths.
        max_image_size (int): If positive, resizes the images while keeping the aspect ratio

    Returns:
        (list, dict): A list with unique identifiers (UUIDs or file names) and a dictionary with the keys as indices and image data as values.
    """
    imgs = {}
    keys = []
    for in_img in in_imgs:
        try:
            if isinstance(in_img, cv2.Mat):
                new_id = str(uuid.uuid4())
                log.debug("Assigning new id %s to image", new_id)
                imgs[new_id] = resize_image(in_img, max_image_size)
                keys.append(new_id)
            else:
                file_id = in_img.split(os.sep)[-1].split(".")[0]
                log.debug("Reading image %s as id %s", in_img, file_id)
                if file_id in imgs:
                    log.warning("Image with the name %s already present in data base. Skipping...", file_id)
                    continue
                im = cv2.imdecode(np.fromfile(str(in_img), np.uint8), cv2.IMREAD_UNCHANGED)
                if max_image_size > 0:
                    imgs[file_id] = resize_image(im, max_image_size)
                else:
                    imgs[file_id] = im
                keys.append(file_id)
        except Exception as e:
            log.warning("Could not read image %s: %s", in_img, str(e))

    return keys, imgs


def resize_image(image, max_size):
    """
    Resizes an OpenCV image to a maximum size while keeping the aspect ratio.

    Args:
        image (cv2.Mat): The input image to be resized.
        max_size (int): The maximum size (width or height) of the resized image.

    Returns:
        cv2.Mat: The resized image with the aspect ratio preserved.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]

    if height <= max_size and width <= max_size:
        return image

    # Calculate the scaling factor
    if height > width:
        scaling_factor = max_size / float(height)
    else:
        scaling_factor = max_size / float(width)

    # Calculate the new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image



def transform_image(img: np.ndarray, matrix: np.ndarray):
    cv2.warpPerspective(img, matrix)


def transform_point(points: np.ndarray, matrix: np.ndarray, input_size=None, target_size=None):
    if points.ndim < 2:
        points = points.reshape(-1, 1, 2)

    if input_size:
        points /= input_size[1::-1]

    out_pt = cv2.perspectiveTransform(points, matrix)
    if target_size:
        out_pt /= target_size[1::-1]  # Normalize points
    if out_pt.shape[0] == 1:
        out_pt = out_pt.reshape(1, 2)
    return out_pt

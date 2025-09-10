import cv2
import numpy as np
import os
import uuid

from typing import Tuple, List, Any, Dict

from tqdm import tqdm

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
                im = cv2.imdecode(np.fromfile(str(in_img), np.uint8), cv2.IMREAD_COLOR)
                if max_image_size > 0:
                    imgs[file_id] = resize_image(im, max_image_size)
                else:
                    imgs[file_id] = im
                keys.append(file_id)
        except Exception as e:
            log.warning("Could not read image %s: %s", in_img, str(e))

    return keys, imgs


def pano_as_cube_map(in_pano: str, max_image_size: int = 0) -> tuple[list[str | Any], dict[str | Any, ndarray | Any]]:
    pass


def cylindrical_to_cube_map(cylindrical_image_path, output_size=(1024, 1024), outpath=""):
    """
    Transform a cylindrical equidistant projection image into six cube-mapped images.

    Parameters:
    - cylindrical_image_path: Path to the cylindrical equidistant projection image.
    - output_size: Size of the output cube map images (default is 512x512).

    Returns:
    - A list of six numpy arrays representing the cube map faces (right, left, top, bottom, front, back).
    """

    # Load the cylindrical image using OpenCV
    cylindrical_image = cv2.imread(cylindrical_image_path)
    if cylindrical_image is None:
        raise ValueError("Could not load the image from the specified path.")

    # Get the dimensions of the cylindrical image
    height, width = cylindrical_image.shape[:2]

    # Create six blank images for the cube map faces
    right = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    left = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    top = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    bottom = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    front = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    back = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Define the output size
    output_width, output_height = output_size

    c_e_up = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    c_e_down = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    c_e_front = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    c_e_back = np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    c_e_right = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    c_e_left = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    c_mat = [c_e_right, c_e_left, c_e_up, c_e_down, c_e_front, c_e_back, ]
    c_img = [right, left, top, bottom, front, back, ]
    c_name = ["00_right.png", "01_left.png", "02_top.png", "03_bottom.png", "04_front.png", "05_back.png"]

    # Map the cylindrical image to the cube map faces
    for mat, img, name in zip(c_mat, c_img, c_name):
        #mat_inv = np.linalg.inv(mat)
        for y in tqdm(range(0, output_height)):
            for x in range(0, output_width):
                # Normalize the coordinates to the range [-1, 1]
                nx = (x / output_width) * 2 - 1
                ny = (y / output_height) * 2 - 1

                # Convert to Euclidean coordinates
                # nx = np.cos(theta) * np.sin(phi)
                # ny = np.sin(theta) * np.sin(phi)
                # nz = np.cos(theta)

                c = np.array((nx, ny, 1, 0))

                v = c @ mat
                v /= np.linalg.norm(v)
                vx = v[0]
                vy = v[1]
                vz = v[2]

                theta = (-np.arctan2(vy, vx)) % (2 * np.pi)
                phi = np.arccos(vz) % np.pi

                # if -1 < theta < 1 and -1 < phi < 1:
                cx_p = theta / (2 * np.pi) * width
                cy_p = phi / np.pi * height
                cx_u = int(np.ceil(cx_p))
                cy_u = int(np.ceil(cy_p))
                cx_l = int(np.floor(cx_p))
                cy_l = int(np.floor(cy_p))
                t = cy_p % 1
                s = cx_p % 1
                p_ll = cylindrical_image[cy_l % height, cx_l % width].astype(np.float32)
                p_ul = cylindrical_image[cy_u % height, cx_l % width].astype(np.float32)
                p_lu = cylindrical_image[cy_l % height, cx_u % width].astype(np.float32)
                p_uu = cylindrical_image[cy_u % height, cx_u % width].astype(np.float32)
                p = ((1 - t) * (1 - s) * p_ll +
                     t * (1 - s) * p_ul +
                     (1 - t) * s * p_lu +
                     t * s * p_uu)
                img[y, x] = np.rint(p).astype(np.uint8)
        cv2.imwrite(outpath + name, img)

    return [right, left, top, bottom, front, back]


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


if __name__ == "__main__":
    cylindrical_to_cube_map(r"C:\dev\liloc\data\StrassenNRW_Siegen\img\1.jpg", output_size=(2048,2048), outpath=r"C:\dev\liloc\data\StrassenNRW_Siegen\img\1")
    cylindrical_to_cube_map(r"C:\dev\liloc\data\StrassenNRW_Siegen\img\2.jpg", output_size=(2048,2048), outpath=r"C:\dev\liloc\data\StrassenNRW_Siegen\img\2")
    cylindrical_to_cube_map(r"C:\dev\liloc\data\StrassenNRW_Siegen\img\3.jpg", output_size=(2048,2048), outpath=r"C:\dev\liloc\data\StrassenNRW_Siegen\img\3")
    cylindrical_to_cube_map(r"C:\dev\liloc\data\StrassenNRW_Siegen\img\4.jpg", output_size=(2048,2048), outpath=r"C:\dev\liloc\data\StrassenNRW_Siegen\img\4")
    cylindrical_to_cube_map(r"C:\dev\liloc\data\StrassenNRW_Siegen\img\5.jpg", output_size=(2048,2048), outpath=r"C:\dev\liloc\data\StrassenNRW_Siegen\img\5")
    cylindrical_to_cube_map(r"C:\dev\liloc\data\StrassenNRW_Siegen\img\6.jpg", output_size=(2048,2048), outpath=r"C:\dev\liloc\data\StrassenNRW_Siegen\img\6")

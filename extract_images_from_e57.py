import os
from gettext import translation

import cv2
import pye57
import json
import numpy as np
import cv2 as cv
import pye57.e57
import tqdm


import pathlib

from helpers import NumpyArrayEncoder


def quaternion_to_euler(x, y, z, w):
    """
    Author: ChatGPT

    Convert a quaternion into yaw, pitch, and roll using NumPy.
    
    Args:
    x, y, z, w: float - The components of the quaternion.
    
    Returns:
    tuple: A tuple containing the yaw, pitch, and roll in radians.
    """
    # Yaw (z-axis rotation)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    
    # Pitch (y-axis rotation)
    sin_pitch = 2.0 * (w * y - z * x)
    if np.abs(sin_pitch) >= 1:
        pitch = np.copysign(np.pi / 2, sin_pitch)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sin_pitch)
    
    # Roll (x-axis rotation)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)
    
    return yaw, pitch, roll


def rotation_matrix_to_quaternion(matrix):
    """
    Author: ChatGPT

    Convert a 4x4 rotation matrix to a quaternion.

    Parameters:
    matrix (numpy.ndarray): A 4x4 rotation matrix.

    Returns:
    numpy.ndarray: A quaternion [w, x, y, z].
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix")

    m = matrix[:3, :3]  # Extract the 3x3 rotation part of the matrix

    trace = np.trace(m)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])

# ==============
# e57 Extraction
# ==============
def extract_pose(pc: pye57.E57):
    header = pc.get_header(0)

    print("Point cloud:", pc.path.split("/")[-1])
    print("    Count:", header.point_count)
    print("    PC Rotation quaternion:", header.rotation)
    print("    PC Translation vector:", header.translation)
    scanner_tr = header.translation
    return header.translation, header.rotation


def extract_images(pc: pye57.E57):
    images = []
    parameters = []

    imf: pye57.libe57.ImageFile = pc.image_file
    imageNodes = imf.root()["images2D"]
    imagesFound = 0
    for imNode in imageNodes:
        camera = imNode["pinholeRepresentation"]
        i = camera["jpegImage"]

        image = read_image(i)
        images.append(image)
        imagesFound += 1

        parameters.append(read_image_parameters(imNode))
        
    print(f"Found {imagesFound} images")
    return images, parameters


def read_image(image_node):
    jpeg_image_data = np.zeros(shape=image_node.byteCount(), dtype=np.uint8)
    image_node.read(jpeg_image_data, 0, image_node.byteCount())
    image = cv.imdecode(jpeg_image_data, cv.IMREAD_COLOR)
    return image


def read_image_parameters(image_node):
    pose_node = image_node["pose"]
    pose = {
        "rw": pose_node[0][0].value(),
        "rx": pose_node[0][1].value(),
        "ry": pose_node[0][2].value(),
        "rz": pose_node[0][3].value(),
        "tx": pose_node[1][0].value(),
        "ty": pose_node[1][1].value(),
        "tz": pose_node[1][2].value(),
    }
    
    camera_node = image_node["pinholeRepresentation"]
    intrinsics = {
        "focalLength":     camera_node["focalLength"].value(),
        "imageHeight":     camera_node["imageHeight"].value(),
        "imageWidth":      camera_node["imageWidth"].value(),
        "pixelHeight":     camera_node["pixelHeight"].value(),
        "pixelWidth":      camera_node["pixelWidth"].value(),
        "principalPointX": camera_node["principalPointX"].value(),
        "principalPointY": camera_node["principalPointY"].value()
    }
    intrinsics["sensorWidth"] = intrinsics["pixelWidth"] * intrinsics["imageWidth"]
    intrinsics["sensorHeight"] = intrinsics["pixelHeight"] * intrinsics["imageHeight"]
    intrinsics["focalLengthPixelsX"] = intrinsics["focalLength"] / intrinsics["pixelWidth"]
    intrinsics["focalLengthPixelsY"] = intrinsics["focalLength"] / intrinsics["pixelHeight"]
    parameters = {
                "pose": pose,
                "intrinsics": intrinsics
            }
    return parameters


# =======
# I/O
# =======

def write_camera_poses(poses, fname):
    with open(fname, "w") as f:
        for location in poses.values():
            for im in location["images"]:
                pose = im["pose"]
                yaw, pitch, roll = quaternion_to_euler(pose['rx'], pose['ry'], pose['rz'], pose['rw'])
                f.write(f"{pose['tx']} {pose['ty']} {pose['tz']} {yaw} {pitch} {roll}\n")


def extract_e57_pose(file, *args, write_imgs=False, render_img_from_rgb=False):
    pose_dict = {}
    image_dict = {}

    filelist = [file]

    filelist.extend(args)

    for path in filelist:
        if path is not pye57.E57:
            p = pathlib.Path(path)
            name = p.stem
            print("Reading " + name)
            pc_e57 = pye57.E57(str(path))
        else:
            print("Reading Point Cloud...")
            pc_e57: pye57.E57 = path
            name = path.split(pc_e57.path)[-1]
            name = name.split(".")[0]

        pose_tr, pose_rot = extract_pose(pc_e57)

        if render_img_from_rgb:
            images, image_parameters = render_from_rgb(pc_e57)
        else:
            images, image_parameters = extract_images(pc_e57)

        if len(images) == 6:
            image_parameters[0]["facing"] = "00_front"
            image_parameters[1]["facing"] = "01_right"
            image_parameters[2]["facing"] = "02_back"
            image_parameters[3]["facing"] = "03_left"
            image_parameters[4]["facing"] = "04_up"
            image_parameters[5]["facing"] = "05_down"

        pose_dict[name] = {
            "pose":
            {
                "translation": pose_tr,
                "rotation": pose_rot
            },
            "images": image_parameters
        }

        # print(image_parameters)

        for i, (image, image_p) in enumerate(zip(images, image_parameters)):
            fname = f"{name}_{image_p.get('facing', i)}.jpg"
            if write_imgs:
                cv.imwrite(f"images/{fname}", image)
            image_dict[fname] = image
            image_p["file"] = fname


    with open("scanner_poses.json", "w") as f:
        json.dump(pose_dict, f, cls=NumpyArrayEncoder, indent=2)

    return pose_dict, image_dict


def render_from_rgb(pc: pye57.E57):
    head = pc.get_header(0)
    t = pc.scan_position(0)[0]
    points = pc.read_scan(0, colors=True, transform=False)
    print(points.keys())
    print(head.translation)
    print(head.rotation_matrix)
    pts = np.vstack([points["cartesianX"], points["cartesianY"], points["cartesianZ"]])
    col = np.vstack([points["colorRed"], points["colorGreen"], points["colorBlue"]])
    print(pts[...,0])
    print(col[...,0])
    print(pts.shape)

    imsize = 2048

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
        [0, 0,  0, 1]
    ], dtype=np.float64)

    c_e_front = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ], dtype=np.float64)

    c_e_back = np.array([
        [-1,  0, 0, 0],
        [ 0,  0,-1, 0],
        [ 0, -1, 0, 0],
        [ 0,  0, 0, 1]
    ], dtype=np.float64)

    c_e_right = np.array([
        [ 0,-1, 0, 0],
        [ 0, 0,-1, 0],
        [ 1, 0, 0, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float64)

    c_e_left = np.array([
        [ 0, 1,  0, 0],
        [ 0, 0, -1, 0],
        [-1, 0,  0, 0],
        [ 0, 0,  0, 1]
    ], dtype=np.float64)

    c_i = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0,   0,   1, 0]
    ], dtype=np.float64)
    images = [
        render_from_pov(pts, c_e_front, c_i, col, imsize),
        render_from_pov(pts, c_e_right, c_i, col, imsize),
        render_from_pov(pts, c_e_back, c_i, col, imsize),
        render_from_pov(pts, c_e_left, c_i, col, imsize),
        render_from_pov(pts, c_e_up, c_i, col, imsize),
        render_from_pov(pts, c_e_down, c_i, col, imsize),
    ]

    q_front = rotation_matrix_to_quaternion(c_e_front)
    q_left  = rotation_matrix_to_quaternion(c_e_left )
    q_back  = rotation_matrix_to_quaternion(c_e_back )
    q_right = rotation_matrix_to_quaternion(c_e_right)
    q_up    = rotation_matrix_to_quaternion(c_e_up   )
    q_down  = rotation_matrix_to_quaternion(c_e_down )

    image_params = [
        {"pose": {"tx": t[0] , "ty": t[1], "tz": t[2], "rw": q_front[0], "rx": q_front[1], "ry": q_front[2], "rz": q_front[3]}},
        {"pose": {"tx": t[0] , "ty": t[1], "tz": t[2], "rw": q_right[0], "rx": q_right[1], "ry": q_right[2], "rz": q_right[3]}},
        {"pose": {"tx": t[0] , "ty": t[1], "tz": t[2], "rw": q_back [0], "rx": q_back [1], "ry": q_back [2], "rz": q_back [3]}},
        {"pose": {"tx": t[0] , "ty": t[1], "tz": t[2], "rw": q_left [0], "rx": q_left [1], "ry": q_left [2], "rz": q_left [3]}},
        {"pose": {"tx": t[0] , "ty": t[1], "tz": t[2], "rw": q_up   [0], "rx": q_up   [1], "ry": q_up   [2], "rz": q_up   [3]}},
        {"pose": {"tx": t[0] , "ty": t[1], "tz": t[2], "rw": q_down [0], "rx": q_down [1], "ry": q_down [2], "rz": q_down [3]}}
    ]

    return images, image_params


def render_from_pov(pts, c_e, c_i, col, imsize):
    pts = np.vstack((pts, np.ones((pts.shape[1]))))
    pc_cam = (c_i @ (c_e @ pts))
    pc_cam[:2,...] = pc_cam[:2,...] / pc_cam[2,...]
    #pc_cam = np.round(pc_cam, 0).astype(np.int32)
    pc_and_col = np.vstack([pc_cam, col])
    img = np.zeros((imsize, imsize, 4), dtype=np.uint8)
    zBuf = np.full((imsize, imsize), np.inf, dtype=np.float64)
    for p in tqdm.tqdm(np.nditer(pc_and_col, flags=["external_loop"], order="F"), total=pts.shape[1]):
        if p[1] < -1 or p[1] > 1 or p[0] < -1 or p[0] > 1 or p[2] < 0:
            continue

        px = int(np.round(p[1] * imsize / 2 + imsize / 2, 0))
        py = int(np.round(p[0] * imsize / 2 + imsize / 2, 0))

        if px < 0 or px >= imsize or py < 0 or py >= imsize or p[2] < 0 or p[2] > zBuf[px, py]:
            continue

        img [px, py, 0] = p[5]
        img [px, py, 1] = p[4]
        img [px, py, 2] = p[3]
        img [px, py, 3] = 255
        zBuf[px, py]    = p[2]

    # Dilate alpha using Blackhat to only get a mask of closed pixels
    alpha = np.array(cv2.morphologyEx(img[...,3], cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))), dtype=np.uint8)
    # Inpaint where possible
    img = cv2.inpaint(img[...,:3], alpha, 5, cv.INPAINT_TELEA)

    return img.astype(np.uint8)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LiLoc Point Cloud Image Extractor Tool")
    parser.add_argument("point_cloud", metavar="pc", type=pathlib.Path)
    parser.add_argument("-o", "--output-dir", type=pathlib.Path, help="Path to output directory", default=None)
    parser.add_argument("-r", "--rgb", action='store_true')

    args = parser.parse_args()

    path: pathlib.Path = args.point_cloud
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.e57"))

    outpath: pathlib.Path = args.output_dir
    if outpath is None:
        outpath = path if path.is_dir() else path.parent

    if len(files) == 0:
        print("No files found in " + path)

    poses, images = extract_e57_pose(*files, render_img_from_rgb=args.rgb)
    for imgname, img in images.items():
        cv2.imwrite(str(outpath.joinpath(imgname)), img)
    write_camera_poses(poses, str(outpath.joinpath("camera_poses.txt")))

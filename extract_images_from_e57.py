import pye57
import json
import numpy as np
import cv2 as cv
import pye57.e57

from helpers import NumpyArrayEncoder

files = [
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 003.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 004.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 005.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 006.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 007.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 008.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 009.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 010.e57",
    r"/home/patrick/dev/BIMKIT/straelen_2024-01-07/Job 009- HiWi Raum 011.e57",
]

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

        image = readImage(i)
        images.append(image)
        imagesFound += 1

        parameters.append(readImageParameters(imNode))
        
    print(f"Found {imagesFound} images")
    return images, parameters

def readImage(imageNode):
    jpeg_image_data = np.zeros(shape=imageNode.byteCount(), dtype=np.uint8)
    imageNode.read(jpeg_image_data, 0, imageNode.byteCount())
    image = cv.imdecode(jpeg_image_data, cv.IMREAD_COLOR)
    return image

def readImageParameters(imageNode):
    poseNode = imageNode["pose"]
    pose = {
        "rw": poseNode[0][0].value(),
        "rx": poseNode[0][1].value(),
        "ry": poseNode[0][2].value(),
        "rz": poseNode[0][3].value(),
        "tx": poseNode[1][0].value(),
        "ty": poseNode[1][1].value(),
        "tz": poseNode[1][2].value(),
    }
    
    cameraNode = imageNode["pinholeRepresentation"]
    intrinsics = {
        "focalLength":     cameraNode["focalLength"].value(),
        "imageHeight":     cameraNode["imageHeight"].value(),
        "imageWidth":      cameraNode["imageWidth"].value(),
        "pixelHeight":     cameraNode["pixelHeight"].value(),
        "pixelWidth":      cameraNode["pixelWidth"].value(),
        "principalPointX": cameraNode["principalPointX"].value(),
        "principalPointY": cameraNode["principalPointY"].value()
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

def write_camera_poses(poses):
    with open("camera_poses.txt", "w") as f:
        for location in poses.values():
            for im in location["images"]:
                pose = im["pose"]
                yaw, pitch, roll = quaternion_to_euler(pose['rx'], pose['ry'], pose['rz'], pose['rw'])
                f.write(f"{pose['tx']} {pose['ty']} {pose['tz']} {yaw} {pitch} {roll}\n")


pose_dict = {}

for path in files:
    name = path.split("/")[-1]
    name = name.split(".")[0]
    pc_e57 = pye57.E57(path)
    pose_tr, pose_rot = extract_pose(pc_e57)
    images, image_parameters = extract_images(pc_e57)
    image_dict = {}

    if(len(images) == 6):
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
        cv.imwrite(f"images/{fname}", image)
        image_p["file"] = fname

with open("scanner_poses.json", "w") as f:
    json.dump(pose_dict, f, cls=NumpyArrayEncoder, indent=2)

write_camera_poses(pose_dict)


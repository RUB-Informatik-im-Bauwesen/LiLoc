import image_tools
from feature_matching import CrossMatching
from feature_matchers.xfeat_matcher import XFeatMatcher
import panoramic_point_to_scan
import cv2
import numpy as np
import glob
import json

from panoramic_point_to_scan import scanner_viz_scale

# point_cloud_filename = "/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360_Abpl m fl Bw an Stütze_Panorama/Punktwolken/Abplatzung+Bewehrung_2m_LR.e57"

# pan_img = cv2.imread("Abplatzung+Bewehrung_2m_MR.jpg")
# in_img = cv2.imread("IMG_6355(1).JPEG")
# in_img_overlay_path = "IMG_6355(1).JPEG"

"""Author: ChatGPT"""
def pose_dict_to_matrix(pose_dict):
    # Extract translation and quaternion
    t = pose_dict['translation']
    q = pose_dict['rotation']
    w, x, y, z = q

    # Normalize quaternion
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

    # Compose transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

"""Author: ChatGPT"""
def dict_to_camera_matrix(cam_dict):
    # --- Intrinsic matrix ---
    intr = cam_dict['intrinsics']
    fx = intr['focalLengthPixelsX']
    fy = intr['focalLengthPixelsY']
    cx = intr['principalPointX']
    cy = intr['principalPointY']
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ])

    # --- Extrinsic matrix ---
    pose = cam_dict['pose']
    rw, rx, ry, rz = pose['rw'], pose['rx'], pose['ry'], pose['rz']
    tx, ty, tz = pose['tx'], pose['ty'], pose['tz']

    # Normalize quaternion
    norm = np.sqrt(rw**2 + rx**2 + ry**2 + rz**2)
    rw, rx, ry, rz = rw/norm, rx/norm, ry/norm, rz/norm

    # Quaternion to rotation matrix
    R = np.array([
        [1 - 2*(ry**2 + rz**2),     2*(rx*ry - rz*rw),     2*(rx*rz + ry*rw)],
        [    2*(rx*ry + rz*rw), 1 - 2*(rx**2 + rz**2),     2*(ry*rz - rx*rw)],
        [    2*(rx*rz - ry*rw),     2*(ry*rz + rx*rw), 1 - 2*(rx**2 + ry**2)]
    ])
    t = np.array([[tx], [ty], [tz]])

    # [R|t] (3x4)
    #Rt = np.hstack((R, t))

    # Camera matrix
    P = K @ R
    return K, R

test_file_path = "C:/sciebo/BRIX/07 - Data/Straelen/"

point_cloud_filename = test_file_path + "scan/Straelen_2024-01-07 010_5cm.e57"
model_file_name = test_file_path + "model/straelen.stl"
model_transform = np.linalg.inv(np.array([
        [0.309558212757, -0.950880110264, 0.000880334934, 8.538056373596],
        [0.950873732567, 0.309559375048, 0.003477090737, -52.835781097412],
        [-0.003578812117, -0.000239274566, 0.999993562698, 56.598762512207],
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]]
    ))

pan_img = cv2.imread(test_file_path + "straelen_pano/Job 009- Bruecke Straelen 010.jpg")
in_img_path = test_file_path + "straelen_img/PXL_20240107_110413700.jpg"
in_img = cv2.imread(in_img_path)
in_img_overlay_path = test_file_path + "img_raw/PXL_20240107_110925984.jpg"

with open(test_file_path + "pano/scanner_poses.json") as f:
    camera_poses = json.load(f)

pts = np.float32([[[0.5, 0.5]]])


matcher = CrossMatching(glob.glob(test_file_path + "pano/*.jpg"), [in_img_path], matcher=XFeatMatcher())
matcher.find_features()
matcher.find_matches()
print(matcher.matches)
if len(matcher.matches) > 0:
    results = matcher.matches[0]
    print("Visualizing " + str(results))
    mat = results["matrix"]
    img_a_id = results["image_a"]
    img_b_id = results["image_b"]
    img_a = matcher.img_set_a[img_a_id]
    img_b = matcher.img_set_b[img_b_id]

    a_to_b = cv2.warpPerspective(img_b, np.linalg.inv(mat), (img_a.shape[1], img_a.shape[0]))
    overlay = cv2.addWeighted(img_a, 0.5, a_to_b, 0.5, 0)

    cv2.imshow(img_b_id, overlay)
    cv2.waitKey(1000)
    # print(pts)
    pan_img_points = np.array([[0.5 * in_img.shape[0],0.5*in_img.shape[1],1]])

    scan_img = results["image_a"]
    point_cloud_name = list(filter(lambda v: scan_img in list(map(lambda l: l["file"].split(".")[0],camera_poses[v]["images"])), camera_poses.keys()))[0]
    point_cloud_filename = test_file_path + "/scan/" + point_cloud_name + ".e57"
    point_cloud_data = camera_poses[point_cloud_name]
    scan_img_mat_int, scan_img_mat_ext = dict_to_camera_matrix(list(filter(lambda l: l["file"].startswith(scan_img), point_cloud_data["images"]))[0])

    print("Mat: " + str(mat))
    r = mat.T @ pan_img_points.T
    r = np.array((1024,1024,1))
    print("Scan img point: " + str(r))
    print("Scan mat: " + str(scan_img_mat_int) + "\n" + str(scan_img_mat_ext))
    s = np.linalg.inv(scan_img_mat_int) @ r
    s = np.linalg.inv(scan_img_mat_ext) @ s  # This is still wrong somehow...
    s = s[:3].reshape(3)
    s /= np.linalg.norm(s)
    rays = [ s ]

    print("Transformed points:", rays)

    # print("relative point:", pan_img_point, pan_img.shape)
    panoramic_point_to_scan.find_rays(
        rays,
        point_cloud_filename, 
        show=True, 
        images=[in_img_overlay_path], 
        reference_model_path=model_file_name,
        reference_model_transform=pose_dict_to_matrix(point_cloud_data["pose"])
    )

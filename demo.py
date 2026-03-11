import image_tools
from extract_images_from_e57 import quaternion_to_euler
from feature_matching import CrossMatching
from feature_matchers.xfeat_matcher import XFeatMatcher
import panoramic_point_to_scan
import cv2
import numpy as np
import glob
import json

from image_tools import read_images
from panoramic_point_to_scan import scanner_viz_scale

# point_cloud_filename = "/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360_Abpl m fl Bw an Stütze_Panorama/Punktwolken/Abplatzung+Bewehrung_2m_LR.e57"

# pan_img = cv2.imread("Abplatzung+Bewehrung_2m_MR.jpg")
# in_img = cv2.imread("IMG_6355(1).JPEG")
# in_img_overlay_path = "IMG_6355(1).JPEG"

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

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
    R = quaternion_rotation_matrix([w,x,y,z])

    # Compose transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

"""Author: ChatGPT"""
def dict_to_camera_matrix(cam_dict):
    # --- Intrinsic matrix ---
    intr = cam_dict['intrinsics']
    fx = intr['focalLengthPixelsX']
    fy = intr['focalLengthPixelsY']
    px = intr['imageWidth']
    py = intr['imageHeight']
    cx = intr['principalPointX']
    cy = intr['principalPointY']
    K = np.array([
        [fx/px*2, 0,   0],
        [0,  fy/py*2,  0],
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
    R = quaternion_rotation_matrix([rw,rx,ry,rz])

    t = np.array([tx, ty, tz])

    # [R|t] (3x4)
    #Rt = np.hstack((R, t))

    # Camera matrix
    P = K @ R
    return K, R, t

from helpers import DataSource
def visualize(test_file_path, point_cloud_filename):

    point_cloud_filename = test_file_path + "/pc/RUB_Parkhaus_2025-03-24_gesamt_10cm.e57"
    model_file_name = test_file_path + "model/parkhaus_ransac.obj"
    #model_transform = np.linalg.inv(np.array([
    #        [0.309558212757, -0.950880110264, 0.000880334934, 8.538056373596],
    #        [0.950873732567, 0.309559375048, 0.003477090737, -52.835781097412],
    #        [-0.003578812117, -0.000239274566, 0.999993562698, 56.598762512207],
    #        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]]
    #    ))

    ds = DataSource(test_file_path + "out/matches_pano_dcim/matches.json")

    in_img_path = test_file_path + "DCIM/DSC_0036.JPG"
    #in_img_overlay_path = test_file_path + "img_raw/PXL_20240107_110925984.jpg"

    with open(test_file_path + "pano/scanner_poses.json") as f:
        camera_poses = json.load(f)

    pts = np.float32([[[0.5, 0.5]]])

    #matcher = CrossMatching(glob.glob(test_file_path + "pano/*.jpg"), glob.glob(test_file_path + "DCIM/*.jpg"), matcher=XFeatMatcher())
    #matcher.find_features()
    #matcher.find_matches()
    origins = []
    rays = []
    images = []
    for results in ds["matches"][:2]:
        print("Casting " + str(results))
        img_a_id = results["image_a"]
        img_b_id = results["image_b"]
        img_a = ds.get_image(img_a_id)
        img_b = ds.get_image(img_b_id)
        origin, ray = match_to_geometry(results, img_a, img_b, camera_poses)
        origins.append(origin)
        rays.append(ray)
        images.append(img_b)

    # print("relative point:", pan_img_point, pan_img.shape)
    panoramic_point_to_scan.find_rays(
        origins,
        rays,
        point_cloud_filename,
        show=True,
        images=images,
        #reference_model_path=model_file_name,
        #reference_model_transform=pose_dict_to_matrix(point_cloud_data["pose"])
    )

def match_to_geometry(match, img_a, img_b, camera_poses):
    mat = match["matrix"]

    #a_to_b = cv2.warpPerspective(img_b, np.linalg.inv(mat), (img_a.shape[1], img_a.shape[0]))
    #overlay = cv2.addWeighted(img_a, 0.5, a_to_b, 0.5, 0)

    #cv2.imshow("debug", overlay)
    #cv2.waitKey(1000)
    # print(pts)

    # Transform from a to b
    pan_img_points = np.array([[0.5 * img_b.shape[1],0.5*img_b.shape[0],1]])
    r = np.linalg.inv(mat) @ pan_img_points.T
    r = r / r[2]
    r_c = (r.T / np.array((img_a.shape[1]*2, img_a.shape[0]*2, 1)) - np.array((0.5,0.5,0))).T  # to cam coords
    r_c = np.array((-r_c[1, ...], r_c[0, ...], -r_c[2, ...]))
    #r[...,2] = -1  # z-Coordinate
    #r[...,1] *= -1  # flip y (because pixels count from the top, but camera doesnt)
    #r_c = r_c.T

    scan_img = match["image_a"]
    point_cloud_name = list(filter(lambda v: scan_img in list(map(lambda l: l["file"].split(".")[0],camera_poses[v]["images"])), camera_poses.keys()))[0]
    #print(f"Looking for point cloud {point_cloud_filename}")
    point_cloud_data = camera_poses[point_cloud_name]
    scan_img_mat_int, scan_img_mat_ext, origin = dict_to_camera_matrix(list(filter(lambda l: l["file"].startswith(scan_img), point_cloud_data["images"]))[0])
    # TODO: is the camera matrix ext transform correct?
    ray = transform_points(r_c, scan_img_mat_ext, scan_img_mat_int)[0]
    return origin, ray



def transform_points(points, scan_img_mat_ext, scan_img_mat_int):
    #r = np.array([[0, 0, -1],
    #              [-1, -1, -1], [0, -1, -1], [1, -1, -1],
    #              [-1, 0, -1], [0, 0, -1], [1, 0, -1],
    #              [-1, 1, -1], [0, 1, -1], [1, 1, -1]]).transpose()
    print("Scan img point: " + str(points))
    print("Scan mat: " + str(scan_img_mat_int) + "\n" + str(scan_img_mat_ext))
    s = np.linalg.inv(scan_img_mat_int) @ points
    s = scan_img_mat_ext @ s
    rays = []
    for i in range(s.shape[1]):
        v = s[..., i].reshape(-1)
        v /= np.linalg.norm(v)
        rays.append(v)
    print("Transformed points:", rays)
    return rays



if __name__ == "__main__":
    test_file_path = "C:/dev/liloc/data/rub_parkhaus/"
    point_cloud_filename = test_file_path + "/pc/RUB_Parkhaus_2025-03-24_gesamt_10cm.e57"

    visualize(test_file_path, point_cloud_filename)


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

test_file_path = "./data/"

point_cloud_filename = test_file_path + "scan/Straelen_2024-01-07 010_5cm.e57"
model_file_name = test_file_path + "straelen_pc/straelen.stl"
model_transform = np.linalg.inv(np.array([
        [0.309558212757, -0.950880110264, 0.000880334934, 8.538056373596],
        [0.950873732567, 0.309559375048, 0.003477090737, -52.835781097412],
        [-0.003578812117, -0.000239274566, 0.999993562698, 56.598762512207],
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]]
    ))

pan_img = cv2.imread(test_file_path + "straelen_pano/Job 009- Bruecke Straelen 010.jpg")
in_img = test_file_path + "straelen_img/PXL_20240107_110413700.jpg"
in_img_overlay_path = test_file_path + "img_raw/PXL_20240107_110925984.jpg"

with open(test_file_path + "straelen_pano/scanner_poses.json") as f:
    camera_poses = json.load(f)

pts = np.float32([[[0.5, 0.5]]])


matcher = CrossMatching([in_img], glob.glob(test_file_path + "straelen_pano/*.jpg"), matcher=XFeatMatcher())
matcher.find_features()
matcher.find_matches()
print(matcher.matches)
if len(matcher.matches) > 0:
    results = matcher.matches[0]
    mat = results["matrix"]
    # print(pts)
    pan_img_points = mat @ np.array([[0.5,0.5]])

    scan_img = results["image_b"]
    point_cloud_name = list(filter(lambda v: scan_img in list(map(lambda l: l["file"].split(".")[0],camera_poses[v]["images"])), camera_poses.keys()))[0]
    point_cloud_filename = test_file_path + "/straelen_pc/" + point_cloud_name + ".e57"
    point_cloud_data = camera_poses[point_cloud_name]
    scan_img_mat = point_cloud_data["imgaes"][scan_img + ".jpg"]["matrix"]  # ???

    rays = [scan_img_mat @ pan_img_points]

    print("Transformed points:", pan_img_points)

    # print("relative point:", pan_img_point, pan_img.shape)
    panoramic_point_to_scan.find_rays(
        rays,
        point_cloud_filename, 
        show=True, 
        images=[in_img_overlay_path], 
        reference_model_path=model_file_name,
        reference_model_transform=point_cloud_data["pose"]
    )


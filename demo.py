from feature_matching import CrossMatching
from feature_matchers.xfeat_matcher import XFeatMatcher
import panoramic_point_to_scan
import cv2
import numpy as np
import glob

# point_cloud_filename = "/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360_Abpl m fl Bw an Stütze_Panorama/Punktwolken/Abplatzung+Bewehrung_2m_LR.e57"

# pan_img = cv2.imread("Abplatzung+Bewehrung_2m_MR.jpg")
# in_img = cv2.imread("IMG_6355(1).JPEG")
# in_img_overlay_path = "IMG_6355(1).JPEG"

test_file_path = "C:/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Testdaten/"

point_cloud_filename = test_file_path + "scan/Straelen_2024-01-07 010_5cm.e57"
model_file_name = test_file_path + "model/straelen.stl"
model_transform = np.linalg.inv(np.array([
        [0.309558212757, -0.950880110264, 0.000880334934, 8.538056373596],
        [0.950873732567, 0.309559375048, 0.003477090737, -52.835781097412],
        [-0.003578812117, -0.000239274566, 0.999993562698, 56.598762512207],
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]]
    ))

pan_img = cv2.imread(test_file_path + "pano/Job 009- Bruecke Straelen 010.jpg")
in_img = cv2.imread(test_file_path + "img_raw/PXL_20240107_110925984.jpg")
in_img_overlay_path = test_file_path + "img_raw/PXL_20240107_110925984.jpg"

pts = np.float32([[[0.5, 0.5]]])

pan_rect = np.array((8192, 3416))  # actual rect of the pano img

matcher = CrossMatching(glob.glob(test_file_path + "img_raw/*.jpg"), glob.glob(test_file_path + "pano/*.jpg"), matcher=XFeatMatcher())
matcher.find_features()
matcher.find_matches()
print(matcher.matches)
results = matcher.matches
if results["transformed_points"] is not []:
    mat = results["matrix"]
    # print(pts)
    pan_img_points = results["transformed_points"]

    pan_img_points = [p * pan_img.shape[1::-1] / pan_rect for p in pan_img_points]

    print("Transformed points:", pan_img_points)

    # print("relative point:", pan_img_point, pan_img.shape)
    panoramic_point_to_scan.find_all_points(
        pan_img_points, 
        point_cloud_filename, 
        show=True, 
        images=[in_img_overlay_path], 
        reference_model_path=model_file_name,
        reference_model_transform=model_transform
    )


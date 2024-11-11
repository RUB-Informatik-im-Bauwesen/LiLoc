import pathlib

import cv2
from cv2 import xfeatures2d
import numpy as np
from scipy.signal import convolve2d
# import matplotlib.pyplot as plt
from PIL import Image
import argparse, json
import sys
import glob
from helpers import NumpyArrayEncoder

def detect_features(image: np.ndarray, hessian_threshold=1000, upright=True, octaves=None):
    # print("Detecting keypoints in input image")
    sift = cv2.SIFT.create()
    if octaves:
        sift.setNOctaveLayers(octaves)
    # print("Octaves: ", sift.getNOctaves(), sift.getNOctaveLayers())

    return sift.detectAndCompute(image, None)


def relocalize(in_img: np.ndarray, pan_img: np.ndarray, points_to_transform=None, show=False, scale=1.0):
    results = relocalize_multi([in_img], pan_img, [points_to_transform], show, scale)
    return results[0]


def relocalize_multi(in_imgs: list[np.ndarray], pan_img: np.ndarray, points_to_transform=None, show=False, scale=1.0):
    pan_img_scale = scale

    pan_img_resized = cv2.resize(pan_img, (int(pan_img.shape[1] * pan_img_scale), int(pan_img.shape[0] * pan_img_scale)))

    in_img_scale = scale * 0.5

    in_imgs_resized = [
        cv2.resize(in_img, (
            int(in_img.shape[1] * pan_img.shape[0] / in_img.shape[0] * in_img_scale),
            int(pan_img.shape[0] * in_img_scale)
        )) for in_img in in_imgs
    ]

    # print(f"Detected {len(pts1)} points")
    # keypoint_img = in_img.copy()
    # cv2.drawKeypoints(in_img, pts1, keypoint_img)
    # cv2.imshow("in_img", keypoint_img)

    in_img_pts = [detect_features(in_img, hessian_threshold=1000, octaves=20) for in_img in in_imgs_resized]

    pts, des = detect_features(pan_img_resized, 250, octaves=20)

    # print(f"Detected {len(pts)} points")
    # keypoint_img = pan_img.copy()
    # cv2.drawKeypoints(pan_img, pts, keypoint_img)
    # cv2.imshow("pan_img", keypoint_img)
    # cv2.waitKey()

    results = []

    if not points_to_transform:
        points_to_transform = [[0.5, 0.5]] * len(in_imgs)

    for (pts_in, des_in), in_img, points_to_transform_ in zip(in_img_pts, in_imgs_resized, points_to_transform):
        matches, mask, matrix = match_points(pts_in, des_in, pts, des)
        if matrix is None:
            continue
        n_good_matches = mask.count([1,0])
        
        h, w, _ = in_img.shape
        rect = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        rect_pts = _transform_point(rect, matrix, pan_img_resized.shape)

        center = np.float32([0.5 * w, 0.5 * h]).reshape(-1, 1, 2)
        center_pt = _transform_point(center, matrix, pan_img_resized.shape)

        points_to_transform_ *= np.array([w, h])
        transformed_points = _transform_point(points_to_transform_, matrix, pan_img_resized.shape)

        results.append({"matrix": matrix.tolist(),
                        "center_point": center_pt.tolist(),
                        "rect_points": rect_pts.tolist(),
                        "transformed_points": transformed_points,
                        "matches": n_good_matches
                        })
        
        print(results)
        if show:
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=mask,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img3 = cv2.drawMatchesKnn(in_img, pts_in, pan_img_resized, pts, matches, None, **draw_params)
            pilimg = Image.fromarray(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            pilimg.show("matches")

            img2 = cv2.polylines(pan_img_resized, [np.int32(rect_pts * pan_img_resized.shape[1::-1])], True, 255, 3, cv2.LINE_AA)
            pilimg = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            pilimg.show("polylines")

            cv2.imwrite("./out/01_knn_matches.png", img3)
            cv2.imwrite("./out/02_rect_match.png", img2)
    return results


def _transform_point(in_img_point: np.ndarray, matrix, pan_img_dims):
    if in_img_point.ndim < 2:
        in_img_point = in_img_point.reshape(-1, 1, 2)
    pan_pt = cv2.perspectiveTransform(in_img_point, matrix)
    pan_pt /= pan_img_dims[1::-1]  # Normalize points
    if pan_pt.shape[0] == 1:
        pan_pt = pan_pt.reshape(1, 2)
    return pan_pt


def match_points(pts1, des1, pts2, des2):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=500))
    matches = flann.knnMatch(des1, des2, k=2)

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    
    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            matches_mask[i] = [1, 0]

    MIN_MATCH_COUNT = 30
    if len(good) < MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return matches, None, None

    print(f"Found {len(good)} matches, attempting homography")
    src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # matches_mask = mask.ravel().tolist()

    return matches, matches_mask, M


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs Services")
    parser.add_argument("-o", "--outputPath", type=pathlib.Path, default=".",
                        help="Path to output directory")

    parser.add_argument("panoramic_image", metavar="panIMG", type=str)
    parser.add_argument("input_images", metavar="IMG", type=str, nargs="+")

    parser.add_argument("--show", action='store_true')

    args = parser.parse_args()
    output_path = pathlib.Path(args.outputPath)

    input_images_args = [pathlib.Path(in_img) for in_img in args.input_images]
    panoramic_image = pathlib.Path(args.panoramic_image)

    input_images = []
    # expanding glob chars
    for f in input_images_args:
        if glob.escape(f) != f:
            # -> There are glob pattern chars in the string
            input_images.extend(pathlib.Path(p) for p in glob.glob(str(f)))
        else:
            input_images.append(f)

    if not panoramic_image.exists():
        print(f"Cannot find panoramic image at {panoramic_image}")
        sys.exit(1)
    
    for p in input_images:
        if not p.exists():
            print(f"Cannot find input image at {p}")
            sys.exit(1)

    pan_img = cv2.imread(str(panoramic_image))
    in_imgs = [cv2.imread(str(in_img)) for in_img in input_images]
    # pan_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360_Abpl m fl Bw und Kiesnest Betonwand_Panorama/Abplatzung,Bewehrung,GKS_0,5m_LR.jpg")
    # in_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360_Abpl m fl Bw und Kiesnest Betonwand_Panorama/smartphone-bilder/out.jpg")

    # pan_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360 Aufnahmen 02.03.2023 0,1mm Querriss/0,1er Riss ID05- RTC, MR, 2m.jpg")
    # in_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360 Aufnahmen 02.03.2023 0,1mm Querriss/smartphone-bilder/IMG_9221.jpg")
    result = relocalize_multi(in_imgs, pan_img, show=args.show)

    for r, input_image in zip(result, input_images):
        r["image"] = str(input_image)

    print(result)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "out.json", 'w') as f:
        json.dump(result, f, indent="\t", cls=NumpyArrayEncoder)


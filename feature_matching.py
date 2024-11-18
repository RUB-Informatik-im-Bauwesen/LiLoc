# import matplotlib.pyplot as plt
import argparse

import coloredlogs
import glob
import logging
import pathlib
import sys

import cv2
import numpy as np

from feature_matchers.sift import SIFTMatcher
from feature_matchers.xfeat_matcher import XFeatMatcher
from image_tools import read_images

# Create a logger object.
log = logging.getLogger("FeatureMatching")
coloredlogs.install(logger=log, level=logging.DEBUG)


def save_keypoint_cache(name, keypoints, descriptors, cache_dir):
    descriptor_fn = pathlib.Path(cache_dir, name + "_desc.npy")
    descriptors.tofile(descriptor_fn)

    n_pts = len(keypoints)
    kp_points = np.zeros((n_pts, 2), np.float32)
    kp_sizes = np.zeros(n_pts, np.float32)
    kp_angles = np.zeros(n_pts, np.float32)
    kp_responses = np.zeros(n_pts, np.float32)
    kp_octaves = np.zeros(n_pts, np.int32)
    kp_classes = np.zeros(n_pts, np.int32)

    for i, point in enumerate(keypoints):
        kp_points[i] = point.pt
        kp_sizes[i] = point.size
        kp_angles[i] = point.angle
        kp_responses[i] = point.response
        kp_octaves[i] = point.octave
        kp_classes[i] = point.class_id

    kp_points.tofile(pathlib.Path(cache_dir, name + "_kp_points.npy"))
    kp_sizes.tofile(pathlib.Path(cache_dir, name + "_kp_sizes.npy"))
    kp_angles.tofile(pathlib.Path(cache_dir, name + "_kp_angles.npy"))
    kp_responses.tofile(pathlib.Path(cache_dir, name + "_kp_responses.npy"))
    kp_octaves.tofile(pathlib.Path(cache_dir, name + "_kp_octaves.npy"))
    kp_classes.tofile(pathlib.Path(cache_dir, name + "_kp_classes.npy"))


def load_keypoint_cache(name, cache_dir):
    descriptor_fn = pathlib.Path(cache_dir, name + "_desc.npy")
    if not descriptor_fn.exists():
        return None, None

    kp_points = np.fromfile(pathlib.Path(cache_dir, name + "_kp_points.npy"), dtype=np.float32).reshape(-1, 2)
    kp_sizes = np.fromfile(pathlib.Path(cache_dir, name + "_kp_sizes.npy"), dtype=np.float32)
    kp_angles = np.fromfile(pathlib.Path(cache_dir, name + "_kp_angles.npy"), dtype=np.float32)
    kp_responses = np.fromfile(pathlib.Path(cache_dir, name + "_kp_responses.npy"), dtype=np.float32)
    kp_octaves = np.fromfile(pathlib.Path(cache_dir, name + "_kp_octaves.npy"), dtype=np.int32)
    kp_classes = np.fromfile(pathlib.Path(cache_dir, name + "_kp_classes.npy"), dtype=np.int32)

    keypoints = []
    for i in range(len(kp_points)):
        keypoint = cv2.KeyPoint(
            x=float(kp_points[i][0]),
            y=float(kp_points[i][1]),
            size=float(kp_sizes[i]),
            angle=float(kp_angles[i]),
            response=float(kp_responses[i]),
            octave=int(kp_octaves[i]),
            class_id=int(kp_classes[i])
        )
        keypoints.append(keypoint)

    descriptors = np.fromfile(descriptor_fn, dtype=np.float32).reshape(len(keypoints), -1)

    return keypoints, descriptors



class FeatureMatching:
    def __init__(self, pano_imgs, in_imgs, max_image_size=2048, matcher=None, output_dir=False):
        self.pano_features = {}
        self.in_img_features = {}
        self.match_matrix = None

        self.output_dir = output_dir

        if matcher:
            self.matcher = matcher
        else:
            self.matcher = SIFTMatcher()

        log.info("Reading panoramic images")
        self.pano_img_names, self.pano_imgs = read_images(pano_imgs, max_image_size)

        log.info("Reading input images")
        self.in_img_names, self.in_imgs = read_images(in_imgs, max_image_size)

        log.info("Finished reading images")

    def find_features(self, skip_existing=False):
        log.info("Finding features in panoramic images")
        _pano_features = self._find_features(self.pano_imgs, self.pano_features.keys() if skip_existing else [])

        log.info("Finding features in input images")
        _in_img_features = self._find_features(self.in_imgs, self.in_img_features.keys() if skip_existing else [])

        self.pano_features.update(_pano_features)
        self.in_img_features.update(_in_img_features)

    def _find_features(self, img_db: dict, skip=None):
        if skip is None:
            skip = []

        feat_dict = {}
        for key in sorted(img_db.keys()):
            if key in skip:
                continue
            in_img = img_db[key]
            features = self.matcher.find_features(in_img)
            feat_dict[key] = features
            log.debug("Found %6d features in image %s", len(features[0]), key)
        return feat_dict

    def find_matches(self):
        len_pano = len(self.pano_img_names)
        len_img = len(self.in_img_names)
        self.match_matrix = np.empty((len_pano, len_img), dtype=np.object_)

        log.info("Starting matching...")
        for i_p, pano in enumerate(self.pano_img_names):
            log.info("%4d/%d matches complete", i_p * len_img, len_pano * len_img)
            for i_i, img in enumerate(self.in_img_names):
                pano_pts, pano_des = self.pano_features[pano]
                img_pts, img_des = self.in_img_features[img]
                matches, matches_mask, matrix = self.matcher.match_points(pano_pts, pano_des, img_pts, img_des)
                self.match_matrix[i_p, i_i] = (matches, matches_mask, matrix)

                if matches is not None:
                    log.info("Found match with %d keypoint matches", len(matches))
                    if self.output_dir:
                        pano_kpts, img_kpts = zip(*matches)
                        pano_img = self.pano_imgs[pano]
                        in_img = self.in_imgs[img]
                        matched_frame = cv2.drawMatches(pano_img, pano_kpts, in_img, img_kpts, matches_mask, None, matchColor=(0, 200, 0), flags=2)
                        cv2.imwrite(f"{self.output_dir}/m_{i_p}_{i_i}_matches.jpg", matched_frame)
                        
                        x_to_y = cv2.warpPerspective(in_img, np.linalg.inv(matrix), (pano_img.shape[1], pano_img.shape[0]))
                        overlay = cv2.addWeighted(pano_img, 0.5, x_to_y, 0.5, 0)
                        cv2.imwrite(f"{self.output_dir}/m_{i_p}_{i_i}_overlay.jpg", overlay)


    def save_to_cache(self, cache_dir="./tmp"):
        try:
            if len(self.pano_features) > 0:
                pano_path = pathlib.Path(cache_dir, "pano_features.dat").absolute()
                pano_path.parent.mkdir(exist_ok=True)
                for key, vals in self.pano_features.items():
                    save_keypoint_cache(key, vals[0], vals[1], cache_dir)
                log.info("%d panoramic image features written to cache", len(self.pano_features))
            else:
                log.info("No features in panoramic images to write to cache")

            if len(self.in_img_features) > 0:
                img_path = pathlib.Path(cache_dir).absolute()
                img_path.mkdir(exist_ok=True)
                for key, vals in self.in_img_features.items():
                    save_keypoint_cache(key, vals[0], vals[1], cache_dir)
                log.info("%d input image features written to cache", len(self.in_img_features))
            else:
                log.info("No features in input images to write to cache")
        except Exception as e:
            log.error("Could not write cache: %s", str(e), exc_info=e, stack_info=True)

    def load_from_cache(self, cache_dir="./tmp"):
        cache_path = pathlib.Path(cache_dir).absolute()

        try:
            if cache_path.exists():
                log.info("Loading from cache...", )
                for key in self.pano_imgs.keys():
                    keypoints, descriptors = load_keypoint_cache(key, cache_dir)
                    if keypoints is not None:
                        self.pano_features[key] = (keypoints, descriptors)
                log.info("%d panoramic image features loaded from cache", len(self.pano_features))
                for key in self.in_imgs.keys():
                    keypoints, descriptors = load_keypoint_cache(key, cache_dir)
                    if keypoints is not None:
                        self.in_img_features[key] = (keypoints, descriptors)
                log.info("%d input image features loaded from cache", len(self.in_img_features))
            else:
                log.info("No cache directory found")
        except Exception as e:
            log.error("Could not read cache: %s", str(e), exc_info=e, stack_info=True)
            return False
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiLoc Image Feature Matcher Tool")

    parser.add_argument("panoramic_image_folder", metavar="panIMG", type=pathlib.Path)
    parser.add_argument("input_image_folder", metavar="IMG", type=pathlib.Path)

    parser.add_argument("-o", "--output-dir", type=pathlib.Path, help="Path to output directory")
    parser.add_argument("-c", "--cache-features",  action='store_true')

    args = parser.parse_args()

    input_images_path: pathlib.Path = args.input_image_folder
    panoramic_images_path: pathlib.Path = args.panoramic_image_folder

    if not panoramic_images_path.exists():
        log.error(f"Cannot find panoramic image folder at {panoramic_images_path}")
        sys.exit(1)

    if not input_images_path.exists():
        log.error(f"Cannot find input image folder at {input_images_path}")
        sys.exit(1)

    file_types = ["jpg", "png", "jpeg"]

    input_images = []
    for file_ext in file_types:
        input_images.extend(glob.glob(str(input_images_path / ("*." + file_ext))))

    panoramic_images = []
    for file_ext in file_types:
        panoramic_images.extend(glob.glob(str(panoramic_images_path / ("*." + file_ext))))

    fm = FeatureMatching(panoramic_images, input_images, matcher=XFeatMatcher(), output_dir=args.output_dir)

    if args.cache_features:
        fm.load_from_cache()
    fm.find_features(skip_existing=True)

    if args.cache_features:
        fm.save_to_cache()

    fm.find_matches()

    print(fm.match_matrix)

    # pan_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360_Abpl m fl Bw und Kiesnest Betonwand_Panorama/Abplatzung,Bewehrung,GKS_0,5m_LR.jpg")
    # in_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360_Abpl m fl Bw und Kiesnest Betonwand_Panorama/smartphone-bilder/out.jpg")

    # pan_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360 Aufnahmen 02.03.2023 0,1mm Querriss/0,1er Riss ID05- RTC, MR, 2m.jpg")
    # in_img = cv2.imread("/home/patrick/sciebo/BIMKIT_DIENSTKETTE_INFRA1/Kamera-relokalsierung/Daten/RTC360 Aufnahmen 02.03.2023 0,1mm Querriss/smartphone-bilder/IMG_9221.jpg")
    # result = relocalize_multi(in_imgs, pan_img, show=args.show)

    # for r, input_image in zip(result, input_images):
    #     r["image"] = str(input_image)

    # print(result)
    # if not output_path.exists():
    #     output_path.mkdir(parents=True, exist_ok=True)
    # with open(output_path / "out.json", 'w') as f:
    #     json.dump(result, f, indent="\t", cls=NumpyArrayEncoder)

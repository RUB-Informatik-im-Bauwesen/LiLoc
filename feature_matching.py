import json
import os

import coloredlogs
import glob
import logging
import pathlib
import sys

import cv2
import numpy as np

from helpers import NumpyArrayEncoder, KeypointEncoder, multiencoder_factory, DMatchEncoder
from image_tools import read_images

# Create a logger object.
log = logging.getLogger("LiLoc")
coloredlogs.install(logger=log, level=logging.INFO)


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


def _find_features(matcher, img_db: dict, skip=None):
    if skip is None:
        skip = []

    feat_dict = {}
    for key in sorted(img_db.keys()):
        if key in skip:
            continue
        in_img = img_db[key]
        features = matcher.find_features(in_img)
        feat_dict[key] = features
        if features is None:
            raise AssertionError("Could not find features in image " + key)
        log.debug("Found %6d features in image %s", len(features[0]), key)
    return feat_dict


def plot_match_matrix(match_matrix, img_set_a_names, img_set_b_names, outfile=None, cap=100):
    if not np.any(match_matrix):
        log.info("No matches to print")
        return
    import matplotlib.pyplot as plt
    plt.imshow(np.clip(match_matrix, 0, cap), interpolation='nearest', cmap=plt.colormaps["Blues"])
    plt.title('Match Matrix')
    plt.colorbar()

    plt.ylabel("img_id")
    if img_set_a_names and len(img_set_a_names) < 20:
        len_a = len(img_set_a_names)
        tick_marks_y = np.arange(len_a)
        plt.yticks(tick_marks_y, img_set_a_names, fontsize=5)

    plt.xlabel("img_id")
    if img_set_b_names and len(img_set_a_names) < 20:
        len_b = len(img_set_b_names)
        tick_marks_x = np.arange(len_b)
        plt.xticks(tick_marks_x, img_set_b_names, rotation=90, fontsize=5)


    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()


def write_match_images(output_dir, img_a, img_a_kpts, img_b, img_b_kpts, match_id, matches_mask, matrix):
    matched_frame = cv2.drawMatches(img_a, img_a_kpts, img_b, img_b_kpts, matches_mask, None,
                                    matchColor=(0, 200, 0), flags=2)
    cv2.imwrite(f"{output_dir}/{match_id}_matches.jpg", matched_frame)
    a_to_b = cv2.warpPerspective(img_b, np.linalg.inv(matrix), (img_a.shape[1], img_a.shape[0]))
    cv2.imwrite(f"{output_dir}/{match_id}_tf.jpg", a_to_b)
    overlay = cv2.addWeighted(img_a, 0.5, a_to_b, 0.5, 0)
    cv2.imwrite(f"{output_dir}/{match_id}_overlay.jpg", overlay)


class ExhaustiveMatching:
    def __init__(self, img_set: list[str], max_image_size=2048, matcher=None, output_dir=""):
        """
        Exhaustive Matching for image sets. All images are matched to all other images in the set.

        :param img_set: List of image files or file paths to read
        :param max_image_size: If positive, the images are scaled to this value, retaining aspect ratio.
        :param matcher: Matcher class. Should support a find_feature(img) method and a match_points(pts1, des1, pts2, des2) method. See feature_matchers for more info.
        :param output_dir: Path to write the results to. Leave empty to discard results.
        """
        self.img_set_features = {}

        self.match_matrix = None
        self.matches = []

        self.output_dir = output_dir

        if matcher:
            self.matcher = matcher
        else:
            from feature_matchers.sift import SIFTMatcher
            self.matcher = SIFTMatcher()

        self.img_set_names, self.img_set = read_images(img_set, max_image_size)

    def find_features(self, skip_existing=False):
        log.info("Finding features in Image Set")
        _img_set_features = _find_features(self.matcher, self.img_set, self.img_set_features.keys() if skip_existing else [])

        self.img_set_features.update(_img_set_features)

    def find_matches(self):
        if len(self.img_set_features) == 0:
            self.find_features(skip_existing=True)

        len_imgs = len(self.img_set_names)
        self.match_matrix = np.zeros((len_imgs, len_imgs))
        self.matches = []

        log.info("Starting matching...")
        for i_p, img_a_id in enumerate(self.img_set_names):
            log.info("%d%% matches complete", i_p / len_imgs * 100.)
            for i_i, img_b_id in enumerate(self.img_set_names):
                if i_i < i_p:
                    self.match_matrix[i_p, i_i] = self.match_matrix[i_i, i_p]
                    continue
                if i_i == i_p:
                    self.match_matrix[i_p, i_i] = 1
                    continue
                img_a_pts, img_a_des = self.img_set_features[img_a_id]
                img_b_pts, img_b_des = self.img_set_features[img_b_id]
                matches, matches_mask, matrix = self.matcher.match_points(img_a_pts, img_a_des, img_b_pts, img_b_des)

                self.match_matrix[i_p, i_i] = len(matches) if matches else 0

                if matches is not None:
                    match_id = f"m_{i_p}_{i_i}"
                    self.matches.append({
                        "match_id": match_id,
                        "image_a": img_a_id,
                        "image_b": img_b_id,
                        "matches": len(matches),
                        "matrix": matrix
                    })
                    log.info("Found match with %d keypoint matches", len(matches))
                    if self.output_dir:
                        img_a_kpts, img_b_kpts = zip(*matches)
                        img_a = self.img_set[img_a_id]
                        img_b = self.img_set[img_b_id]
                        write_match_images(self.output_dir, img_a, img_a_kpts, img_b, img_b_kpts, match_id, matches_mask, matrix)
        log.info("Found %d matches", len(self.matches))
        if self.output_dir:
            with open(str(self.output_dir) + "/matches.json", 'w') as f:
                json.dump(self.matches, f, indent=2, cls=NumpyArrayEncoder)
            plot_match_matrix(self.match_matrix, self.img_set_names, self.img_set_names, str(self.output_dir) + "/match_matrix.svg")
            plot_match_matrix(self.match_matrix, self.img_set_names, self.img_set_names, str(self.output_dir) + "/match_matrix.png")
            log.info("Results written to %s", str(self.output_dir))

    def save_to_cache(self, cache_dir="./tmp"):
        pass

    def load_from_cache(self, cache_dir="./tmp"):
        pass


class CrossMatching:
    def __init__(self, img_set_a, img_set_b, max_image_size=2048, matcher=None, output_dir=""):
        """
        Exhaustive Matching for image sets. All images are matched to all other images in the set.

        :param img_set_a: List of image files or file paths to read for image set A
        :param img_set_b: List of image files or file paths to read for image set B
        :param max_image_size: If positive, the images are scaled to this value, retaining aspect ratio.
        :param matcher: Matcher class. Should support a find_feature(img) method and a match_points(pts1, des1, pts2, des2) method. See feature_matchers for more info.
        :param output_dir: Path to write the results to. Leave empty to discard results.
        """
        self.img_set_a_features = {}
        self.img_set_b_features = {}

        self.match_matrix = None
        self.matches = []

        self.output_dir = output_dir

        if matcher:
            self.matcher = matcher
        else:
            from feature_matchers.sift import SIFTMatcher
            self.matcher = SIFTMatcher()

        log.info("Reading Image Set A")
        self.img_set_a_names, self.img_set_a = read_images(img_set_a, max_image_size)

        log.info("Reading Image Set B")
        self.img_set_b_names, self.img_set_b = read_images(img_set_b, max_image_size)

        log.info("Finished reading images")

    def find_features(self, skip_existing=False):
        log.info(f"Finding features in Image Set A ({len(self.img_set_a)} images)")
        _img_set_a_features = _find_features(self.matcher, self.img_set_a, self.img_set_a_features.keys() if skip_existing else [])

        log.info(f"Finding features in Image Set B ({len(self.img_set_b)} images)")
        _img_set_b_features = _find_features(self.matcher, self.img_set_b, self.img_set_b_features.keys() if skip_existing else [])

        self.img_set_a_features.update(_img_set_a_features)
        self.img_set_b_features.update(_img_set_b_features)

    def find_matches(self):
        if len(self.img_set_a_features) == 0 or len(self.img_set_b_features) == 0:
            self.find_features(skip_existing=True)

        len_a = len(self.img_set_a_names)
        len_b = len(self.img_set_b_names)
        self.match_matrix = np.zeros((len_a, len_b))
        self.matches = []

        log.info("Starting matching...")
        for i_i, img_b_id in enumerate(self.img_set_b_names):
            log.info("%d%% matches complete", i_i / len_b * 100.)
            for i_p, img_a_id in enumerate(self.img_set_a_names):
                img_set_a_pts, img_set_a_des = self.img_set_a_features[img_a_id]
                img_pts, img_des = self.img_set_b_features[img_b_id]

                matches = None
                try:
                    matches, matches_mask, matrix = self.matcher.match_points(img_set_a_pts, img_set_a_des, img_pts, img_des)
                except Exception as e:
                    log.exception(f"Could not match {img_a_id} and {img_b_id}.", exc_info=e)
                    log.debug(f"Could not match {img_a_id} and {img_b_id}.", exc_info=e, stack_info=True)

                self.match_matrix[i_p, i_i] = len(matches) if matches else 0

                if matches is not None:
                    match_id = f"m_{i_p}_{i_i}"
                    self.matches.append({
                        "match_id": match_id,
                        "image_a": img_a_id,
                        "image_b": img_b_id,
                        "matches": len(matches),
                        "matrix": matrix
                    })
                    log.info("Found match with %d keypoint matches", len(matches))
                    if self.output_dir:
                        img_a_kpts, img_b_kpts = zip(*matches)
                        img_a = self.img_set_a[img_a_id]
                        img_b = self.img_set_b[img_b_id]
                        write_match_images(self.output_dir, img_a, img_a_kpts, img_b, img_b_kpts, match_id, matches_mask, matrix)

        log.info("Found %d matches", len(self.matches))
        if self.output_dir:
            with open(str(self.output_dir) + "/matches.json", 'w') as f:
                json.dump(self.matches, f, indent=2, cls=NumpyArrayEncoder)
            plot_match_matrix(self.match_matrix, self.img_set_a_names, self.img_set_b_names, str(self.output_dir) + "/match_matrix.svg")
            plot_match_matrix(self.match_matrix, self.img_set_a_names, self.img_set_b_names, str(self.output_dir) + "/match_matrix.png")
            log.info("Results written to %s", str(self.output_dir))


    def save_to_cache(self, cache_dir="./tmp"):
        try:
            if len(self.img_set_a_features) > 0:
                img_set_a_path = pathlib.Path(cache_dir).absolute()
                img_set_a_path.mkdir(exist_ok=True)
                for key, vals in self.img_set_a_features.items():
                    save_keypoint_cache(key, vals[0], vals[1], cache_dir)
                log.info("%d Image Set A image features written to cache", len(self.img_set_a_features))
            else:
                log.info("No features in Image Set A to write to cache")

            if len(self.img_set_b_features) > 0:
                img_path = pathlib.Path(cache_dir).absolute()
                img_path.mkdir(exist_ok=True)
                for key, vals in self.img_set_b_features.items():
                    save_keypoint_cache(key, vals[0], vals[1], cache_dir)
                log.info("%d Image Set B features written to cache", len(self.img_set_b_features))
            else:
                log.info("No features in Image Set B to write to cache")
        except Exception as e:
            log.error("Could not write cache: %s", str(e), exc_info=e, stack_info=True)

    def load_from_cache(self, cache_dir="./tmp"):
        cache_path = pathlib.Path(cache_dir).absolute()

        try:
            if cache_path.exists():
                log.info("Loading from cache...", )
                for key in self.img_set_a.keys():
                    keypoints, descriptors = load_keypoint_cache(key, cache_dir)
                    if keypoints is not None:
                        self.img_set_a_features[key] = (keypoints, descriptors)
                log.info("%d Image Set A features loaded from cache", len(self.img_set_a_features))
                for key in self.img_set_b.keys():
                    keypoints, descriptors = load_keypoint_cache(key, cache_dir)
                    if keypoints is not None:
                        self.img_set_b_features[key] = (keypoints, descriptors)
                log.info("%d Image Set B features loaded from cache", len(self.img_set_b_features))
            else:
                log.info("No cache directory found")
        except Exception as e:
            log.error("Could not read cache: %s", str(e), exc_info=e, stack_info=True)
            return False
        return True



def start_exhaustive_match(args):
    input_images_path: pathlib.Path = args.input_image_folder

    if not input_images_path.exists():
        log.error(f"Cannot find input image folder at {input_images_path}")
        sys.exit(1)

    file_types = ["jpg", "png", "jpeg"]

    input_images = []
    for file_ext in file_types:
        if args.recurse_dirs:
            input_images.extend(glob.glob(str(input_images_path / ("**/*." + file_ext)), recursive=True))
        else:
            input_images.extend(glob.glob(str(input_images_path / ("*." + file_ext))))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = input_images_path / "matches"
        os.makedirs(args.output_dir, exist_ok=True)

    if len(input_images) == 0:
        log.error("No input images found in %s", str(input_images_path))
        return

    if args.matcher == "SIFTkNN":
        from feature_matchers.sift import SIFTMatcher
        matcher = SIFTMatcher()
    else:
        from feature_matchers.xfeat_matcher import XFeatMatcher
        matcher = XFeatMatcher()

    em = ExhaustiveMatching(input_images, matcher=matcher, output_dir=str(args.output_dir))

    if args.cache_features:
        em.load_from_cache()
    em.find_features(skip_existing=True)

    if args.cache_features:
        em.save_to_cache()

    em.find_matches()


def start_cross_match(args):
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
    panoramic_images = []
    for file_ext in file_types:
        if args.recurse_dirs:
            input_images.extend(glob.glob(str(input_images_path / ("**/*." + file_ext)), recursive=True))
        else:
            input_images.extend(glob.glob(str(input_images_path / ("*." + file_ext))))

    for file_ext in file_types:
        if args.recurse_dirs:
            panoramic_images.extend(glob.glob(str(panoramic_images_path / ("**/*." + file_ext)), recursive=True))
        else:
            panoramic_images.extend(glob.glob(str(panoramic_images_path / ("*." + file_ext))))


    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = panoramic_images_path / "matches"
        os.makedirs(args.output_dir, exist_ok=True)

    if args.matcher == "SIFTkNN":
        from feature_matchers.sift import SIFTMatcher
        matcher = SIFTMatcher()
    else:
        from feature_matchers.xfeat_matcher import XFeatMatcher
        matcher = XFeatMatcher()

    fm = CrossMatching(panoramic_images, input_images, matcher=matcher, output_dir=args.output_dir)

    if args.cache_features:
        fm.load_from_cache()
    fm.find_features(skip_existing=True)

    if args.cache_features:
        fm.save_to_cache()

    fm.find_matches()

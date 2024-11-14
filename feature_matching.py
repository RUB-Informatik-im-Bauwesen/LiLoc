# import matplotlib.pyplot as plt
import argparse
import coloredlogs
import glob
import logging
import os
import pathlib
import sys
import uuid

import cv2

# Create a logger object.
log = logging.getLogger(__name__)
coloredlogs.install(logger=log, level=logging.DEBUG)


def read_images(in_imgs: list, max_image_size: int = 0) -> dict:
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
        dict: A dictionary with unique identifiers (UUIDs or file names) as keys and image data as values.
    """
    imgs = {}
    for in_img in in_imgs:
        try:
            if isinstance(in_img, cv2.Mat):
                new_id = uuid.uuid4()
                log.debug("Assigning new id %s to image", new_id)
                imgs[new_id] = resize_image(in_img, max_image_size)
            else:
                file_id = in_img.split(os.sep)[-1].split(".")[0]
                log.debug("Reading image %s as id %s", in_img, file_id)
                if file_id in imgs:
                    log.warning("Image with the name %s already present in data base. Skipping...", file_id)
                    continue
                im = cv2.imread(str(in_img))
                if max_image_size > 0:
                    imgs[file_id] = resize_image(im, max_image_size)
                else:
                    imgs[file_id] = im
        except Exception as e:
            log.warning("Could not read image %s: %s", in_img, str(e))

    return imgs


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


class FeatureMatching():
    def __init__(self, pan_imgs, in_imgs, max_image_size=2048):
        self.pan_imgs = pan_imgs
        self.in_imgs = in_imgs

        self.pano_features = {}
        self.in_img_features = {}

        log.info("Reading panoramic images")
        self.pan_imgs = read_images(pan_imgs, max_image_size)

        log.info("Reading input images")
        self.in_imgs = read_images(in_imgs, max_image_size)
        
        log.info("Finished reading images")


    def find_features(self):
        log.info("Finding features in panoramic images")
        self.pano_features = self._find_features(self.pan_imgs)
        log.info("Finding features in input images")
        self.in_img_features = self._find_features(self.in_imgs)

    def _find_features(self, img_db: dict):
        sift = getattr(self, "sift", cv2.SIFT.create())
        self.sift = sift

        feat_dict = {}
        for key  in sorted(img_db.keys()):
            in_img = img_db[key]
            features = sift.detectAndCompute(in_img, None)
            feat_dict[key] = features
            log.debug("Found %6d features in image %s", len(features[0]), key)
        return feat_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs Services")
    parser.add_argument("-o", "--outputPath", type=pathlib.Path, default=".",
                        help="Path to output directory")

    parser.add_argument("panoramic_image_folder", metavar="panIMG", type=pathlib.Path)
    parser.add_argument("input_image_folder", metavar="IMG", type=pathlib.Path)

    parser.add_argument("--show", action='store_true')

    args = parser.parse_args()
    output_path = pathlib.Path(args.outputPath)

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

    f = FeatureMatching(panoramic_images, input_images)
    f.find_features()

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



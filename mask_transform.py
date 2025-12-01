import json
from logging import DEBUG

import numpy as np
import cv2
import logging
import pathlib

log = logging.getLogger("LiLoc")

mask_key = "_mask.png"
mask_folder = "masks"

def alpha_to_black(img):
    alpha_channel = img[:, :, 3]
    rgb_channels = img[:, :, :3]
    # black Background Image
    black_background_image = np.zeros_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = black_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)

def display_mask_difference(folder, match):
    img_key_now, img_key_then = match["image_a"], match["image_b"]
    masks = {}
    imgs = {}

    folder = pathlib.Path(folder)
    mask_path_then = folder / mask_folder / (img_key_then + mask_key)
    log.info(f"Loading mask {mask_path_then}")
    masks[img_key_then] = cv2.imread(mask_path_then, cv2.IMREAD_UNCHANGED)

    mask_path_now = folder / mask_folder / (img_key_now + mask_key)
    log.info(f"Loading mask {mask_path_now}")
    masks[img_key_now] = cv2.imread(mask_path_now, cv2.IMREAD_UNCHANGED)

    img_path_then = folder / (img_key_then + ".jpg")
    log.info(f"Loading image {img_path_then}")
    imgs[img_key_then] = cv2.imread(img_path_then)

    img_path_now = folder / (img_key_now + ".jpg")
    log.info(f"Loading image {img_path_now}")
    imgs[img_key_now] = cv2.imread(img_path_now)

    try:
        assert imgs[img_key_then] is not None
        assert imgs[img_key_now] is not None
        assert masks[img_key_then] is not None
        assert masks[img_key_now] is not None
    except AssertionError as e:
        log.error(e)
        return

    t1 = match["matrix"]

    log.info(f"Transforming mask")
    a_to_b = cv2.warpPerspective(masks[img_key_then], np.linalg.inv(t1),
                                 (imgs[img_key_now].shape[1], imgs[img_key_now].shape[0]))

    log.info(f"Computing mask difference")
    difference = np.ones_like(masks[img_key_now]) * 255
    difference[..., 3] = np.logical_xor(masks[img_key_now][..., 3] > 0, a_to_b[..., 3] > 0).astype(np.uint8) * 255

    log.info(f"Display")
    overlay_now = cv2.addWeighted(imgs[img_key_now], 1, cv2.resize(alpha_to_black(masks[img_key_now]),
                                                                   (imgs[img_key_now].shape[1],
                                                                    imgs[img_key_now].shape[0])), 1, 0)
    overlay_then = cv2.addWeighted(imgs[img_key_now], 1, alpha_to_black(a_to_b), 1, 0)
    overlay_diff = cv2.addWeighted(imgs[img_key_now], 1, alpha_to_black(difference), 1, 0)

    overlay_now = cv2.putText(overlay_now, "Image: " + img_key_now + ", Mask: " + img_key_now, (16, 32),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    overlay_then = cv2.putText(overlay_then, "Image: " + img_key_now + ", Mask: " + img_key_then, (16, 32),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    overlay_diff = cv2.putText(overlay_diff, "Image: " + img_key_now + ", Mask: " + img_key_then + ", Difference",
                               (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    cv2.imshow("now", imgs[img_key_now])
    cv2.imshow("then", imgs[img_key_then])
    cv2.imshow("now_mask", cv2.addWeighted(imgs[img_key_now], 1, alpha_to_black(masks[img_key_now]), 1, 0))
    cv2.imshow("then_mask", cv2.addWeighted(imgs[img_key_then], 1, alpha_to_black(masks[img_key_then]), 1, 0))
    cv2.imshow("difference", cv2.addWeighted(imgs[img_key_now], 1, alpha_to_black(difference), 1, 0))
    cv2.imshow("overlay_compare", overlay_now)

    now = False
    while cv2.waitKey(1000) < 0:
        now = not now
        cv2.imshow("overlay_compare", overlay_then if now else overlay_diff)

def test():
    import coloredlogs
    coloredlogs.install(logger=log, level=logging.INFO)
    log.setLevel(logging.DEBUG)

    import argparse
    import pathlib
    argp = argparse.ArgumentParser()
    argp.add_argument("folder", default=".", type=pathlib.Path)

    args = argp.parse_args()

    folder = args.folder

    with open(folder / "matches/matches.json") as f:
        mdict = json.load(f)
        matches = mdict["matches"]
        images = mdict["image_set"]

    for match in matches:
        display_mask_difference(folder, match)

if __name__ == '__main__':
    test()


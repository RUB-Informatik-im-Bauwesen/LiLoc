import json
import numpy as np
import cv2

folder = "out/512/"

with open(folder + "matches.json") as f:
        matches = json.load(f)
print(matches[0])

img_key_now = "2017_1"
img_key_then = "2005"

masks = {}
imgs = {}
masks[img_key_then]=cv2.imread(folder + img_key_then + ".png", cv2.IMREAD_UNCHANGED)
masks[img_key_now]=cv2.imread(folder + img_key_now + ".png", cv2.IMREAD_UNCHANGED)
imgs[img_key_then]=cv2.imread(folder + img_key_then + ".jpg")
imgs[img_key_now]=cv2.imread(folder + img_key_now + ".jpg")

m1 = list(filter(lambda m: m["image_a"] == img_key_now and m["image_b"] == img_key_then, matches))
t1 = m1[0]["matrix"]

a_to_b = cv2.warpPerspective(masks[img_key_then], np.linalg.inv(t1), (imgs[img_key_now].shape[1], imgs[img_key_now].shape[0]))

difference = np.array(masks[img_key_now])
difference[...,3] = np.clip(masks[img_key_now][..., 3] - a_to_b[...,3], 0, 255)

def alpha_to_black(img):
    alpha_channel = img[:,:,3]
    rgb_channels = img[:, :, :3]
    # black Background Image
    black_background_image = np.zeros_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = black_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)

overlay_now = cv2.addWeighted(imgs[img_key_now], 1, cv2.resize(alpha_to_black(masks[img_key_now]), (imgs[img_key_now].shape[1], imgs[img_key_now].shape[0])), 1, 0)
overlay_then = cv2.addWeighted(imgs[img_key_now], 1, alpha_to_black(a_to_b), 1, 0)
overlay_diff = cv2.addWeighted(imgs[img_key_now], 1, alpha_to_black(difference), 1, 0)

overlay_now = cv2.putText(overlay_now, "Image: " + img_key_now + ", Mask: " + img_key_now, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
overlay_then = cv2.putText(overlay_then, "Image: " + img_key_now + ", Mask: " + img_key_then, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
overlay_diff = cv2.putText(overlay_diff, "Image: " + img_key_now + ", Mask: " + img_key_then + ", Difference", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

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

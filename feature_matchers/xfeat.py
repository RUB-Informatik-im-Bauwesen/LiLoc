"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm

from modules.xfeat import XFeat

import cv2

xfeat = XFeat()

#Random input
x = cv2.imread(r"E:\dev\python\liloc\data\match_test\141202-130.JPG")
y = cv2.imread(r"E:\dev\python\liloc\data\match_test\DSCN1853 BLICK VON UNTEN.JPG")

#Simple inference with batch = 1
outputA = xfeat.detectAndCompute(x, top_k = 4096)[0]
print("----------------")
print("keypoints: ", outputA['keypoints'].shape)
print("descriptors: ", outputA['descriptors'].shape)
print("scores: ", outputA['scores'].shape)
print("----------------\n")
outputB = xfeat.detectAndCompute(y, top_k = 4096)[0]
print("----------------")
print("keypoints: ", outputB['keypoints'].shape)
print("descriptors: ", outputB['descriptors'].shape)
print("scores: ", outputB['scores'].shape)
print("----------------\n")

# Match two images with sparse features
mkpts_0, mkpts_1 = xfeat.match_xfeat(x, y)

xc = xfeat.detectAndCompute(x, None)[0]
kpts1, descs1 = xc['keypoints'], xc['descriptors']
yc = xfeat.detectAndCompute(y, None)[0]
kpts2, descs2 = yc['keypoints'], yc['descriptors']


idx0, idx1 = xfeat.match(descs1, descs2, 0.82)
points1 = kpts1[idx0].cpu().numpy()
points2 = kpts2[idx1].cpu().numpy()

if len(mkpts_0) > 10 and len(mkpts_1) > 10:
    # Find homography
    H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 4., maxIters=700,
                                         confidence=0.995)
    inliers = inliers.flatten() > 0

    kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in points1[inliers]]
    kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in points2[inliers]]
    good_matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

    # Draw matches
    matched_frame = cv2.drawMatches(x, kp1, y, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
    cv2.imshow("matched_frame", matched_frame)

    x_to_y = cv2.warpPerspective(x, H, (y.shape[1], y.shape[0]))
    overlay1 = cv2.addWeighted(y, 0.1, x_to_y, 0.9, 0)
    cv2.imshow("overlay1", overlay1)
    overlay2 = cv2.addWeighted(y, 0.5, x_to_y, 0.5, 0)
    cv2.imshow("overlay2", overlay2)
    overlay3 = cv2.addWeighted(y, 0.9, x_to_y, 0.1, 0)
    cv2.imshow("overlay3", overlay3)
    cv2.waitKey(0)

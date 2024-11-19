import sys
sys.path.append("./accelerated_features/")
from modules.xfeat import XFeat

import numpy as np
import os
import torch
import tqdm

import cv2

import logging
log = logging.getLogger("LiLoc")

class XFeatMatcher:
    def __init__(self, match_threshold=20, min_keypoints=10):
        self.feature_detector = XFeat()
        self.match_threshold = match_threshold
        self.min_keypoints = min_keypoints
        self.use_lighterglue = True

    def find_features(self, img):
        if img.ndim > 3:
            batchmode = True
        else:
            batchmode = False
        
        xc = self.feature_detector.detectAndCompute(img, None)

        if self.use_lighterglue:
            xc[0]["image_size"] = (img.shape[1], img.shape[0])
            return xc[0], None
        if not batchmode:
            kpts1, descs1 = xc[0]['keypoints'], xc[0]['descriptors']
            return kpts1, descs1
        else:
            kpts = [x['keypoints'] for x in xc]
            descs = [x['descriptors'] for x in xc]
            return kpts, descs
    
    def match_points(self, pts1, des1, pts2, des2):
        if self.use_lighterglue:
            points1, points2, good_matches = self.feature_detector.match_lighterglue(pts1, pts2, 0.82)
            if len(points1) >= self.min_keypoints and len(points2) >= self.min_keypoints and len(good_matches > self.match_threshold):
                H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 4., maxIters=700,
                                                    confidence=0.995)

                kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in points1]
                kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in points2]
                good_matches = [cv2.DMatch(i, i, 0) for i in range(len(good_matches))]

                return list(zip(kp1, kp2)), good_matches, H
            else:
                log.debug("Match rejected with %d matches", len(good_matches))
                return None, None, None
        else:
            idx0, idx1 = self.feature_detector.match(des1, des2, 0.82)
            points1 = pts1[idx0].cpu().numpy()
            points2 = pts2[idx1].cpu().numpy()

            if len(points1) < self.min_keypoints or len(points2) < self.min_keypoints:
                log.debug("Match rejected, not enough keypoints (%d)", min(len(points1), len(points2)))
                return None, None, None

            # Find homography
            H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 4., maxIters=700,
                                                confidence=0.995)
            inliers = inliers.flatten() > 0

            kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in points1[inliers]]
            kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in points2[inliers]]
            good_matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

            if len(good_matches) >= self.match_threshold:
                return list(zip(kp1, kp2)), good_matches, H
            else:
                log.debug("Match rejected with %d matches", len(good_matches))
                return None, None, None




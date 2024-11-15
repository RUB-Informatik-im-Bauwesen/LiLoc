import cv2
import numpy as np

import logging, coloredlogs

# Create a logger object.
log = logging.getLogger("SIFTMatcher")
coloredlogs.install(logger=log, level=logging.DEBUG)

sift_params = dict(algorithm=1, trees=10), dict(checks=500)


class SIFTMatcher:
    def __init__(self, min_match_count=30):
        self.feature_detector = cv2.SIFT.create()
        self.feature_detector.setNOctaveLayers(8)

        self.min_match_count = min_match_count

    def match_points(self, pts1, des1, pts2, des2):
        flann = cv2.FlannBasedMatcher(*sift_params)
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

        if len(good) < self.min_match_count:
            log.debug("Not enough matches are found - {}/{}".format(len(good), self.min_match_count))
            return matches, None, None

        log.debug(f"Found {len(good)} matches, attempting homography")
        src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matches_mask = mask.ravel().tolist()

        return matches, matches_mask, M

    def find_features(self, img):
        keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
        return keypoints, descriptors

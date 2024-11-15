import cv2

from feature_matchers import sift
import image_tools
from image_tools import resize_image

img = image_tools.resize_image(cv2.imread("data/match_test/141202-130.JPG"), 2048)
scanimg = image_tools.resize_image(cv2.imread("data/match_test/DSCN1853 BLICK VON UNTEN.JPG"), 2048)

sm = sift.SIFTMatcher()
sm.min_match_count = 10

imgfeat = sm.find_features(img)
scanfeat = sm.find_features(scanimg)

matches, matches_mask, M = sm.match_points(imgfeat[0], imgfeat[1], scanfeat[0], scanfeat[1])

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matches_mask,
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = cv2.drawMatchesKnn(img, imgfeat[0], scanimg, scanfeat[0], matches, None, **draw_params)
cv2.imshow("matches", resize_image(img3, 1080))
cv2.waitKey()

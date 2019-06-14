# !/usr/bin/env pyhton
# encoding: utf-8
# File: pairwise_stitch.py
# Author: Toxic
# Created on 2019/6/13 下午4:36

import cv2
import numpy as np

# global variable
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0
sift = cv2.xfeatures2d.SIFT_create()
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def partition(image, kp, des, thresh=0.10):
    h, w = image.shape[0:2]
    top, bottom, left, right = int(h * thresh), int(h - h * thresh), int(w * thresh), int(w - w * thresh)
    kp_top, kp_bottom, kp_left, kp_right = [], [], [], []
    des_top, des_bottom, des_left, des_right = [], [], [], []
    collection = zip(kp, des)
    for item in collection:
        pt = item[0].pt
        if pt[0] <= left:
            kp_left.append(item[0])
            des_left.append(item[1])
        if pt[0] >= right:
            kp_right.append(item[0])
            des_right.append(item[1])
        if pt[1] <= top:
            kp_top.append(item[0])
            des_top.append(item[1])
        if pt[1] >= bottom:
            kp_bottom.append(item[0])
            des_bottom.append(item[1])
    return kp_top, np.array(des_top), kp_bottom, np.array(des_bottom),\
           kp_left, np.array(des_left), kp_right, np.array(des_right)


def pairwise_stitch(tiff_file_1, tiff_file_2, mode='XY', debug=False):
    ret_1, img_list_1 = cv2.imreadmulti(tiff_file_1)
    if not ret_1:
        raise ValueError("error image path: '%s' ." % tiff_file_1)
    ret_2, img_list_2 = cv2.imreadmulti(tiff_file_2)
    if not ret_2:
        raise ValueError("error image path: '%s' ." % tiff_file_2)

    if mode == 'XY':
        img1 = np.uint8(np.max(img_list_1, axis=0))
        img2 = np.uint8(np.max(img_list_2, axis=0))
    elif mode == 'Z':
        img1 = img_list_1[0]
        # img_list_2[-1]
        # only for testing
        img2 = img_list_2[1]
    else:
        raise ValueError("unsupported stitch mode: '%s' ." % mode)


    # compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if mode == 'XY':
        kp1_top, des1_top, kp1_bottom, des1_bottom,\
        kp1_left, des1_left, kp1_right, des1_right = partition(img1, kp1, des1)
        kp2_top, des2_top, kp2_bottom, des2_bottom,\
        kp2_left, des2_left, kp2_right, des2_right = partition(img2, kp2, des2)
        matches = [
            flann.knnMatch(des1_bottom, des2_top, k=2),
            flann.knnMatch(des1_top, des2_bottom, k=2),
            flann.knnMatch(des1_right, des2_left, k=2),
            flann.knnMatch(des1_left, des2_right, k=2)
        ]
        good = []
        for i in range(4):
            temp_good = []
            for m, n in matches[i]:
                if m.distance < 0.5 * n.distance:
                    temp_good.append(m)
            good.append(temp_good)
        index = -1
        count = 0
        for i, item in enumerate(good):
            if len(item) > count:
                index = i
                count = len(item)

        if count < MIN_MATCH_COUNT:
            print('not enough matching pairs')
            return None

        good = good[index]
        kp1 = [kp1_bottom, kp1_top, kp1_right, kp1_left][index]
        kp2 = [kp2_top, kp2_bottom, kp2_left, kp2_right][index]
    else: # mode = 'Z'
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)


    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    x_diff = src_pts[:, 0] - dst_pts[:, 0]
    y_diff = src_pts[:, 1] - dst_pts[:, 1]

    x_diff_rep = np.repeat(x_diff, len(x_diff)).reshape(len(x_diff), -1)
    x_diff_rep_T = x_diff_rep.T
    y_diff_rep = np.repeat(y_diff, len(y_diff)).reshape(len(x_diff), -1)
    y_diff_rep_T = y_diff_rep.T

    x_diff_2 = x_diff_rep - x_diff_rep_T
    y_diff_2 = y_diff_rep - y_diff_rep_T

    x_diff_2_sq = x_diff_2 ** 2
    y_diff_2_sq = y_diff_2 ** 2
    dis_mtx = x_diff_2_sq + y_diff_2_sq
    dis_sum = np.sum(dis_mtx, axis=0)

    target_idx = np.argmin(dis_sum)
    x_offset = int(round(x_diff[target_idx]))
    y_offset = int(round(y_diff[target_idx]))


    if debug:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        print('x_offset: %d, y_offset: %d' % (x_offset, y_offset))
        w = 2 * img2.shape[1] + img1.shape[1]
        h = 2 * img2.shape[0] + img1.shape[0]
        fusion_map = np.zeros((h, w, 3), dtype=np.uint8)
        start_x = img2.shape[1]
        start_y = img2.shape[0]

        fusion_map[start_y: start_y + img1.shape[0], start_x: start_x + img1.shape[1]] = img1
        fusion_map[start_y + y_offset: start_y + y_offset + img2.shape[0],
                   start_x + x_offset: start_x + x_offset + img2.shape[1]] = img2

        cv2.imwrite('debug.png', fusion_map)
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
        cv2.imwrite('match.png', match_img)
    return x_offset, y_offset


if __name__ == '__main__':
    pairwise_stitch('images/5.tif', 'images/7.tif', mode='XY')
    # pairwise_stitch('images/8.tif', 'images_z/8-1.tif', mode='Z')

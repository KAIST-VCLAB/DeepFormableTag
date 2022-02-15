# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
import cv2
from cv2 import aruco

def get_aruco_dict(name, default=aruco.DICT_5X5_100):
    """
    For a given string returns corresponding aruco dictionary if exists.
    Check cv2.aruco.__dict__ keys for supported markers.
    """
    name = 'DICT_' + name.upper()
    d = default
    if name in aruco.__dict__:
        d = aruco.__dict__[name]
    return aruco.Dictionary_get(d)

def detect_aruco_markers(
    img,
    aruco_dict,
    mtx=None,
    detect_params=aruco.DetectorParameters_create(),
    subpix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001),
    max_winsize=9,
    min_winsize=2,
):
    """
    Detects aruco markers and refines corners in subpixel accuracy.
    """
    marker_corners, ids, tmp = cv2.aruco.detectMarkers(
        img, aruco_dict,
        parameters=detect_params,
        cameraMatrix=mtx)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1]==3 else img
    if len(marker_corners) > 0 and subpix_criteria is not None:
        for corners in marker_corners:
            dif = corners[0]- np.roll(corners[0],2)
            dist_avg = np.average(np.linalg.norm(dif, axis=1))
            win_size = min(max(min_winsize, int(dist_avg/12)), max_winsize)
            cv2.cornerSubPix(
                gray_img, corners,
                winSize=(win_size, win_size),
                zeroZone=(-1, -1),
                criteria=subpix_criteria)
    return marker_corners, ids, tmp


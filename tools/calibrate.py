# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import json, argparse, random
import cv2
from cv2 import aruco
import numpy as np
from pathlib import Path

from deepformable.utils import (
    img_flexible_reader, get_aruco_dict, detect_aruco_markers, calculate_board_dims)

def detect_charuco_corners(
    img,
    aruco_dict,
    charuco_board,
    min_corners=5,
):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1]==3 else img
    marker_corners, ids, _ = detect_aruco_markers(gray_img, aruco_dict)
    num_corners, img_corners, corner_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, ids, gray_img, charuco_board)
    if num_corners >= min_corners:
        row_size = charuco_board.getChessboardSize()[0]
        corner_world = np.zeros((num_corners, 3), np.float32)
        for i, index in enumerate(corner_ids):
            corner_world[i, 0] = index[0] % (row_size - 1)
            corner_world[i, 1] = index[0] // (row_size - 1)
        return num_corners, np.squeeze(img_corners), np.squeeze(corner_world)
    return 0, None, None


def calculate_sharpness(img, pts):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1]==3 else img
    mask = np.zeros(gray_img.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    masked_values = cv2.Laplacian(gray_img, cv2.CV_64F)[mask == 255]
    return masked_values.std() * (masked_values.shape[0] ** 0.5)


def calculate_reprojection_error(frames_info, mtx, dist):
    repr_dist = []
    for _, img_corners, corner_world in frames_info:
        retval, rvec, tvec = cv2.solvePnP(corner_world, img_corners, mtx, dist)
        projected_points, _ = cv2.projectPoints(corner_world, rvec, tvec, mtx, dist)
        dif = projected_points.squeeze() - img_corners.squeeze()
        repr_dist.append(np.linalg.norm(dif, axis=1))
    
    repr_dist = np.concatenate(repr_dist)
    return np.average(repr_dist), np.std(repr_dist)


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, default='config.json', help='Input config file')
    parser.add_argument(
        '-s', '--save-params', action='store_true', help='Saves calibration params to provided json file')
    parser.add_argument(
        '--sharpness-cut-ratio', type=float, default=0.7, help='Ignores remaining portion of frames, sorted by sharpnes')
    parser.add_argument(
        '--random-cut-ratio', type=float, default=0.5, help='Ignores close frames by provided extent')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Show steps if enabled')
    parser.add_argument(
        '-t', '--test-params', action='store_true', help='Test reprojection error with current params')
    return parser


if __name__ == '__main__':
    args = setup_argparse().parse_args()

    # Read the config json into python format
    data_json_path = Path(args.input)
    with open(data_json_path) as cfg_file:
        cfg = json.load(cfg_file)
    if args.verbose: print("Config loaded!")

    # Get necessary values for calibration
    calib_video_path = (data_json_path.parent / cfg["calib_video"]).resolve()
    aruco_dict = get_aruco_dict(cfg['aruco_dict'])
    boards_dict = {i['board_name']: i for i in cfg['boards']}
    charuco_board_info = boards_dict['charuco']['descriptions'][0]

    board_dims = calculate_board_dims(boards_dict['charuco'])
    charuco_scale = float(min([board_dims[i] / charuco_board_info['size'][i] for i in range(2)]) // 1)
    dims = (charuco_scale * charuco_board_info['size'][1], charuco_scale * charuco_board_info['size'][0])

    tag_length = aruco_dict.markerSize + 2 * cfg['border_bits']
    square_length = 2 * charuco_board_info['tag_border'] + tag_length
    charuco_board = aruco.CharucoBoard_create(*charuco_board_info['size'], square_length, tag_length, aruco_dict)
    board_size = charuco_board.getChessboardSize()

    if args.verbose: print("Dims parsed, detecting corners")
    frames_info = []
    for frame in img_flexible_reader(calib_video_path, tqdm_on=args.verbose):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_corners, img_corners, corner_world = detect_charuco_corners(
            frame_gray, aruco_dict, charuco_board, 
            min_corners=(board_size[0] - 1) * (board_size[1] - 1))  # Accept only if all corners are detected
        
        if num_corners == 0: continue
        
        pts = np.array([
            img_corners[0], img_corners[board_size[0] - 2],
            img_corners[-1], img_corners[-board_size[0] + 1]], dtype=np.int32)  # Provide corners of the board
        sharpness_value = calculate_sharpness(frame_gray, pts)
        frames_info.append((sharpness_value, img_corners, corner_world * charuco_scale))

    # Works better than previous calibration implementation 
    if args.verbose: print("Thresholding frames.")
    cut_index = int(len(frames_info) * args.sharpness_cut_ratio)
    thresh_frames_info = sorted(frames_info, reverse=True, key=lambda x: x[0])[:cut_index]
    random_select_index = int(len(thresh_frames_info) * args.random_cut_ratio)
    random.shuffle(thresh_frames_info)
    thresh_frames_info = thresh_frames_info[:random_select_index]

    image_points = np.array([i[1] for i in thresh_frames_info])
    world_points = np.array([i[2] for i in thresh_frames_info])

    if not args.test_params:
        if args.verbose: print("Calibrating...")
        ret, mtx, dist, _, _ = cv2.calibrateCamera(
            world_points,image_points,(frame.shape[1], frame.shape[0]), None, None)
        if args.verbose:
            avg, std = calculate_reprojection_error(thresh_frames_info, mtx, dist)
            print("Training(selected frames) Reprojection error:{:.4f}, std: {:.4f} for selected frames".format(avg, std))
            avg, std = calculate_reprojection_error(frames_info, mtx, dist)
            print("Testing    (all frames)   Reprojection error:{:.4f}, std: {:.4f} for selected frames".format(avg, std))
    else:
        mtx, dist = np.array(cfg["calib_mtx"]), np.array(cfg["calib_dist"])
        avg, std = calculate_reprojection_error(frames_info, mtx, dist)
        print("Testing    (all frames)   Reprojection error:{:.4f}, std: {:.4f} for selected frames".format(avg, std))
    
    if args.save_params:
        cfg["calib_mtx"] = mtx.tolist()
        cfg["calib_dist"] = dist.tolist()
        with open(data_json_path, 'w') as cfg_file:
            json.dump(cfg, cfg_file, indent=4)
        if args.verbose: print("Config saved!")
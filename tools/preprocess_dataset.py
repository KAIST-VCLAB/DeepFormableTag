# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import json, argparse, os, sys
import datetime
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from deepformable.utils import (
        img_flexible_reader, get_aruco_dict, 
        detect_aruco_markers, calculate_board_dims,
        NoInpaint, OpenCVInpaint)
except ImportError:
    root_path = Path(__file__).parent.resolve()
    print("Deepformable installation is not found, will add the paths manually")
    sys.path.insert(0, str(root_path.parent / "deepformable/utils"))
    sys.path.insert(0, str(root_path.parent / "inpaint"))
    from aruco_utils import get_aruco_dict, detect_aruco_markers
    from board_utils import calculate_board_dims
    from general_utils import img_flexible_reader
    from inpaint_utils import NoInpaint, OpenCVInpaint
    from deepfill import DeepfillInpaint


def read_process_configs(input_path, output_path):
    # Start board_id from 1 to fix COCO mapping
    board_id, cam_id = 1, 0
    marker_configs = []
    for p in input_path.rglob('**/config.json'):
        with open(p.absolute()) as cfg_file:
            cfg = json.load(cfg_file)
            # Specify paths
            cfg['video_dir'] = str(p.parent / cfg['video_dir'])
            cfg['rel_path'] = str(Path(os.path.relpath(str(p), str(input_path))).parent)
            cfg['output_path'] = str(output_path / cfg['rel_path'])
            cfg['cam_id'] = cam_id
            # Remove charuco boards from config
            cfg['boards'] = [i for i in cfg['boards'] if i['board_name'] != 'charuco']   
            # Process boards
            cfg['boardid_to_world'], cfg['boardid_to_board'] = {}, {}
            cfg['markerid_to_board'], cfg['markerid_to_world'] = {}, {}
            cfg['boardid_to_markersworld'] = {}
            for board in cfg['boards']:
                # Assign unique id to each board
                board['category_id'] = board_id
                # Calculate dimensions and assign
                board_dims, margin = calculate_board_dims(board), board['paper_margins']
                cfg['boardid_to_world'][board_id] = np.array([
                        (0,0,0), (board_dims[0]+margin, 0, 0),(0, board_dims[1]+margin, 0),
                        (board_dims[0]+margin, board_dims[1]+margin, 0)])
                cfg['boardid_to_board'][board_id] = board
                cfg['boardid_to_markersworld'][board_id] = []
                for d in board['descriptions']:
                    if d['type'] != 'aruco': continue   # External markers are not included
                    cfg['markerid_to_board'][d['marker_id']] = board
                    marker_world = np.array(d['location'])[[0,1,3,2]] + [margin/2, margin/2, 0]
                    cfg['markerid_to_world'][d['marker_id']] = marker_world
                    cfg['boardid_to_markersworld'][board_id].append(marker_world)
                cfg['boardid_to_markersworld'][board_id] = np.array(cfg['boardid_to_markersworld'][board_id])
                board_id += 1
            cam_id += 1
        marker_configs.append(cfg)
    return marker_configs

def preprocess_cfg(
    cfg,
    inpaint_save_fn,
    image_counter=0,
    ann_counter=0,
    skip_frames=0,
    least_pose_markers=2,
    verbose=True,
):
    boards, cam_id = cfg['boards'], cfg['cam_id']
    output_path, rel_path = cfg['output_path'], cfg['rel_path']
    os.makedirs(str(output_path), exist_ok=True)
    mtx, dist = np.array(cfg['calib_mtx']), np.array(cfg['calib_dist'])
    aruco_dict = get_aruco_dict(cfg['aruco_dict'])

    boardid_to_world = cfg['boardid_to_world']
    markerid_to_board, markerid_to_world = cfg['markerid_to_board'], cfg['markerid_to_world']

    # Start processing frames
    images_info, annotations  = [], []
    for frame in img_flexible_reader(
        cfg['video_dir'], 
        tqdm_on=verbose, 
        num_skip_frames=skip_frames,
    ):
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        gray_img = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        marker_corners, ids, _ = detect_aruco_markers(gray_img, aruco_dict, mtx=mtx)
        if ids is None: continue
        boardid_to_detectedmarker = {board['category_id']: [] for board in boards}
        # Detect corresponding boards and ignore wrt least_pose_markers
        for [marker_id], [corners] in zip(ids, marker_corners):
            board = markerid_to_board.get(marker_id, None)
            if board is None: continue
            boardid_to_detectedmarker[board['category_id']].append((marker_id, corners))
        for board_id in list(boardid_to_detectedmarker.keys()):
            if len(boardid_to_detectedmarker[board_id]) < least_pose_markers:
                del boardid_to_detectedmarker[board_id]
        if len(boardid_to_detectedmarker) == 0: continue    # Ignore if nothing detected properly

        # Calculate board annotations
        cur_annotations = []
        for board_id, detected_marker in boardid_to_detectedmarker.items():
            markers_screen = np.array([i[1] for i in detected_marker]).reshape(-1,2)
            markers_world = np.array(
                [markerid_to_world[i[0]] for i in detected_marker], dtype=np.float32).reshape(-1,3)
            board_world = boardid_to_world[board_id]
            _, rvec, tvec = cv2.solvePnP(markers_world, markers_screen, mtx, None)
            board_screen_repr = cv2.projectPoints(board_world, rvec, tvec, mtx, None)[0].reshape(4,2)
            cur_annotations.append({
                'id': ann_counter, 'image_id': image_counter, 'iscrowd': 0,
                'category_id': board_id, 
                'segmentation': [board_screen_repr[[0,1,3,2]].flatten().tolist()],
                'rvec': rvec.squeeze().tolist(), 'tvec': tvec.squeeze().tolist()
            })
            ann_counter += 1

        # Inpaint and save
        file_name = "{}.png".format(str(image_counter).zfill(8))
        # data = os.path.join(output_path, file_name), undistorted_frame, cur_annotations, cfg
        cur_markers_world = [cfg['boardid_to_markersworld'][i['category_id']] for i in cur_annotations]
        data = os.path.join(output_path, file_name), undistorted_frame, cur_annotations, cur_markers_world, mtx
        inpaint_save_fn(data)

        annotations.extend(cur_annotations)
        images_info.append({
            'id': image_counter,
            'file_name': os.path.join(rel_path, file_name),
            'height': undistorted_frame.shape[0],
            'width': undistorted_frame.shape[1],
            'camera_id': cam_id
        })
        image_counter += 1
    return images_info, annotations

def compute_additional_info(output_path, annotations, verbose=False):
    # Create camera map
    camera_map = {i['id']: i for i in annotations['cameras']}
    # Create annotations map
    annotations_map = {i['id']: [] for i in annotations['annotations']}
    for ann in annotations['annotations']:
        annotations_map[ann['image_id']].append(ann)

    normal_points = np.array([[0,0,0,1],[0,0,-10,1]])

    if verbose: pbar = tqdm(total=len(annotations['images']), smoothing=0)

    for image_info in annotations['images']:
        camera = camera_map[image_info['camera_id']]
        mtx = np.array(camera['mtx'])
        inv_mtx = np.linalg.inv(mtx)
        img = cv2.imread(str(output_path/image_info['file_name']))
        img_blurred = cv2.GaussianBlur(img, (17,17), 5)
        img_blurred_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)

        ann_list = annotations_map[image_info['id']]
        for ann in ann_list:
            board_screen = np.array(ann['segmentation']).reshape(-1,2)
            rvec, _ = cv2.Rodrigues(np.array(ann['rvec']))
            extrinsic = np.concatenate(
                [rvec, np.array(ann['tvec'])[:,np.newaxis]], axis=-1)
            points_cam_space = np.matmul(normal_points, extrinsic.T)
            normal = points_cam_space[1] - points_cam_space[0]
            normal /= np.linalg.norm(normal)

            mask = np.ones(img.shape)
            mask = cv2.fillConvexPoly(mask, np.int32(board_screen), [0]*3, cv2.LINE_4)
            masked_array = np.ma.masked_array(img_blurred, mask=mask)
            gray_masked_array = np.ma.masked_array(img_blurred_gray, mask=mask[...,0])
            max_index = np.unravel_index(gray_masked_array.argmax(), gray_masked_array.shape)

            view_dir = -np.matmul([max_index[1], max_index[0],1], inv_mtx.T)
            view_dir /= np.linalg.norm(view_dir)
            refl_dir = (np.dot(view_dir, normal) * 2.0) * normal - view_dir
            refl_dir /= np.linalg.norm(refl_dir)

            # Convert pixel centers (0, 0) to COCO format (0.5, 0.5)
            board_screen += 0.5
            min_val = board_screen.min(axis=0)
            dif = board_screen.max(axis=0) - min_val
            ann['normal'] = normal.tolist()
            ann['view_dir'] = view_dir.tolist()
            ann['refl_dir'] = refl_dir.tolist()
            ann['brightness_max'] = int(img_blurred_gray[max_index])
            ann['avg_color'] = np.mean(masked_array, axis=(0,1)).tolist()
            ann['segmentation'] = [board_screen.flatten().tolist()]
            ann['bbox'] = [*list(min_val), *list(dif)]
        if verbose: pbar.update()
    if verbose: pbar.close()

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, default='config.json', help='Input directory to traverse config files')
    parser.add_argument(
        '-o', '--output', type=str, default='data_out', help='Output directory to save frames and annotations')
    parser.add_argument(
        '--inpaint-method', type=str, default='None', help='Inpainting method to choose (None/opencv/deepfill)')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Show steps if enabled')
    parser.add_argument(
        '--least-pose-markers', type=int, default=2, 
        help='Least number of markers need to be detected to predict board pose')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='The number of frames to be skipped while selecting frames')
    return parser 


if __name__ == '__main__':
    args = setup_argparse().parse_args()

    output_path = Path(args.output).absolute()
    marker_configs = read_process_configs(
        Path(args.input).absolute().resolve(), output_path.resolve())
    if args.verbose: print("Configs are loaded!")

    inpaint_save_fn = None
    if  "none" in args.inpaint_method.lower():
        inpaint_save_fn = NoInpaint()
    elif 'opencv' in args.inpaint_method.lower():
        inpaint_save_fn = OpenCVInpaint()
    elif 'deepfill' in args.inpaint_method.lower():
        inpaint_save_fn = DeepfillInpaint()
    else:
        raise ValueError("Unknown inpainting method")

    images_info, annotations_info = [], []
    for i, cfg in enumerate(marker_configs):
        if args.verbose: print('Processing #{}/{} directory:{}'.format(i+1, len(marker_configs), cfg['output_path']))
        images_info_cur, annotations_info_cur = preprocess_cfg(
            cfg, inpaint_save_fn, verbose=args.verbose, image_counter=len(images_info), 
            skip_frames=args.skip_frames, ann_counter=len(annotations_info),
            least_pose_markers=args.least_pose_markers)
        images_info.extend(images_info_cur), annotations_info.extend(annotations_info_cur)
    
    description = "DeepformableTag training dataset"
    if args.inpaint_method != 'None':
        description += " inpainted by " + args.inpaint_method + " method."
    else:
        description += " without inpainting."
    data_info = {
        'description': description, 'version': '1.0', 'year': 2021,
        'contributor': 'VCLAB', 'date_created': str(datetime.datetime.now())}
    categories, cameras = [], []
    for cfg in marker_configs:
        cameras.append({
            'id': cfg['cam_id'], 'name': cfg['camera'],
            'mtx': cfg['calib_mtx'], 'dist': [0.0]*5})
        for board_id, board in cfg['boardid_to_board'].items():
            marker_ids = [d['marker_id'] for d in board['descriptions']]
            markers_world = [cfg['markerid_to_world'][i].tolist() for i in marker_ids]
            categories.append({
                'id': board_id, 'marker_ids': marker_ids,
                'markers_world': markers_world, 'name': board['board_name'],
                'board_world': cfg['boardid_to_world'][board_id].tolist()})
            
    annotations = {
        'info': data_info,
        'images': images_info, 'annotations': annotations_info,
        'categories': categories, 'cameras': cameras}

    inpaint_save_fn.wait_finish()
    if args.verbose: print("Images are saved, computing additional info")
    
    compute_additional_info(output_path, annotations, verbose=args.verbose)
    
    with open(output_path / 'annotations.json', 'w') as file:
        file.write(json.dumps(annotations))
    if args.verbose: print("Json annotation file is added to the directory!")

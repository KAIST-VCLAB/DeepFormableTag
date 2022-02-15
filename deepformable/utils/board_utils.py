# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import json
import numpy as np
from pathlib import Path

import shapely
from shapely.geometry import MultiPolygon, Polygon


paper_sizes = {
    "a2": (420.0, 420.0 * (2.0 ** 0.5)),
    "a3": (210.0 * (2.0 ** 0.5), 420.0), # 296, 420
    "a4": (210.0, 210.0 * (2.0 ** 0.5)), # 210, 296
    "a5": (210.0 / (2.0 ** 0.5), 210.0),
    "a6": (105.0, 105.0 * (2.0 ** 0.5)),
    "a3-s": (305.0, 457.0),
}


def calculate_board_dims(board):
    if isinstance(board['paper_type'], str):
        paper_size = paper_sizes.get(board['paper_type'], 'a4')
    elif isinstance(board['paper_type'], list):
        paper_size = board['paper_type']
    margins = board.get('paper_margins', 10.5)
    board_dims = (paper_size[0] - margins, paper_size[1] - margins)
    return board_dims


def is_polygon_intersects(src_poly, polygons):
    if len(polygons) == 0:
        return False
    src_poly = Polygon(src_poly)
    polygons = MultiPolygon([Polygon(i) for i in polygons])
    return polygons.intersects(src_poly)


def marker_placer(
    board_size=(210,296),
    marker_min=40,
    marker_max=140,
    num_classes=64,
    class_array=[],
    safety_size=4,
    random_trials=75,
    p_reg=[0.2, 0.3, 0.2, 0.15, 0, 0.15],
    p_reg_rand=[0.57, 0.37, 0.06],
):
    """
    TODO: This method requires bug-fix and clean-up!!
    """
    def place_random(marker_min, marker_max, board_size):
        polygons = np.empty((0,4,2))
        val_range = range(marker_min,marker_max+1)
        p = np.flip(np.array(val_range))
        p = p / np.sum(p)
        for _ in range(random_trials):
            marker_size = np.random.choice(val_range, p=p)
            src_poly_margin = np.array([[0,0],[1,0],[1,1],[0,1]]) * (marker_size+8)
            theta = np.random.uniform(0, np.pi)
            rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta),  np.cos(theta)]])
            src_poly_margin = np.matmul(src_poly_margin, rotMatrix.T)
            src_poly_margin -= np.amin(src_poly_margin, axis=0)
            bounding_box = np.amax(src_poly_margin, axis=0)
            offset = np.random.uniform((0,0), board_size - bounding_box)
            src_poly_margin += offset
            if not is_polygon_intersects(src_poly_margin, polygons):
                src_poly = np.array([[0,0],[1,0],[1,1],[0,1]]) * marker_size
                src_poly = np.matmul(src_poly, rotMatrix.T)
                src_poly -= np.amin(src_poly, axis=0)
                src_poly += offset + (bounding_box - np.amax(src_poly, axis=0))/2
                polygons = np.append(polygons, [src_poly], axis=0)
        polygons -= np.amin(polygons, axis=(0,1))
        return polygons
    
    def place_regular(marker_min, marker_max, board_size):
        regular_max = marker_min + (marker_max-marker_min)//3 + 1
        norm_dims = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) * 0.5
        marker_size = np.random.randint(marker_min, regular_max)
        
        def checkerboard_regular():
            angle = lambda: np.random.choice([0,1,2,3])*np.pi/2
            r, c = marker_size, marker_size
            ofs = marker_size
            return r, c, ofs, angle

        def checkerboard_random():
            r, c, ofs, _ = checkerboard_regular()
            angle = lambda: np.random.uniform(0, np.pi)
            return r, c, ofs, angle

        def checkerboard_dense():
            _, _, ofs, angle = checkerboard_regular()
            r = np.random.randint(marker_size//2+3, marker_size+4)
            ofs += np.random.randint(safety_size, 15)
            c = ofs + np.random.randint(safety_size, 15)
            return r, c, ofs, angle

        def grid_regular():
            r, c, _, angle = checkerboard_regular()
            ofs = 0
            r += np.random.randint(safety_size, marker_size)
            c = np.random.randint(safety_size, marker_size)
            return r, c, ofs, angle
        
        def grid_regular2():
            r, c, _, angle = checkerboard_regular()
            angle = lambda: 0
            ofs = 0
            # r += 10
            # c = 10
            r += marker_size/3
            c = marker_size/3
            return r, c, ofs, angle

        def grid_skewed():
            r, c, ofs, angle = checkerboard_regular()
            r += np.random.randint(safety_size, marker_size)
            c = ofs
            return r, c, ofs, angle

        row_gap, column_gap, even_row_offset, angle_choice = np.random.choice([
            checkerboard_regular, checkerboard_random, checkerboard_dense,
            grid_regular, grid_regular2, grid_skewed],
            p=p_reg
        )()
        
        polygons = np.empty((0,4,2))

        cur_pos, index = np.array([0.0, 0.0]), 0
        while np.all(cur_pos + marker_size < board_size):
            while np.all(cur_pos + marker_size < board_size):
                theta = angle_choice()
                rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta),  np.cos(theta)]])
                cur_poly = np.matmul(norm_dims, rotMatrix.T) * marker_size
                cur_poly += marker_size/2
                polygons = np.append(polygons, [cur_pos + cur_poly], axis=0)
                cur_pos += [marker_size+column_gap, 0]
            cur_pos = np.array([0 if index%2 else even_row_offset, cur_pos[1]+row_gap])
            index += 1
        polygons -= np.amin(polygons, axis=(0,1))
        pol_max = np.amax(polygons, axis=(0,1))
        large_index = (pol_max > board_size)
        if large_index.any():
            polygons *= (np.array(board_size)[large_index] / pol_max[large_index])
        return polygons
    
    def place_single(marker_min, marker_max, board_size):
        marker_size = min(board_size) * np.random.uniform(0.8, 0.999)
        polygon = np.array([[0,0],[1,0],[1,1],[0,1]]) * marker_size
        return polygon.reshape(1,4,2)
    
    placer = np.random.choice([place_regular, place_random, place_single], p=p_reg_rand)
    polygons = placer(marker_min, marker_max, board_size)
    polygons += (board_size - np.amax(polygons, axis=(0,1)))/2
    markers = np.dstack([polygons, np.zeros((*polygons.shape[:2],1))])[:,[0,1,3,2]]

    if len(class_array) == 0:
        classes = np.random.randint(0, num_classes, size=len(markers))
        # return [], []
    else:
        classes = []
        for _ in range(len(markers)):
            if len(class_array) == 0:
                break
            # val = random.choice(class_array)
            val = class_array[0]
            class_array.remove(val)
            classes.append(val)
    return markers[:len(classes)], classes

def image_placer(
    board_size=(210,296),
    marker_ratio=(4,3),
    margin_ratio=0.8,
    marker_min=40,
    marker_max=140,
    num_classes=64,
    class_array=[],
    safety_size=4,
    random_trials=75,
    p_reg=[0.2, 0.3, 0.2, 0.15, 0, 0.15],
    p_reg_rand=[0.57, 0.37, 0.06],
):
    swp = False
    if board_size[0] < board_size[1]:
        board_size = (board_size[1], board_size[0])
        swp = True
    
    mx, my = board_size[0] / 2, board_size[1] / 2
    limx, limy = board_size[0] * margin_ratio, board_size[1] * margin_ratio
    mulx, muly = (limx - mx) / marker_ratio[0], (limy - my) / marker_ratio[1]
    mul = min(mulx, muly)
    ux, uy = mx - mul * marker_ratio[0], my - mul * marker_ratio[1]
    bx, by = mx + mul * marker_ratio[0], my + mul * marker_ratio[1]
    markers = []
    if not swp:
        markers = [[[ux, uy, 0.0], [bx, uy, 0.0], [bx, by, 0.0], [ux, by, 0.0]]]
    else:
        markers = [[[by, ux, 0.0], [by, bx, 0.0], [uy, bx, 0.0], [uy, ux, 0.0]]]
    
    flip = np.random.randint(2, size=1).astype(np.bool).item()
    if flip:
        markers = [markers[0][2:], markers[0][:2]]

    classes = [np.random.randint(num_classes, size=1).item()]
    return markers, classes

def marker_metadata_loader(cfg, marker_config_file):
    from detectron2.data import MetadataCatalog
    marker_config_path = Path(marker_config_file)
    if marker_config_path.exists():
        with open(marker_config_path) as cfg_file:
            marker_config = json.load(cfg_file)

        markers = sorted(marker_config['markers'], key=lambda x: x['marker_id'])
        binary_messages = [[float(i) for i in m['binary']] for m in markers]
        thing_classes = [m['text'] for m in markers]

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

        metadata_name = cfg.DATASETS.TEST[0]
        MetadataCatalog.get(metadata_name).set(
            messages=binary_messages, thing_classes=thing_classes)
    else:
        return False
    return True
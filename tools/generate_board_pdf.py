# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
This code creates pdfs for given configs.
"""
import os, json, argparse
from pathlib import Path
from collections import OrderedDict
import cairo
import numpy as np
from cv2 import aruco
import torch

from deepformable.utils import (
    get_aruco_dict, calculate_board_dims, get_cfg)
from deepformable.modeling import build_marker_generator

def setup_cairo(board, output_file):
    board_dims = calculate_board_dims(board)
    width, height = [i * 7.2 / 2.54 for i in board_dims]
    cairo_surface = cairo.PDFSurface(output_file, width, height)
    cairo_ctx = cairo.Context(cairo_surface)
    cairo_ctx.scale(7.2 / 2.54, 7.2 / 2.54)  # Scale to mm back
    return cairo_surface, cairo_ctx
    # print("Cairo setup is done, paper scales are: {:.2f}mm-{:2f}mm".format(paper_size[0], paper_size[1]))
    # print("Please don't select scale to fit option during printing for accurate board scales")

def draw_marker(cairo_ctx, marker, locations):
    marker = np.array(marker)
    if marker.shape[-1] != 3:
        marker = np.repeat(marker, 3, axis=-1).reshape(*marker.shape, 3)
    locations = np.array(locations)[:,:2]
    x_tick = (locations[1] - locations[0])/marker.shape[0]
    y_tick = (locations[2] - locations[0])/marker.shape[1]
    for i in range(marker.shape[1]):
        for j in range(marker.shape[0]):
            if (marker[j,i] == [1,1,1]).all():
                continue
            pos = (locations[0] + i * x_tick + j * y_tick)
            cairo_ctx.move_to(*(locations[0] + i * x_tick + j * y_tick))
            cairo_ctx.line_to(*(locations[0] + (i+1) * x_tick + j * y_tick))
            cairo_ctx.line_to(*(locations[0] + (i+1) * x_tick + (j+1) * y_tick))
            cairo_ctx.line_to(*(locations[0] + (i) * x_tick + (j+1) * y_tick))
            cairo_ctx.close_path()
            cairo_ctx.set_source_rgb(*marker[j,i])
            cairo_ctx.fill_preserve()
            cairo_ctx.set_line_width (0.001)
            cairo_ctx.set_source_rgb(*marker[j,i])
            cairo_ctx.stroke()

def draw_rectangle(cairo_ctx, locations, color=(0.0,0.0,0.0)):
    return draw_marker(cairo_ctx, np.array([[color]]), locations)

def draw_cutlines(cairo_ctx, locations, margin, color=(0.0,0.0,0.0)):
    locations = np.array(locations)[:,:2]
    x_tick = (locations[1] - locations[0])
    x_tick /= np.linalg.norm(x_tick)
    y_tick = (locations[2] - locations[0])
    y_tick /= np.linalg.norm(y_tick)

    locations[0] -= (x_tick+y_tick)*margin
    locations[1] += margin*(x_tick-y_tick)
    locations[3] += margin*(x_tick+y_tick)
    locations[2] -= margin*(x_tick-y_tick)
    
    cairo_ctx.set_line_width(0.1)
    cairo_ctx.move_to(*locations[0])
    cairo_ctx.line_to(*locations[1])
    cairo_ctx.line_to(*locations[3])
    cairo_ctx.line_to(*locations[2])
    
    cairo_ctx.close_path()
    cairo_ctx.set_source_rgb(*color)
    cairo_ctx.stroke()

def draw_marker_board(cairo_ctx, board, markers):
    aruco_markers, markers = markers
    for d in board["descriptions"]:
        # Get location for the object to draw
        loc = d['location']
        if d['type'] == 'marker':
            draw_marker(cairo_ctx, markers[d['marker_id']], loc)
        elif d['type'] == 'aruco':
            draw_marker(cairo_ctx, aruco_markers[d['marker_id']], loc)
        elif d['type'] == 'rectangle':
            draw_rectangle(cairo_ctx, loc, d['color'])
        else:
            raise ValueError("Unknown type of element, possible ones: marker, aruco, rectangle")

def draw_cutlines_board(cairo_ctx, board):
    for d in board["descriptions"]:
        loc = d['location']
        if d['type'] == 'marker':
            draw_cutlines(cairo_ctx, loc, d.get('cutline_margins', 5))
        elif d['type'] == 'aruco':
            draw_cutlines(cairo_ctx, loc, d.get('cutline_margins', 5))
        elif d['type'] == 'rectangle':
            draw_cutlines(cairo_ctx, loc, d.get('cutline_margins', 0))
        else:
            raise ValueError("Unknown type of element, possible ones: marker, aruco, rectangle")

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, default='config.json', help='Input config file')
    parser.add_argument(
        '-o', '--output', type=str, default='boards', help='Output board directory')
    parser.add_argument(
        '-d', '--draw-cutlines', action='store_true', help='Creates cutline pdfs')
    parser.add_argument(
        '--marker-config-file', type=str, default='config.yaml', help='marker model config file')
    parser.add_argument(
        '--model-weights', type=str, default='weights.pth', help='marker model config file')
    return parser


if __name__ == '__main__':
    args = setup_argparse().parse_args()

    # Read the config json into python format
    data_json_path = Path(args.input)
    with open(data_json_path) as cfg_file:
        cfg = json.load(cfg_file)
    print("Config loaded!")

    # Load the aruco markers
    aruco_dict = get_aruco_dict(cfg['aruco_dict'])
    tag_length = aruco_dict.markerSize + 2 * cfg['border_bits']
    markers_aruco = []
    for i in range(aruco_dict.bytesList.shape[0]):
        markers_aruco.append(aruco_dict.drawMarker(i, tag_length, borderBits=cfg['border_bits']))
    
    # Load the model markers
    markers_model = None
    config_path = Path(args.marker_config_file)
    model_weights_path = Path(args.model_weights)
    if 'markers' in cfg and config_path.exists() and model_weights_path.exists():
        # Load config
        model_cfg = get_cfg()
        model_cfg.merge_from_file(config_path)

        # Change default device if GPU is not available
        if not torch.cuda.is_available():
            model_cfg.MODEL.DEVICE = "cpu"

        # Get markers into tensor
        markers = sorted(cfg['markers'], key=lambda x: x['marker_id'])
        binary_messages = torch.tensor([[float(i) for i in m['binary']] for m in markers])
        # Construct generator and load weights
        model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(binary_messages)
        marker_generator = build_marker_generator(model_cfg)
        weights = torch.load(model_weights_path, map_location=model_cfg.MODEL.DEVICE)
        state_dict = marker_generator.state_dict()
        converted_weights = OrderedDict()
        for key, value in weights['model'].items():
            if 'marker_generator' in key:
                items = key.split('.')
                param_name = ".".join(items[items.index('marker_generator')+1:])
                if param_name in state_dict:
                    converted_weights[param_name] = value
                else:
                    print("- Ignoring:", param_name)
        marker_generator.load_state_dict(converted_weights)
        marker_generator.messages = binary_messages.to(marker_generator.device)
        print("Model loaded!")
        markers_model = marker_generator.get_markers_numpy([i for i in range(len(binary_messages))])
    else:
        print("WARNING: Could not load the model!")

    markers = np.array(markers_aruco), markers_model

    os.makedirs(args.output, exist_ok=True)
    output_path = Path(args.output)

    charuco_board = None
    for board in cfg['boards']:
        if board['board_name'] == "charuco": 
            charuco_board = board
            continue

        output_file = str(output_path / f"{board['board_name']}.pdf")
        cairo_surface, cairo_ctx = setup_cairo(board, output_file)
        draw_marker_board(cairo_ctx, board, markers)
        
        cairo_surface.flush()
        cairo_surface.finish()

        # Also create cutlines if requested
        if args.draw_cutlines:
            output_file = str(output_path / f"{board['board_name']}_cutline.pdf")
            cairo_surface, cairo_ctx = setup_cairo(board, output_file)
            draw_cutlines_board(cairo_ctx, board)
            cairo_surface.flush()
            cairo_surface.finish()

    # Draw charuco board if exists
    if charuco_board:
        # Calculate paper dimensions for the charuco
        charuco_board_info = charuco_board['descriptions'][0]
        dims = charuco_board_info['dims']
        if dims == "max":
            board_dims = calculate_board_dims(charuco_board)
            charuco_scale = float(min([board_dims[i] / charuco_board_info['size'][i] for i in range(2)]) // 1)
            dims = (charuco_scale * charuco_board_info['size'][1], charuco_scale * charuco_board_info['size'][0])
        elif dims == 2:
            charuco_scale = dims[0] / charuco_board_info['size'][1]
        else:
            print("Please provide 2 dimensional size for the charuco dimensions")
            raise
        print("Charuco unit size(length of two neighbouring corners) is calculated as {}mm".format(charuco_scale))

        cairo_surface, cairo_ctx = setup_cairo(
            charuco_board, str(output_path / "charuco.pdf"))

        # Create the board
        square_length = 2 * charuco_board_info['tag_border'] + tag_length
        charuco_board = aruco.CharucoBoard_create(*charuco_board_info['size'], square_length, tag_length, aruco_dict)
        
        # Draw the board
        draw_size = tuple([int(i * charuco_board.getSquareLength()) for i in charuco_board.getChessboardSize()])
        board_svg = charuco_board.draw(draw_size)
        loc = [[0,0], [dims[0], 0], [0, dims[1]], dims]
        draw_marker(cairo_ctx, board_svg, loc) 
        cairo_surface.flush()
        cairo_surface.finish()

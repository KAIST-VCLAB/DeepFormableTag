"""
# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
This code creates random board configurations.
"""
import json, argparse
from pathlib import Path
import numpy as np
import requests

from deepformable.utils.general_utils import if_continue_execution
from deepformable.utils import (
    if_continue_execution, get_aruco_dict,
    calculate_board_dims, marker_placer
)

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, default='config.json', help='Input config file')
    parser.add_argument(
        '-o', '--output', type=str, default='out.json', help='Output config file')
    parser.add_argument(
        '--random-trials', type=int, default=200, help='Number of random trials to place non-overlapping markers')
    parser.add_argument(
        '--marker-min', type=int, default=25, help="Smallest marker size in cm's")
    parser.add_argument(
        '--marker-max', type=int, default=110, help="Biggest marker size in cm's")
    parser.add_argument(
        '--safety-size', type=int, default=10, help="Marker safety distance to each other")
    parser.add_argument(
        '--generate-aruco', action='store_true', help='Marker type will be aruco instead of general markers')
    parser.add_argument(
        '--num-bits', type=int, default=36, help="Number of bits markers encode")
    parser.add_argument(
        '--num-markers', type=int, default=0, help="Number of markers to be used")
    
    return parser


if __name__ == '__main__':
    args = setup_argparse().parse_args()

    # Read the config json into python format
    data_json_path = Path(args.input)
    with open(data_json_path) as cfg_file:
        cfg = json.load(cfg_file)
    print("Config loaded!")

    aruco_dict = get_aruco_dict(cfg['aruco_dict'])
    
    num_markers, num_bits = args.num_markers, args.num_bits
    if args.generate_aruco and num_markers == 0:
        num_markers = len(aruco_dict.bytesList)
    assert num_markers > 0, "Enter positive number for the number of markers"

    # Standard settings
    p_reg, p_reg_rand=[0.0,0.25,0.25,0.25,0.25,0.0], [0.5,0.5,0.0]
    # p_reg, p_reg_rand = [1/6,1/6,1/6,1/6,1/6,1/6], [1/3, 1/3, 1/3])
    # p_reg, p_reg_rand=[0.1,0.2,0.2,0.2,0.2,0.1], [0.4,0.4,0.2]
    # p_reg, p_reg_rand = [0,0,0,0,1,0], [1.0,0.0,0.0]
    
    # Generate markers
    if not args.generate_aruco:
        # Generate unique binary messages
        binary_messages = np.unique(
            np.random.randint(0, 2, (num_markers, num_bits)), axis=0)
        while len(binary_messages) != num_markers:
            additional_messages = np.random.randint(
                0, 2, (num_markers-len(binary_messages), num_bits))
            binary_messages = np.concatenate([binary_messages, additional_messages], axis=0)
            binary_messages = np.unique(binary_messages, axis=0)
        
        # Load some random english words from web as a message
        word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = requests.get(word_site)
        words = [
            word.decode("utf-8") for word in response.content.splitlines() 
            if len(word) > 3
        ] if response.ok else None
        text_messages = np.random.choice(words, num_markers, replace=False)

        # Generate information and save it to config
        markers = []
        for index in range(num_markers):
            markers.append({
                "marker_id": index,
                "binary": "".join([str(i) for i in binary_messages[index]]),
                "text": text_messages[index] if text_messages is not None else ""
            })
        cfg['markers'] = markers

    
    # Generate boards
    class_indexes = list(range(num_markers))
    for board in cfg['boards']:
        if board['board_name'] == 'charuco':
            continue
        board_dims = calculate_board_dims(board)
        markers, marker_indexes = marker_placer(
            board_dims,
            random_trials=args.random_trials,
            marker_min=args.marker_min, marker_max=args.marker_max,
            class_array=class_indexes, safety_size=args.safety_size, 
            p_reg=p_reg, p_reg_rand=p_reg_rand)
        
        descriptions = []
        for marker, marker_id in zip(markers, marker_indexes):
            description = {
                "type": "aruco" if args.generate_aruco else "marker",
                "location": marker.tolist(),
                "marker_id": int(marker_id),
            }
            descriptions.append(description)
        
        board['descriptions'] = descriptions

        if len(class_indexes) == 0:
            break

    if args.output != '':
        if args.input == args.output and not if_continue_execution(
            "This will override input file, continue? (yes/no): "):
            exit(0)
        print("Saving confing!")
        with open(args.output, 'w') as cfg_file:
            json.dump(cfg, cfg_file, indent=4)

# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import os
import pathlib
import cv2
from tqdm import tqdm
from os.path import isfile, join

def img_flexible_reader(directory, num_skip_frames=0, tqdm_on=False):
    """
    Reads videos in sorted order for a give directory, 
    if path is a file tries to read it.
    """
    directory = pathlib.Path(directory)
    directory = str(directory.resolve())

    files = [directory]
    if not isfile(directory):
        files = [join(directory, f) for f in sorted(os.listdir(directory)) if isfile(join(directory, f))]
    
    total_frames = 0
    for f in files:
        cap = cv2.VideoCapture(f)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = frame_count if frame_count >= 0 else 1
        total_frames += int(frame_count)

    skip_count = 0
    if tqdm_on: pbar = tqdm(total=total_frames, smoothing=0)
    for f in files:
        cap = cv2.VideoCapture(f)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_frames = int(num_frames if num_frames >= 0 else 1)
        prev_frame, frame = None, None
        for _ in range(num_frames):
            prev_frame = frame if frame is not None else prev_frame
            success, frame = cap.read()
            if tqdm_on: pbar.update()
            skip_count += 1
            if success and skip_count == num_skip_frames+1:
                skip_count = 0
                yield frame
    if tqdm_on: pbar.close()

def if_continue_execution(message="Continue (yes/no): "):
    while True:
        answer = str(input(message)).lower()
        if answer in ["yes", "y", ""]:
            return True
        elif answer in ["no", "n"]:
            return False
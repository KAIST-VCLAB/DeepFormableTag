# DeepFormableTag Tools Instructions

The **tools** folder includes some of the necessary tools and 
components to interact with the marker system. 
It provides functionalities to create dataset, evaluate models, and
visualize the predictions.
Later we will provide the training code here in the near future.

Here is a short summary of what each file does:
- [preprocess_data.py](#preprocessing-videos-into-dataset) uses folder of videos and
preprocess it into the modified COCO format for the training.
- [generate_board_json.py](#marker-config-file) generates json files with random 
board arrangement.
- [generate_board_pdf.py](#generating-board-pdfs) generates vector graphics pdfs
of boards drawn.
- [calibrate.py](#calibration) uses video frames to calibrate cameras which is later
used to rectify the images while generating the COCO formatted dataset.
- [predictor_demo.py](../README.md/#predictor-demo) visualizes the predictions.
- [eval.py](../README.md/#evaluation-on-test-data) evaluates model given the dataset.

## Preprocessing Videos into Dataset

In the preprocessing step, our `tools/preprocess_data.py` applies the following steps to create the dataset.
1. Loads videos from provided directories
2. For a frame from video, detects the markers and board position, creates annotations
3. Inpaints the markers (optional)
4. Saves the processed frames and combines annotations in the COCO format

- There are three different inpainting methods, to use the DeepFill method, which we use to create our training and testing dataset, build and run the environment:
  ```bash
  # Creates inpainting environment
  docker build -t deepfill-inpaint -f docker/DeepfillInpaint.Dockerfile .   
  # Runs the preprocessing code
  docker run --rm -it --runtime=nvidia --ipc=host -v $PWD:/host -v /home/myaldiz/Data/Deepformable:/Data deepfill-inpaint \
    /bin/sh -c 'cd /host; python tools/preprocess_dataset.py -v -i /Data/Dataset/train-raw/ -o /Data/Dataset/train --inpaint-method deepfill'	
  ```
- `preprocess_dataset.py` file has several options you might want to use:
  - `--least-pose-markers` option ignores detected board if provided number of markers are not detected for that board.
  - `--skip-frames` skips frames for processing. You can use this option to generate toy dataset.
  - `-i` specifies input folder for dataset, `-o` is the output folder. 
  - `-v` option is prints the progress to terminal.
- You can use OpenCV inpainting method as well but the inpainting quality is not as good, so we recommend deepfill. 
However, this option requires larger GPU memory (>12GB).
- You need to [download weights](https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO) for deepfill into `inpaint/inpaint_weights` folder.

You can download training and testing videos from [this link](https://drive.google.com/drive/folders/1picphIb6Hbj6pM3Wu_Vxu53wzKBV0jdV?usp=sharing).

## Preparing a Custom Dataset

Dataset preparation requires variety of scripts to process the video frames into the final COCO format dataset.
In summary you need to:
- Generate the `config.json` file with the boards
- Generate PDFs for the boards and capture datasets
- Calibrate cameras with charuco board and save it to config
- Capture and preprocess videos into frames.

### Marker Config File

Marker config file is a file in the `json` format to specify board arrangements.
We provide a [template config file](../files/template_config.json).
Here are some information about they keys:
- `aruco_dict` is used to generate aruco markers for the board.
- `video_dir` is the folder to search for video files.
- `calib_video` is the video that will be used to calibrate the camera.
- `boards` provides information about the boards to be detected.
  - You can provide a name for the board to be created, paper margin sizes and
  descriptions of (ie: location, id) markers to be placed.
- `markers` provide a text for class id, binary message and name for the markers 
used in visualization demos.
  ```json
  {
      "marker_id": 0, 
      "binary": "111011011001000111101111100011011011",
      "text": "informational"
  }
  ``` 

`tools/generate_board_json.py` code reads the board sizes written in the config file and replaces them with random marker configurations. You need to enter the board names, type and dims. An example template config file is given at [template_config.json](../files/template_config.json).
Here is an example script:
```bash
python tools/generate_board_json.py -i files/template_config.json -o output/config.json
```

### Generating Board PDFs

Here is an example script to generate pdfs of boards:
```bash
python tools/generate_board_pdf.py -i tools/config.json -o tools/boards
```
In the config file board descriptions, if the type of marker is `aruco` then it will produce aruco markers.
However, if the type is `marker`, then supplied model will be used to generate the markers like below:
```bash
python tools/generate_board_pdf.py -i files/template_config.json -o output/boards \
  --marker-config-file configs/deepformable-main.yaml \
  --model-weights models/deepformable_model.pth
```
Try to print the board pdfs without scaling. This way dimensions specified in the `location` key for each marker will match the printed size.

### Calibration

Python script for calibration is located at `tools/calibrate.py`. Config file must include the relative path to calibration video or folder such as `calib_video: "../../calib/canon_28mm_5x5.MOV"`. Following script will calculate camera calibration parameters:
```bash
python tools/calibrate.py -i /Data/Datasets/PlacementDataset_Nov2/train-raw/28mm/config.json
```
Notes:
- To save the parameters into json file use `-s` option.
- If using datasets we provided, they most likely include calibration parameters, no need to run the scripts again.
- Our code thresholds blurry frames. Depending on video length, it may take longer time. You can change the ratio of the selected frames using arguments.

## Miscellaneous

### Running a Docker Container with X11 Window Support on Mac
In order to create windows from docker on Mac, you can follow below, taken from [stackoverflow](https://stackoverflow.com/questions/37826094/xt-error-cant-open-display-if-using-default-display).
```bash
# Below will install required things to the host
brew install socat
brew install --cask xquartz
# From xquartz Preferences/Security allow connections from network clients

# Create port for display
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" 

# In another terminal run below to create the container
docker run --rm -it --privileged --ipc=host -v $PWD:/host -e DISPLAY=docker.for.mac.host.internal:0  -v /tmp/.X11-unix:/tmp/.X11-unix deepformable /bin/sh -c 'cd /host; python -m pip install -e .; bash'
```

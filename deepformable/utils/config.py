# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
from detectron2.config import CfgNode as CN

def add_marker_generator_config(cfg: CN):
    _C = cfg
    _C.MODEL.MARKER_GENERATOR = CN()
    _C.MODEL.MARKER_GENERATOR.NAME = "GeneralizedGenerator"
    _C.MODEL.MARKER_GENERATOR.MARKER_SIZE = (16, 16)
    _C.MODEL.MARKER_GENERATOR.ARUCO_DICT = "6x6_1000"
    _C.MODEL.MARKER_GENERATOR.BORDER_BITS = 0
    _C.MODEL.MARKER_GENERATOR.NUM_GENERATION_BITS = 36
    _C.MODEL.MARKER_GENERATOR.INIT_STD = 1.4
    _C.MODEL.MARKER_GENERATOR.INIT_METHOD = "uniform"
    _C.MODEL.MARKER_GENERATOR.CONV_DIMS = [[8],[6]]
    _C.MODEL.MARKER_GENERATOR.FC_DIMS = [256,256]
    _C.MODEL.MARKER_GENERATOR.UPSAMPLE_TYPE = "bilinear"
    _C.MODEL.MARKER_GENERATOR.UPSAMPLE_SCALE = 2
    _C.MODEL.MARKER_GENERATOR.INITIAL_SIZE = 4
    _C.MODEL.MARKER_GENERATOR.NORM_TYPE = "adain"
    _C.MODEL.MARKER_GENERATOR.ACTIVATION_TYPE = "leaky"
    _C.MODEL.MARKER_GENERATOR.RESIDUAL = False
    _C.MODEL.MARKER_GENERATOR.EQUALIZED = False
    _C.MODEL.MARKER_GENERATOR.PADDING_MODE = "zeros"
    _C.MODEL.MARKER_GENERATOR.FINAL_CONV_KERNEL_SIZE = 3
    _C.MODEL.MARKER_GENERATOR.MARKERS_FILE_LOCATION = "data/e2e_markers.npz"


def add_intermediate_augmentor_config(cfg: CN):
    _C = cfg
    _C.INTERMEDIATE_AUGMENTOR = CN()
    # _C.INTERMEDIATE_AUGMENTOR.AUG_LIST = [
    # "PerspectiveAugmentor", "RadialDistortionAugmentor", "TpsTransformer", "ImageResize", 
    # "DefocusBlurAugmentor", "MotionBlurAugmentor", "HueShiftAugmentor", 
    # "BrightnessAugmentor", "NoiseAugmentor", "GammaAugmentor", "GammaCorrector", "JPEGAugmentor"]   # Make sure the correct order of augmentations
    # _C.INTERMEDIATE_AUGMENTOR.EXEC_PROBA_LIST = [0.55, 0.55, 0.55, 1.0, 0.4, 0.4, 0.4, 0.4, 0.45, 0.4, 1.0, 0.4]   # Make sure the correct order of augmentations
    # _C.INTERMEDIATE_AUGMENTOR.TEST_STRENGTH_LIST = [0.6, 0.6, 0.6, 1.0, 0.3, 0.3, 0.3, 0.3, 0.4, 0.1, 1.0, 0.4]
    _C.INTERMEDIATE_AUGMENTOR.AUG_LIST = ["GammaCorrector"]
    _C.INTERMEDIATE_AUGMENTOR.EXEC_PROBA_LIST = [1.0]
    _C.INTERMEDIATE_AUGMENTOR.PerspectiveAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.PerspectiveAugmentor.CORNER_SHIFT_RANGE = (0.0, 0.2, 0.12)
    _C.INTERMEDIATE_AUGMENTOR.GammaAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.GammaAugmentor.GAMMA_RANGE = (0.85, 1.15, 1.0)    # Original is ~0.75
    _C.INTERMEDIATE_AUGMENTOR.DefocusBlurAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.DefocusBlurAugmentor.BLUR_RADIUS_RANGE = (0.5, 2.0, 1.5) # Original is not continuous
    _C.INTERMEDIATE_AUGMENTOR.MotionBlurAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.MotionBlurAugmentor.BLUR_RADIUS_RANGE = (0.51, 3.0, 2.0)
    _C.INTERMEDIATE_AUGMENTOR.HueShiftAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.HueShiftAugmentor.HUE_SHIFT_RANGE = (0.0, 0.1, 0.04) # Original 0.15
    _C.INTERMEDIATE_AUGMENTOR.BrightnessAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.BrightnessAugmentor.BRIGHTNESS_RANGE = (0.2, 1.2, 0.4)
    _C.INTERMEDIATE_AUGMENTOR.NoiseAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.NoiseAugmentor.NOISE_RANGE = (0.0, 0.012, 0.05)
    _C.INTERMEDIATE_AUGMENTOR.JPEGAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.JPEGAugmentor.Y_QUALITY_RANGE = (12, 20, 15) # Andreas_prev (10,20)
    _C.INTERMEDIATE_AUGMENTOR.JPEGAugmentor.UV_QUALITY_RANGE = (5, 8, 6) # Andreas_prev (4,8)
    _C.INTERMEDIATE_AUGMENTOR.MAX_IMAGE_SIZE = (1080, 1920) # Andreas_prev (4,8)
    _C.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor = CN()
    _C.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.UNDISTORT_ITER = 20
    _C.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.FOCAL_LENGTH_RANGE = (1.4, 2.0, 1.75)
    _C.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.CENTER_SHIFT_RANGE = (0.0, 0.1, 0.06)
    _C.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.DISTORTION_RANGE = (0.0, 1.25, 0.5)
    _C.INTERMEDIATE_AUGMENTOR.TpsTransformer = CN()
    # Number of control points (vertical,horizontal). 
    # More points yields increase computations and smaller scale warping patterns
    _C.INTERMEDIATE_AUGMENTOR.TpsTransformer.CTRL_PTS_SIZE = (16, 20)
    # Maximum displacement of the control points. Should be bellow 2 / max(CTRL_PTS_HEIGHT, CTRL_PTS_WIDTH) to prevent unrealistic behaviour
    _C.INTERMEDIATE_AUGMENTOR.TpsTransformer.WARP_RANGE = (0, 0.02, 0.012)
    # coordinates location maximum error in pixel, as we iteratively optimize their location
    _C.INTERMEDIATE_AUGMENTOR.TpsTransformer.STOP_THRESHOLD = 0.05
    # Maximum number of iterations if the threshold is not reached
    _C.INTERMEDIATE_AUGMENTOR.TpsTransformer.MAX_ITER = 1000


def add_roi_head_config(cfg: CN):
    _C = cfg
    _C.MODEL.ROI_TRANSFORM_HEAD = CN()
    _C.MODEL.ROI_TRANSFORM_HEAD.NAME = "SpatialTransformerHeadV2"
    _C.MODEL.ROI_TRANSFORM_HEAD.NORM = ""
    _C.MODEL.ROI_TRANSFORM_HEAD.POOLER_SAMPLING_RATIO = 0
    _C.MODEL.ROI_TRANSFORM_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_TRANSFORM_HEAD.POOLER_RESOLUTION = 12
    _C.MODEL.ROI_TRANSFORM_HEAD.TRANSFORMER_RESOLUTION = 8
    _C.MODEL.ROI_TRANSFORM_HEAD.NUM_FC = 2
    _C.MODEL.ROI_TRANSFORM_HEAD.FC_DIM = 512
    _C.MODEL.ROI_TRANSFORM_HEAD.NUM_CONV = 0
    _C.MODEL.ROI_TRANSFORM_HEAD.CONV_DIM = 256
    _C.MODEL.ROI_TRANSFORM_HEAD.LOSS_WEIGHT = 1.0
    _C.MODEL.ROI_TRANSFORM_HEAD.FC_COMMON_DIMS = [256]
    _C.MODEL.ROI_TRANSFORM_HEAD.FC_CORNER_DIMS = [128]
    _C.MODEL.ROI_TRANSFORM_HEAD.FC_RESAMPLE_DIMS = [128]
    _C.MODEL.ROI_TRANSFORM_HEAD.AFFINE_PREDICTOR_ON = False

    _C.MODEL.ROI_CORNER_HEAD = CN()
    _C.MODEL.ROI_CORNER_HEAD.NAME = "CornerHeadV2"
    _C.MODEL.ROI_CORNER_HEAD.SMOOTH_L1_BETA = 0.0
    _C.MODEL.ROI_CORNER_HEAD.LOSS_WEIGHT = 0.1 # 1.2 for CornerHead
    _C.MODEL.ROI_CORNER_HEAD.REGRESSION_WEIGHTS = (10.0, 10.0)
    _C.MODEL.ROI_CORNER_HEAD.SAMPLE_RESOLUTION = 8
    _C.MODEL.ROI_CORNER_HEAD.CONV_DIMS = [32]
    _C.MODEL.ROI_CORNER_HEAD.FC_DIMS = [128, 64]

    _C.MODEL.ROI_DECODER_HEAD = CN()
    _C.MODEL.ROI_DECODER_HEAD.NAME = "DecoderHead"
    _C.MODEL.ROI_DECODER_HEAD.DECODER_ON = True
    _C.MODEL.ROI_DECODER_HEAD.LOSS_TYPE = "mse"
    _C.MODEL.ROI_DECODER_HEAD.CONV_DIMS = []
    _C.MODEL.ROI_DECODER_HEAD.FC_DIMS = [512, 256]
    _C.MODEL.ROI_DECODER_HEAD.DECODING_LOSS_WEIGHT = 10.0
    _C.MODEL.ROI_DECODER_HEAD.CLASS_LOSS_WEIGHT = 0.5

    _C.MODEL.PROPOSAL_GENERATOR.ADAPTIVE_LOSS = True
    
    _C.TEST.SORT_INSTANCES = True
    _C.TEST.APPLY_NMS = True
    _C.TEST.DECODING_SCORE_BY_MESSAGE_CONFIDENCE = True # Otherwise uses objectness score
    _C.TEST.MARKER_POSTPROCESSING = True
    # This option choses which scoring criteria to use for NMS. Options are:
    # "bit_similarity" uses the distance of predictions to the provided class of messages [used option in the paper]
    # "message_confidence" uses the confidence of how each bit is predicted
    # "objectness" uses the predicted objectness 
    # "mc_obj_product" uses the product of "message_confidence" and "objectness"
    # "mc_obj_bs_product" uses the product of "message_confidence", "bit_similarity" and "objectness"
    _C.TEST.NMS_SCORE_CRITERIA = "mc_obj_bs_product" 


def add_vovnet_config(cfg: CN):
    _C = cfg
    _C.MODEL.VOVNET = CN()
    _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VOVNET.NORM = "FrozenBN"
    _C.MODEL.VOVNET.OUT_CHANNELS = 256
    _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256


def add_model_other_config(cfg: CN):
    _C = cfg
    _C.MODEL.PREDICTIONS_PATH = 'datasets/evaluation/e2etags/flat/all.json'
    _C.MODEL.SAVE_RENDERED = False
    _C.MODEL.SAVE_RENDERED_DIR = 'datasets/evaluation/e2etags/rendered_outputs/'

    _C.INPUT.PREDICTOR_RESIZE = False
    _C.INPUT.FILTER_BOX_THRESHOLD = 30
    _C.INPUT.MARKER_MIN = 40
    _C.INPUT.MARKER_MAX = 190
    _C.INPUT.MAX_MARKERS_PER_IMAGE = 100
    _C.INPUT.MARKER_TEST_SIZE = 50

    _C.INPUT.SPECULAR_MAX = 1.2
    _C.INPUT.SPECULAR_TEST = 0.35
    _C.INPUT.ROUGHNESS_TEST = 0.25
    _C.INPUT.DIFFUSE_BOARD = 0.9
    _C.INPUT.PROCESS_RENDERER_PARAMS = True
    _C.INPUT.NO_GRAD_GENERATOR = False

    _C.RENDERER = CN()
    _C.RENDERER.SHADING_METHOD = "cook-torrance"
    _C.RENDERER.GAMMA = 2.2
    _C.RENDERER.EPSILON = 1e-8
    _C.RENDERER.BLUR_RANGE = (1.5, 2.0, 1.0)    # Third parameter is testing value
    _C.RENDERER.ROUGHNESS_RANGE = (0.14, 0.6, 0.25)
    _C.RENDERER.DIFFUSE_RANGE = (0.9, 1.0, 0.94)
    _C.RENDERER.NORMAL_NOISE_RANGE = (0.0, 0.015, 0.005)
    _C.RENDERER.SPECULAR_RANGE = (0.3, 1.0, 0.35)

    _C.DEMO = CN()
    _C.DEMO.DRAW_MASK = False
    _C.DEMO.DRAW_BBOX = True
    _C.DEMO.DRAW_CORNERS = True
    _C.DEMO.COLOR_REDGREEN_THRESHOLD = 0.0

def get_cfg() -> CN:
    from detectron2.config.defaults import _C
    cfg = _C.clone()
    add_marker_generator_config(cfg)
    add_intermediate_augmentor_config(cfg)
    add_roi_head_config(cfg)
    add_vovnet_config(cfg)
    add_model_other_config(cfg)
    return cfg
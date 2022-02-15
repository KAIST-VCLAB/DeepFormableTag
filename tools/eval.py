# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import logging
import os
from collections import OrderedDict
import logging
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np

import torch
import detectron2
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.engine import (
    DefaultTrainer, create_ddp_model,
    default_argument_parser, default_setup, launch)
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, DatasetMapper, MetadataCatalog
from detectron2.evaluation import (
    DatasetEvaluator, DatasetEvaluators,
    inference_on_dataset, print_csv_format)

import deepformable
import deepformable.modeling
from deepformable.evaluation import DeepformableEvaluator
from deepformable.data import register_deepformable_dataset
from deepformable.utils import get_cfg, marker_metadata_loader

class Evaluator:
    def __init__(self, cfg, verbose=True):
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        
        # Change default device if GPU is not available
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = build_model(cfg).to(torch.device(cfg.MODEL.DEVICE))
        if verbose == True: logger.info("Render model:\n{}".format(model)) 

        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR)
        self.checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        self.model = model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([DeepformableEvaluator(dataset_name, output_dir=output_folder)])
        # return RenderRCNNEvaluator(dataset_name, cfg, distributed=True, output_dir=output_folder)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, rendered=False):
        mapper = DatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def test(self, cfg, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """
        model = self.model
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            # Load messages to the marker_generator if exists
            metadata = MetadataCatalog.get(dataset_name)
            if model.marker_generator and "messages" in metadata.as_dict():
                model.marker_generator.messages = metadata.messages
            
            data_loader = self.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = self.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

def setup(args):
    # Register datasets
    dataset_dir = Path(args.dataset_test_dir)
    register_deepformable_dataset(
        "deepformable-test", {},
        str(dataset_dir / "annotations.json"),
        str(dataset_dir),
        load_markers=True)
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    evaluator = Evaluator(cfg)
    if not marker_metadata_loader(cfg, args.marker_config_file):
        print("Failed to load marker metadata")
    return evaluator.test(cfg)


if __name__ == "__main__":
    # Use --eval-only to skip training and only run evaluation
    arg_parser = default_argument_parser()
    arg_parser.add_argument(
        '--dataset-test-dir', type=str, default='/Data/Datasets/Dataset_Release/test-inpainted', help='Provide test dataset path')
    arg_parser.add_argument(
        "--marker-config-file", default="files/template_config.json", metavar="FILE",
        help="path to marker config file for the metadata")
    args = arg_parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
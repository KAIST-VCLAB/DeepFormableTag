# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
from pathlib import Path
import json

import torch
import detectron2
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, launch

import deepformable
from deepformable.engine import DeepformableTrainer
from deepformable.utils import get_cfg, marker_metadata_loader
from deepformable.data import register_deepformable_dataset


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Register datasets
    dataset_dir = Path(args.dataset_train_dir)
    register_deepformable_dataset(
        cfg.DATASETS.TRAIN[0], {},
        str(dataset_dir / "annotations.json"),
        str(dataset_dir),
        load_markers=False)

    for test_dataset in cfg.DATASETS.TEST:
        # Check if need to load markers
        load_markers = True if "load_markers" in test_dataset.lower() else False
        # Select proper dataset path
        dataset_suffix = test_dataset.split("-")[-1].lower()
        dataset_dir = args.dataset_test1_dir
        if dataset_suffix == "test2":
            dataset_dir = args.dataset_test2_dir
        elif dataset_suffix == "test3":
            dataset_dir = args.dataset_test3_dir
        # Reguster dataset
        dataset_dir = Path(dataset_dir)
        register_deepformable_dataset(
            test_dataset, {},
            str(dataset_dir / "annotations.json"),
            str(dataset_dir),
            load_markers=load_markers     # if this option is false, mapper should create marker locations
        )                         # based on board location information. For more info check the mapper
    # Load metadata
    if not marker_metadata_loader(cfg, args.marker_config_file):
        print("Failed to load marker metadata")

    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    trainer = DeepformableTrainer(cfg, verbose=True)
    trainer.resume_or_load(resume=args.resume)
    
    if args.eval_only:
        res = trainer.test(cfg)
        if comm.is_main_process():
            result_path = Path(cfg.OUTPUT_DIR) / "results.json"
            with open(result_path, 'w') as result_file:
                json.dump(res, result_file, indent=4)
        return res
    return trainer.train()


if __name__ == "__main__":
    # Use --eval-only to skip training and only run evaluation
    arg_parser = default_argument_parser()
    arg_parser.add_argument(
        '--dataset-train-dir', type=str, default='/Data/Datasets/train', help='Provide train dataset path')
    arg_parser.add_argument(
        '--dataset-test1-dir', type=str, default='/Data/Datasets/test-inpainted', help='Provide test1 dataset path')
    arg_parser.add_argument(
        '--dataset-test2-dir', type=str, default='/Data/Datasets/test-realworld/flat', help='Provide test2 dataset path')
    arg_parser.add_argument(
        '--dataset-test3-dir', type=str, default='/Data/Datasets/test-realworld/deformation', help='Provide test3 dataset path')
    arg_parser.add_argument(
        "--marker-config-file", default='/Data/Datasets/marker_config.json', metavar="FILE",
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

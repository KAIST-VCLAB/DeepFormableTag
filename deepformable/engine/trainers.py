"""
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""

import logging
import weakref
import time
from contextlib import ExitStack
import os
from collections import OrderedDict
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
from fvcore.nn.precise_bn import get_bn_modules

from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.engine import (
    SimpleTrainer, DefaultTrainer, default_writers, create_ddp_model, hooks)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.evaluation import (
    DatasetEvaluator, DatasetEvaluators,
    inference_on_dataset, print_csv_format)
from detectron2.utils.file_io import PathManager


from deepformable.utils import save_seed_info, load_seed_info
from deepformable.evaluation import DeepformableEvaluator
from deepformable.modeling import (
    build_intermediate_augmentations, IntermediateAugmentor)
from deepformable.modeling import MarkerRenderer
from deepformable.data import (
    DeepformableMapper, DetectronMapperWAnn, build_detection_train_loader)


class RenderModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Build model
        self.model = build_model(cfg)
        # Build augmentations
        self.intermediate_augmentations = nn.ModuleList(build_intermediate_augmentations(cfg))
        self.register_buffer(
            "aug_prob", torch.tensor(cfg.INTERMEDIATE_AUGMENTOR.EXEC_PROBA_LIST) + 1e-6, False)
        # Build renderer
        if cfg.RENDERER.NAME == "MarkerRenderer":
            self.renderer = MarkerRenderer(cfg)
        else:
            raise Exception("Such renderer does not exists")
        self.register_buffer("gamma", torch.tensor(cfg.RENDERER.GAMMA), False)
        
        self.convert_uint8 = not cfg.MODEL.MARKER_GENERATOR.TRAINABLE
        self.render_images = True
        self.apply_augmentations = True

    @property
    def device(self):
        return self.model.device

    @property
    def marker_generator(self):
        return self.model.marker_generator

    def carry_to_gpu(self, data):
        # Carry objects to GPU
        for d in data:
            for key, obj in d.items():
                to_op = getattr(obj, "to", None)
                if callable(to_op): d[key] = to_op(self.device)
        return data

    def render_data(self, data):
        markers_batch, marker_loss = self.marker_generator(
            [d["instances"] for d in data])
        for d, markers in zip(data, markers_batch):
            d["image"] = self.renderer(d, markers)
        return data, marker_loss

    def augment_data(self, data):
        for d in data:
            probabilities = torch.rand(self.aug_prob.shape, device=self.device)
            indexes = (probabilities < self.aug_prob).nonzero(as_tuple=True)[0].tolist()
            if self.training:
                selected_augmentations = [self.intermediate_augmentations[i] for i in indexes]
            else:
                selected_augmentations = self.intermediate_augmentations
            for aug in selected_augmentations:
                d["image"], d["instances"] = aug(d["image"], d["instances"])
            d["instances"] = IntermediateAugmentor.fix_instances(d["instances"])
        return data
    
    def forward(self, data):
        data = self.carry_to_gpu(data)
        # linearize images if render or augmentations enabled
        if self.render_images or self.apply_augmentations:
            for d in data:
                d["image"] = (d["image"] / 255.0) ** self.gamma

        if self.render_images:
            data, marker_loss = self.render_data(data)

        if self.apply_augmentations:
            data = self.augment_data(data)

        # Multiply by 255.0 to scale back to network input
        if self.render_images or self.apply_augmentations:
            for d in data:
                image = d["image"]
                # Convert gamma back
                if not self.apply_augmentations:
                    image = torch.clamp((F.relu(image) + 1e-8) ** (1.0/self.gamma), 0, 1)
                image = image * 255.0
                # Rounding operation replicates uint8 conversion to better simulate quantization errors
                if not self.training or self.convert_uint8:
                    image = torch.floor(image)
                d["image"] = image

        output = self.model(data)
        if self.training:
            if isinstance(marker_loss, dict):
                output.update(marker_loss)
        else:
            # Include gt_instances
            for d, o in zip(data, output):
                if "instances" in d:
                    o["gt_instances"] = d["instances"]
                    # o["image"] = d["image"]

        return output

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

class DeepformableTrainer(SimpleTrainer):
    def __init__(self, cfg, verbose=True):
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = RenderModel(cfg).to(torch.device(cfg.MODEL.DEVICE))
        if verbose == True: logger.info("Render model:\n{}".format(model)) 

        optimizer = build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        
        # Setup mixed precision training if enabled
        self.grad_scaler = None
        if cfg.SOLVER.AMP.ENABLED:
            unsupported = "AMPTrainer does not support single-process multi-device training!"
            if isinstance(model, DistributedDataParallel):
                assert not (model.device_ids and len(model.device_ids) > 1), unsupported
            assert not isinstance(self.model, DataParallel), unsupported
            self.grad_scaler = GradScaler()

        # Initialize SimpleTrainer!
        super().__init__(model, data_loader, optimizer)
        
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.load_messages = cfg.TEST.LOAD_MESSAGES

        self.register_hooks(self.build_hooks())

    def train(self):
        super().train(self.start_iter, self.max_iter)
    
    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        
        data = next(self._data_loader_iter)
        
        data_time = time.perf_counter() - start
        
        with ExitStack() as stack:
            if self.grad_scaler:
                stack.enter_context(autocast())
            
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
        
        self.optimizer.zero_grad()
        if self.grad_scaler:
            self.grad_scaler.scale(losses).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)
    
    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
    
    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DeepformableMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            PathManager.mkdirs(output_folder)
        return DatasetEvaluators([DeepformableEvaluator(dataset_name, output_dir=output_folder)])
        # return RenderRCNNEvaluator(dataset_name, cfg, distributed=True, output_dir=output_folder)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, rendered=False):
        if rendered:
            mapper = DeepformableMapper(cfg, False)
        else:
            mapper = DetectronMapperWAnn(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def test(self, cfg):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """
        # Run testing with deterministic behaviour
        # torch.use_deterministic_algorithms(True)
        seed_info = save_seed_info() # Save current seed info for upcoming random generations
        seed_all_rng(0) # Set testing seeds to zero

        logger = logging.getLogger(__name__)

        model_instance = self.model if not isinstance(self.model, DistributedDataParallel) else self.model.module

        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            # Set behaviour for rendering and augmentations during testing
            render_images = True if "rendered" in dataset_name.lower() else False
            apply_augmentations = True if "aug" in dataset_name.lower() else False
            model_instance.render_images = render_images
            model_instance.apply_augmentations = apply_augmentations

            # Load messages to the marker_generator if exists for consistancy
            metadata = MetadataCatalog.get(dataset_name)
            if self.load_messages and model_instance.marker_generator and "messages" in metadata.as_dict():
                model_instance.marker_generator.messages = metadata.messages
            
            data_loader = self.build_test_loader(cfg, dataset_name, render_images)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            
            evaluator = self.build_evaluator(cfg, dataset_name)
            
            results_i = inference_on_dataset(self.model, data_loader, evaluator)
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
        
        # Revert to the training behaviour
        # torch.use_deterministic_algorithms(False)
        load_seed_info(seed_info)
        model_instance.render_images = True
        model_instance.apply_augmentations = True

        return results
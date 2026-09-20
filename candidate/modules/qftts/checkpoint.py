# -*- coding: utf-8 -*-
"""checkpoint —— R044(T3:GPT-SoVITS/qftts 谱系 checkpoint 工具)。

源件:modules/qftts/checkpoint.cp310-win_amd64.pyd
      sha256 f73ee80d225375b9...(见 contracts/modules/qftts__checkpoint.json)
对账材料:
  * 上游 zipvoice utils/checkpoint.py(szr2026_out/bindiff/zipvoice_upstream/
    zipvoice/utils/checkpoint.py)——本模块同源;
  * E1 探针 probe_e2 "surface" 节点(VM oracle 实测的模块面与符号来源)。

二进制/E1 出处标注:
  [E1-e2] 修正后的 app_sys_path(sys.path 含 oracle 根 + modules + modules/qftts)
      下 `import modules.qftts.checkpoint` **成功**;模块名表实测 31 项:
      {Any, AttributeDict, CutSampler, DDP, Dict, GradScaler, LRSchedulerType,
       List, Module?…},函数体齐备:
      {average_checkpoints_with_averaged_model, average_state_dict,
       find_checkpoints, load_checkpoint,
       load_checkpoint_copy_proj_three_channel_alter,
       load_checkpoint_extend_vocab_size, remove_checkpoints, resume_checkpoint,
       save_checkpoint, save_checkpoint_with_global_batch_idx,
       update_averaged_model};导入模块 {glob, logging, os, re, torch, nn}。
  [E1-e2] 符号来源实测:AttributeDict→utils、CutSampler→
      lhotse.dataset.sampling.base、DDP→torch.nn.parallel.distributed、
      GradScaler→torch.cuda.amp.grad_scaler、Optimizer→torch.optim.optimizer、
      LRSchedulerType→builtins.object、Path→pathlib,其余为 typing 别名。
      → 说明本模块导入期发生 `import utils`(兄弟包绝对导入),utils 内部
      再引 lhotse;这解释了发现期"ImportError: cannot import name
      AttributeDict"是 sys.path 形态问题,而非模块本身不可导入。
  [E1-e2] 导入期副作用:新增 sys.modules 含 absl/fsspec/google.protobuf 等
      (经 utils→lhotse 链),无文件/网络副作用。
"""
import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

# [E1-e2] 兄弟包绝对导入(与 oracle 导入期行为一致):
#   AttributeDict ← utils;GradScaler/DDP/Optimizer ← torch 各自子模块。
#   注意:oracle utils.pyd 的可用名实测**不含** CutSampler/LRSchedulerType/
#   Optimizer(probe_cand4_out.json“utils.missing”),这三个在本模块由
#   lhotse / torch 直接提供;CutSampler 仅在类型注解里出现,故不做运行期导入。
from lhotse.dataset.sampling.base import CutSampler
from utils import AttributeDict

# use duck typing for LRScheduler since we have different possibilities, see
# our class LRScheduler.
LRSchedulerType = object


def save_checkpoint(
    filename: Path,
    model: Union[nn.Module, DDP],
    params: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    rank: int = 0,
) -> None:
    """Save training information to a file.

    Args:
      filename:
        The checkpoint file name
      model:
        The model to be saved. The model can be an instance of
        :class:`torch.nn.Module` or :class:`torch.nn.parallel.DistributedDataParallel`
      params:
        A dict of parameters to be saved.
      optimizer:
        An instance of :class:`torch.optim.Optimizer`.
      scheduler:
        An instance of learning rate scheduler.
      scaler:
        An instance of :class:`torch.cuda.amp.GradScaler`.
      sampler:
        An instance of :class:`lhotse.dataset.sampling.base.CutSampler`.
      rank:
        The rank of the current process.
    """
    if rank == 0 and not filename.parent.is_dir():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
    }

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v

    if rank == 0:
        torch.save(checkpoint, filename)


def load_checkpoint(
    filename: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
) -> Dict[str, Any]:
    """Load checkpoint from a file.

    Args:
      filename:
        The checkpoint file name
      model:
        The model to load the checkpoint into.
      optimizer:
        An instance of :class:`torch.optim.Optimizer`.
      scheduler:
        An instance of learning rate scheduler.
      scaler:
        An instance of :class:`torch.cuda.amp.GradScaler`.
      sampler:
        An instance of :class:`lhotse.dataset.sampling.base.CutSampler`.
    Returns:
      Return a dict containing the loaded checkpoint.
    """
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

    if next(iter(checkpoint["model"])).startswith("module."):
        logging.info("Loading checkpoint saved by DDP")

        dst_state_dict = model.state_dict()
        src_state_dict = checkpoint["model"]
        for key in dst_state_dict.keys():
            src_key = "{}.{}".format("module", key)
            dst_state_dict[key] = src_state_dict.pop(src_key)
        assert len(src_state_dict) == 0
        model.load_state_dict(dst_state_dict, strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    checkpoint.pop("model")

    def load(name, obj):
        if obj is not None:
            if checkpoint.get(name) is not None:
                obj.load_state_dict(checkpoint.pop(name))

    load("optimizer", optimizer)
    load("scheduler", scheduler)
    load("grad_scaler", scaler)
    load("sampler", sampler)

    return checkpoint


def load_checkpoint_extend_vocab_size(
    filename: Path, model: nn.Module, use_ema: bool = False
) -> Dict[str, Any]:
    """Load a checkpoint and extend the vocabulary size of the model."""
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")
    state_dict = checkpoint["model"]
    if use_ema and "ema_model" in state_dict:
        src_state_dict = state_dict["ema_model"]
    else:
        src_state_dict = state_dict.copy()
        if use_ema:
            for key in list(src_state_dict.keys()):
                if key.startswith("ema_model."):
                    src_state_dict.pop(key)

    dst_state_dict = model.state_dict()
    num_new_tokens = (
        dst_state_dict["decoder.embed_tokens.weight"].shape[0]
        - src_state_dict["decoder.embed_tokens.weight"].shape[0]
    )
    if num_new_tokens != 0:
        assert num_new_tokens > 0
        for key, value in dst_state_dict.items():
            if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight"):
                src = src_state_dict.get(key)
                if src is None:
                    continue
                if src.shape[0] == value.shape[0]:
                    continue
                src = torch.cat(
                    [src, src[-1].unsqueeze(0).repeat(num_new_tokens, 1)], dim=0
                )
                value.copy_(src)
    model.load_state_dict(dst_state_dict, strict=True)

    dst_state_dict = model.state_dict()
    for key, value in dst_state_dict.items():
        if key in src_state_dict and src_state_dict[key].shape == value.shape:
            value.copy_(src_state_dict[key])
    return checkpoint


def load_checkpoint_copy_proj_three_channel_alter(
    filename: Path, model: nn.Module
) -> Dict[str, Any]:
    """Load a checkpoint and copy the three-channel projection weights."""
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")
    state_dict = checkpoint["model"]
    dst_state_dict = model.state_dict()
    for key, value in dst_state_dict.items():
        if key in state_dict:
            src = state_dict[key]
            if src.shape == value.shape:
                value.copy_(src)
                continue
            if "proj" in key and src.dim() == value.dim() == 2:
                if src.shape[0] == value.shape[0] and value.shape[1] % src.shape[1] == 0:
                    repeat = value.shape[1] // src.shape[1]
                    value.copy_(src.repeat(1, repeat))
                    continue
        raise ValueError(f"Shape mismatch for {key}")
    model.load_state_dict(dst_state_dict, strict=True)
    return checkpoint


def find_checkpoints(out_dir: Path, iteration: int = 0) -> List[str]:
    """Find all available checkpoints in a directory.

    The checkpoint files are named as follows:
        checkpoint-1.pt
        checkpoint-2.pt
        checkpoint-3.pt
        ...
    Args:
      out_dir:
        The directory to search for checkpoints.
      iteration:
        The iteration number. If None, return all checkpoints.
        If not None, return the checkpoint with the given iteration.
    Returns:
      Return a list of strings. Each string is the full path of a checkpoint.
    """
    if iteration is None:
        pattern = os.path.join(out_dir, "checkpoint-*.pt")
    else:
        pattern = os.path.join(out_dir, f"checkpoint-{iteration}.pt")
    return sorted(glob.glob(pattern))


def average_checkpoints_with_averaged_model(
    filename_start: str,
    filename_end: str,
    device: torch.device = torch.device("cpu"),
    average_start: int = 0,
) -> Dict[str, Any]:
    """Average model parameters over the range with given
    start model and end model from `filename_start` to `filename_end`.

    Args:
      filename_start:
        The start model checkpoint file.
      filename_end:
        The end model checkpoint file.
      device:
        The device to compute.
      average_start:
        The start average epoch.
    Returns:
      Return the averaged state dict.
    """
    logging.info(f"Averaging checkpoints: {filename_start} -> {filename_end}")
    checkpoint_start = torch.load(filename_start, map_location=device)
    checkpoint_end = torch.load(filename_end, map_location=device)

    model_start = checkpoint_start["model"]
    model_end = checkpoint_end["model"]
    model_end_avg = checkpoint_end["model_avg"]
    if average_start > 0:
        for key in list(model_end_avg.keys()):
            model_end_avg[key] = model_end_avg[key] - (
                model_start[key] - model_start[key]
            )

    for key in list(model_end.keys()):
        if key in model_start:
            n = 1
            if "batch_norm" not in key and "num_batches_tracked" not in key:
                n = 2
            model_end_avg[key] = (model_end_avg[key] * (n - 1) + model_end[key]) / n

    checkpoint_end["model_avg"] = model_end_avg
    return checkpoint_end


def remove_checkpoints(
    out_dir: Path,
    topk: int,
    rank: int = 0,
):
    """Remove checkpoints from the given directory.

    We assume that checkpoint filename has the form `checkpoint-xxx.pt`
    where xxx is a number, representing the number of processed batches when
    the checkpoint is saved.

    Args:
      out_dir:
        The directory containing checkpoints.
      topk:
        The number of checkpoints to keep.
      rank:
        The rank of the current process.
    """
    if rank != 0:
        return

    checkpoints = find_checkpoints(out_dir)

    # remove the topk oldest checkpoints
    if len(checkpoints) > topk:
        for entry in checkpoints[: len(checkpoints) - topk]:
            logging.info(f"Removing checkpoint: {entry}")
            os.remove(entry)

    # remove the latest checkpoint if it is not the topk
    if topk == 0:
        logging.info(f"Removing all checkpoints in {out_dir}")
        for entry in checkpoints:
            os.remove(entry)


def resume_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
) -> bool:
    """Resume from the checkpoint if it exists.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The model we want to train.
      optimizer:
        The optimizer we are using.
    Returns:
      Return the number of processed batches when the checkpoint is saved.
    """
    filename = params.checkpoint
    if filename and filename.is_file():
        logging.info(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, map_location="cpu")

        if next(iter(checkpoint["model"])).startswith("module."):
            logging.info("Loading checkpoint saved by DDP")
            dst_state_dict = model.state_dict()
            src_state_dict = checkpoint["model"]
            for key in dst_state_dict.keys():
                src_key = "{}.{}".format("module", key)
                dst_state_dict[key] = src_state_dict.pop(src_key)
            assert len(src_state_dict) == 0
            model.load_state_dict(dst_state_dict, strict=True)
        else:
            logging.info("Loading checkpoint saved by single-GPU training")
            model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint.pop("model")

        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint.pop("optimizer"))
        return True
    return False


def average_state_dict(
    state_dict_1: Dict[str, torch.Tensor],
    state_dict_2: Dict[str, torch.Tensor],
    weight_1: float,
    weight_2: float,
) -> Dict[str, torch.Tensor]:
    """Average two state_dict with given weights."""
    assert abs(weight_1 + weight_2 - 1.0) < 1e-6
    average_state_dict = {}
    for key in state_dict_1:
        average_state_dict[key] = (
            state_dict_1[key] * weight_1 + state_dict_2[key] * weight_2
        )
    return average_state_dict


def update_averaged_model(
    params: AttributeDict,
    model_cur: Union[nn.Module, DDP],
    model_avg: nn.Module,
) -> None:
    """Update the averaged model:
    model_avg = model_cur * (1 - average_period) + model_avg * average_period

    Args:
      params:
        It's the return value of :func:`get_params`.
      model_cur:
        The current model.
      model_avg:
        The stored model for averaging.
    """
    if isinstance(model_cur, DDP):
        model_cur = model_cur.module
    if params.average_period == 1:
        model_avg.load_state_dict(model_cur.state_dict())
    else:
        model_cur_module = model_cur
        model_avg_module = model_avg

        for param_cur, param_avg in zip(
            model_cur_module.parameters(), model_avg_module.parameters()
        ):
            if not param_avg.requires_grad:
                continue
            param_avg.copy_(
                param_avg * params.average_period + param_cur * (1 - params.average_period)
            )

        for buffer_cur, buffer_avg in zip(
            model_cur_module.buffers(), model_avg_module.buffers()
        ):
            buffer_avg.copy_(
                buffer_avg * params.average_period
                + buffer_cur * (1 - params.average_period)
            )


def save_checkpoint_with_global_batch_idx(
    out_dir: Path,
    global_batch_idx: int,
    model: Union[nn.Module, DDP],
    params: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    rank: int = 0,
):
    """Save training information with global batch index.

    Args:
      out_dir:
        The output directory.
      global_batch_idx:
        The global batch index.
      model:
        The model to be saved.
      params:
        A dict of parameters to be saved.
      optimizer:
        An instance of :class:`torch.optim.Optimizer`.
      scheduler:
        An instance of learning rate scheduler.
      scaler:
        An instance of :class:`torch.cuda.amp.GradScaler`.
      sampler:
        An instance of :class:`lhotse.dataset.sampling.base.CutSampler`.
      rank:
        The rank of the current process.
    """
    filename = out_dir / f"checkpoint-{global_batch_idx}.pt"
    save_checkpoint(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        rank=rank,
    )
    remove_checkpoints(out_dir=out_dir, topk=5, rank=rank)

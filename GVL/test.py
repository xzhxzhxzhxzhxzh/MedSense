# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
import time
import torch
import os
import sys
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import dirname, abspath
import albumentations as A
from albumentations.pytorch import ToTensorV2

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, "densevid_eval3"))
sys.path.insert(0, os.path.join(pdvc_dir, "densevid_eval3/SODA"))

torch.multiprocessing.set_sharing_strategy("file_system")

from eval_utils import evaluate
import opts
from tensorboardX import SummaryWriter
from pdvc.pdvc import build
from misc.utils import (
    print_alert_message,
    build_floder,
    create_logger,
    backup_envir,
    print_opt,
    set_seed,
)
from video_dataset import CustomDataset, collate_fn
from pdvc.pdvc import build
from collections import OrderedDict, defaultdict
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


def build_scheduler(opt, optimizer, total_steps):
    if opt.learning_strategy == "warmup_linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.warm_up_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif opt.learning_strategy == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.warm_up_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif opt.learning_strategy == "multi_step":
        milestone = [
            opt.learning_rate_decay_start + opt.learning_rate_decay_every * _
            for _ in range(
                int(
                    (opt.epoch - opt.learning_rate_decay_start)
                    / opt.learning_rate_decay_every
                )
            )
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestone, gamma=opt.learning_rate_decay_rate
        )
    else:
        raise NotImplementedError()
    return scheduler


def build_text_encoder_scheduler(opt, optimizer, total_steps):
    if opt.text_encoder_learning_strategy == "warmup_linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.text_encoder_warm_up_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif opt.text_encoder_learning_strategy == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.text_encoder_warm_up_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif opt.text_encoder_learning_strategy == "multi_step":
        milestone = [
            opt.text_encoder_lr_decay_start + opt.text_encoder_lr_decay_every * _
            for _ in range(
                int(
                    (opt.epoch - opt.text_encoder_lr_decay_start)
                    / opt.text_encoder_lr_decay_every
                )
            )
        ]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestone, gamma=opt.text_encoder_lr_decay_rate
        )
    else:
        raise AssertionError("Undefined text encoder scheduler type")
    return scheduler


def update_task_best_score_details(task_name, task_details, eval_score):
    if task_name == "dvc":
        # meteor_result = {f"METEOR_{k}": v for k, v in eval_score["METEOR"].items()}
        reca_result = {f"Recall_{k}": v for k, v in eval_score["Recall"].items()}
        prec_result = {f"Precision_{k}": v for k, v in eval_score["Precision"].items()}
        # task_details.update(meteor_result)
        task_details.update(reca_result)
        task_details.update(prec_result)
    elif task_name == "pc":
        task_details["para_METEOR"] = np.array(eval_score["para_METEOR"]).mean()
        task_details["para_CIDEr"] = np.array(eval_score["para_CIDEr"]).mean()
        task_details["para_Bleu_4"] = np.array(eval_score["para_Bleu_4"]).mean()
    elif task_name == "grounding":
        task_details["grounding_R@1IOU0.3"] = np.array(
            eval_score["grounding_R@1IOU0.3"]
        ).mean()
        task_details["grounding_R@1IOU0.5"] = np.array(
            eval_score["grounding_R@1IOU0.5"]
        ).mean()
        task_details["grounding_R@1IOU0.1"] = np.array(
            eval_score["grounding_R@1IOU0.1"]
        ).mean()
    else:
        raise AssertionError("Undefined task")


def remove_weight_by_prefix(checkpoint_model, prefix, logger):
    delete = []
    for key in checkpoint_model.keys():
        if key.startswith(prefix):
            delete.append(key)
    for key in delete:
        if logger is not None:
            logger.info("Removing key {} from pretrained checkpoint".format(key))
        del checkpoint_model[key]
    return checkpoint_model


def load_pretrained_model(model, opt, logger):
    # Load the pre-trained model
    if opt.pretrain:
        logger.info("Load pre-trained parameters from {}".format(opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path, map_location="cpu")
        checkpoint_model = model_pth["model"]
        model.load_state_dict(checkpoint_model, strict=False)
    return model


def test(opt):
    # initialize environment
    set_seed(opt.seed)
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, "train.log")

    if not opt.start_from:
        backup_envir(save_folder)
        logger.info("backup evironment completed !")

    saved_info = {"best": {}, "last": {}, "history": {}}

    # Initialize transforms
    transform = A.Compose(
        [
            A.Normalize(
                normalization="min_max_per_channel",
                always_apply=True,
            ),
        ]
    )

    # Prepare Dataset
    val_dataset = CustomDataset(
        ann_file=opt.val_ann,
        feature_dir=opt.val_feat,
        vocabulary_dict=opt.dict_file,
        opt=opt,
        pooling=opt.pooling,
        transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=opt.nthreads,
        persistent_workers=opt.nthreads > 0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    epoch = saved_info[opt.start_from_mode].get("epoch", 0)
    iteration = saved_info[opt.start_from_mode].get("iter", 0)
    best_dvc_score = saved_info[opt.start_from_mode].get("best_dvc_score", -1e5)
    best_grounding_score = saved_info[opt.start_from_mode].get(
        "best_grounding_score", -1e5
    )
    train_loss_hist = saved_info["history"].get("train_loss_hist", {})
    eval_metric_hist = saved_info["history"].get("eval_metric_hist", {})
    eval_loss_hist = saved_info["history"].get("eval_loss_hist", {})
    lr_hist = saved_info["history"].get("lr_hist", {})
    opt.current_lr = vars(opt).get("current_lr", opt.lr)
    print_opt(opt, None, logger)

    best_grounding_details = {}
    best_dvc_details = {}

    # Build model
    model, criterion, constrastive_criterion, postprocessors = build(opt)
    model.translator = val_dataset.translator
    model = load_pretrained_model(model, opt, logger)

    if opt.transfer_learning_stage1:
        other_params = model.base_encoder.parameters()
        for k, p in model.named_parameters():
            if not k.startswith("base_encoder"):
                p.requires_grad = False

        param_groups = [{"params": other_params, "lr": opt.lr}]

    elif opt.enable_contrastive:
        text_encoder_params = list(map(id, model.text_encoder.parameters()))
        other_params = filter(
            lambda p: id(p) not in text_encoder_params, model.parameters()
        )

        for _, p in model.text_encoder.named_parameters():
            p.requires_grad = False

        param_groups = [{"params": other_params, "lr": opt.lr}]

    else:
        raise NotImplementedError

    model = model.to(opt.device)

    if opt.optimizer_type == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=opt.weight_decay)

    lr_scheduler = build_scheduler(opt, optimizer, opt.epoch)
    cl_schedule_time = opt.cl_schedule_time
    cl_schedule_val = opt.cl_schedule_val
    cl_weight = 0.0

    # Load tokenizer for text encoder
    for i in range(10):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                opt.pretrained_language_model, cache_dir=opt.huggingface_cache_dir
            )
            break
        except:
            print("download error in AutoTokenizer, retry...")
            time.sleep(1)

    # print the args for debugging
    print_opt(opt, model, logger)
    print_alert_message("Strat training !", logger)

    for key, val in criterion.weight_dict.items():
        if "contrastive_loss" in key:
            criterion.weight_dict[key] = cl_weight
            criterion.matcher.cost_cl = 0 if cl_weight == 0 else opt.set_cost_cl

    weight_dict = criterion.weight_dict
    logger.info("loss type: {}".format(weight_dict.keys()))
    logger.info("loss weights: {}".format(weight_dict.values()))

    # Epoch-level iteration
    while True:
        # scheduled sampling rate update
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (
                epoch - opt.scheduled_sampling_start
            ) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(
                opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                opt.scheduled_sampling_max_prob,
            )
            model.caption_head.ss_prob = opt.ss_prob

        print("lr:{}".format(float(opt.current_lr)))

        if epoch in cl_schedule_time:
            cl_weight = cl_schedule_val[cl_schedule_time.index(epoch)]
            for key, val in weight_dict.items():
                if "contrastive_loss" in key:
                    weight_dict[key] = cl_weight
                    criterion.matcher.cost_cl = 0 if cl_weight == 0 else opt.set_cost_cl
            logger.info("Update loss weight !")
            logger.info("Loss type: {}".format(weight_dict.keys()))
            logger.info("Loss weights: {}".format(weight_dict.values()))

        # evaluation
        if True:
            model.eval()
            criterion.eval()
            result_json_path = os.path.join(
                save_folder,
                "prediction",
                f"_num_{len(val_dataset)}-ep_{epoch}.json",
            )
            eval_score, eval_loss = evaluate(
                model,
                criterion,
                constrastive_criterion,
                postprocessors,
                val_loader,
                result_json_path,
                logger=logger,
                alpha=opt.ec_alpha,
                device=opt.device,
                debug=opt.debug,
                tokenizer=tokenizer,
                dvc_eval_version=opt.eval_tool_version,
            )

            current_grounding_score = 0
            current_dvc_score = 2 * eval_score["Recall"]["mean"] * eval_score["Precision"]["mean"] / (eval_score["Recall"]["mean"] + eval_score["Precision"]["mean"] + 1e-6)

            find_best = False
            if opt.only_ft_class_head:
                raise NotImplementedError
            elif opt.criteria_for_best_ckpt == "val_loss":
                raise NotImplementedError
            elif opt.criteria_for_best_ckpt == "dvc":
                current_score = current_dvc_score
                if current_dvc_score > best_dvc_score:
                    update_task_best_score_details("dvc", best_dvc_details, eval_score)
                    best_dvc_score = current_dvc_score
                    find_best = True
            elif opt.criteria_for_best_ckpt == "grounding":
                current_score = current_grounding_score
                if current_grounding_score > best_grounding_score:
                    update_task_best_score_details(
                        "grounding", best_grounding_details, eval_score
                    )
                    best_grounding_score = current_grounding_score
                    find_best = True
            elif opt.caption_decoder_type == "none":
                raise NotImplementedError
            else:
                raise NotImplementedError

            print_info = "\n".join(
                [key + ":" + str(eval_score[key]) for key in eval_score.keys()]
            )
            logger.info(
                f"\nValidation results of ep {epoch}:\n{print_info}"
            )
            logger.info(
                f"\nOverall score for task '{opt.criteria_for_best_ckpt}'of ep {epoch}: {current_score}\n"
            )

            if find_best:
                saved_info["best"] = {
                    "opt": vars(opt),
                    "epoch": epoch,
                    "task": opt.criteria_for_best_ckpt,
                    "result_json_path": result_json_path,
                    "eval_score": eval_score,
                }
                if opt.only_ft_class_head:
                    raise NotImplementedError
                elif opt.criteria_for_best_ckpt == "val_loss":
                    raise NotImplementedError
                elif opt.criteria_for_best_ckpt == "dvc":
                    saved_info["best"].update({"best_dvc_score": best_dvc_score})
                    saved_info["best"].update({"details": best_dvc_details})
                elif opt.criteria_for_best_ckpt == "grounding":
                    saved_info["best"].update({"best_grounding_score": best_grounding_score})
                    saved_info["best"].update({"details": best_grounding_details})
                elif opt.caption_decoder_type == "none":
                    raise NotImplementedError
                else:
                    raise NotImplementedError

            eval_metric_hist[epoch] = {k: v for k, v in eval_score.items()}
            eval_loss_hist[epoch] = eval_loss
            lr_hist[epoch] = optimizer.param_groups[0]["lr"]
            saved_info["history"] = {
                "train_loss_hist": train_loss_hist,
                "eval_metric_hist": eval_metric_hist,
                "eval_loss_hist": eval_loss_hist,
                "lr_hist": lr_hist
            }

            with open(os.path.join(save_folder, "info.json"), "w") as f:
                json.dump(saved_info, f)
            logger.info("Save info to info.json")

            model.train()
            criterion.train()

        epoch += 1
        lr_scheduler.step()
        torch.cuda.empty_cache()
        break

    return saved_info


if __name__ == "__main__":
    opt = opts.parse_opts()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # to avoid OMP problem on macos
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    test(opt)

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
    if opt.pretrain and (not opt.start_from):
        logger.info("Load pre-trained parameters from {}".format(opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path, map_location="cpu")
        # query_weight = model_pth["model"].pop("query_embed.weight")
        if opt.pretrain == "encoder":
            encoder_filter = model.get_filter_rule_for_encoder()
            encoder_pth = {
                k: v for k, v in model_pth["model"].items() if encoder_filter(k)
            }
            model.load_state_dict(encoder_pth, strict=True)
        elif opt.pretrain == "decoder":
            encoder_filter = model.get_filter_rule_for_encoder()
            decoder_pth = {
                k: v for k, v in model_pth["model"].items() if not encoder_filter(k)
            }
            model.load_state_dict(decoder_pth, strict=True)
            pass
        elif opt.pretrain == "full":
            checkpoint_model = model_pth["model"]
            model.load_state_dict(checkpoint_model, strict=False)
        elif opt.pretrain == "tacos":
            checkpoint_model = model_pth["model"]
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="base_encoder.input_proj.0.0", logger=logger
            )
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="base_encoder.input_proj.1.0", logger=logger
            )
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="caption_head.0.embed", logger=logger
            )
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="caption_head.1.embed", logger=logger
            )
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="caption_head.0.logit", logger=logger
            )
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="caption_head.1.logit", logger=logger
            )
            model.load_state_dict(checkpoint_model, strict=False)
        elif opt.pretrain == "ft_s1":
            checkpoint_model = model_pth["model"]
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="base_encoder.input_proj.0.0", logger=logger
            )
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="base_encoder.input_proj.1.0", logger=logger
            )
            checkpoint_model = remove_weight_by_prefix(
                checkpoint_model, prefix="caption_head", logger=logger
            )
            model.load_state_dict(checkpoint_model, strict=False)
        else:
            raise ValueError("wrong value of opt.pretrain")
        # model.init_query_embed_weight_from_gt_timestamps()
    return model


def train(opt):
    # initialize environment
    set_seed(opt.seed)
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, "train.log")
    tf_writer = SummaryWriter(os.path.join(save_folder, "tf_summary"))

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
    train_dataset = CustomDataset(
        ann_file=opt.train_ann,
        feature_dir=opt.train_feat,
        vocabulary_dict=opt.dict_file,
        opt=opt,
        pooling=opt.pooling,
        transform=transform,
    )
    val_dataset = CustomDataset(
        ann_file=opt.val_ann,
        feature_dir=opt.val_feat,
        vocabulary_dict=opt.dict_file,
        opt=opt,
        pooling=opt.pooling,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.nthreads,
        persistent_workers=opt.nthreads > 0,
        pin_memory=True,
        collate_fn=collate_fn,
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
    model.translator = train_dataset.translator

    # Recover the parameters
    if opt.start_from and (not opt.pretrain):
        raise NotImplementedError

    model = load_pretrained_model(model, opt, logger)

    if opt.transfer_learning_stage1:
        other_params = model.base_encoder.parameters()
        for k, p in model.named_parameters():
            if (not k.startswith("base_encoder")) and (not k.startswith("caption_head")):
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
    model.train()
    criterion.train()

    if opt.optimizer_type == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=opt.weight_decay)

    total_steps = int(opt.epoch * len(train_loader) // opt.batch_step)
    lr_scheduler = build_scheduler(opt, optimizer, total_steps)
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

    if opt.start_from:
        raise NotImplementedError

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

        # Batch-level iteration
        train_loss = defaultdict(list)
        batch_step = 0
        batch_loss = []
        for dt in tqdm(train_loader, disable=opt.disable_tqdm):
            if opt.device == "cuda":
                torch.cuda.synchronize(opt.device)
            iteration += 1

            optimizer.zero_grad()
            dt = {
                key: _.to(opt.device) if isinstance(_, torch.Tensor) else _
                for key, _ in dt.items()
            }
            dt["video_target"] = [
                {
                    key: _.to(opt.device) if isinstance(_, torch.Tensor) else _
                    for key, _ in vid_info.items()
                }
                for vid_info in dt["video_target"]
            ]

            if opt.enable_contrastive:
                captions = list()
                for video_sents in dt["cap_raw"]:
                    captions.extend(video_sents)
                text_encoder_input = tokenizer(
                    captions,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=opt.max_text_input_len,
                )
                text_encoder_input = {
                    key: _.to(opt.device) if isinstance(_, torch.Tensor) else _
                    for key, _ in text_encoder_input.items()
                }
                dt["text_encoder_input"] = text_encoder_input

            # dt = collections.defaultdict(lambda: None, dt)

            output, loss = model(dt, criterion, constrastive_criterion)

            final_loss = sum(
                loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict
            )
            batch_loss.append(final_loss)
            
            batch_step += 1
            if batch_step % opt.batch_step == 0:
                final_loss = torch.stack(batch_loss).mean()
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                batch_loss = []

                optimizer.step()
                if opt.learning_strategy != "multi_step":
                    lr_scheduler.step()

                train_loss["total_loss"].append(final_loss.item())

                if opt.device == "cuda":
                    torch.cuda.synchronize()

                losses_log_every = int(len(train_loader) / 10)
                if iteration % losses_log_every == 0:
                    torch.cuda.empty_cache()

        # evaluation
        if (epoch % opt.save_checkpoint_every == 0) and (
            epoch >= opt.min_epoch_when_save
        ):
            # Save model
            saved_pth = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            if opt.save_all_checkpoint:
                checkpoint_path = os.path.join(save_folder, f"model-ep_{epoch}.pth")
            else:
                checkpoint_path = os.path.join(save_folder, "model-last.pth")
            torch.save(saved_pth, checkpoint_path)

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

            current_dvc_score = 2 * eval_score["Recall"]["mean"] * eval_score["Precision"]["mean"] / (eval_score["Recall"]["mean"] + eval_score["Precision"]["mean"] + 1e-6)

            find_best = False
            if opt.criteria_for_best_ckpt == "dvc":
                current_score = current_dvc_score
                if current_dvc_score > best_dvc_score:
                    update_task_best_score_details("dvc", best_dvc_details, eval_score)
                    best_dvc_score = current_dvc_score
                    find_best = True
            else:
                raise NotImplementedError

            # add to tf summary
            for k, v in eval_score.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        tf_writer.add_scalar(f"eval_{k}_{kk}", np.mean(vv), epoch)
                else:
                    tf_writer.add_scalar(f"eval_{k}", np.mean(v), epoch)
            for k, v in eval_loss.items():
                tf_writer.add_scalar(f"eval_{k}", v, epoch)
            for k, v in train_loss.items():
                tf_writer.add_scalar(f"train_{k}", np.mean(v), epoch)
            tf_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            tf_writer.add_scalar("eval_f1", current_dvc_score, epoch)

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

                best_checkpoint_path = os.path.join(
                    save_folder, f"model-best-{opt.criteria_for_best_ckpt}.pth"
                )
                logger.info(
                    f"Save model at ep {epoch} to {best_checkpoint_path}."
                )
                torch.save(
                    saved_pth,
                    best_checkpoint_path,
                )

            eval_metric_hist[epoch] = {k: v for k, v in eval_score.items()}
            eval_loss_hist[epoch] = eval_loss
            train_loss_hist[epoch] = {k: np.mean(v) for k, v in train_loss.items()}
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
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= opt.epoch:
            tf_writer.close()
            break

    return saved_info


if __name__ == "__main__":
    opt = opts.parse_opts()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # to avoid OMP problem on macos
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    train(opt)

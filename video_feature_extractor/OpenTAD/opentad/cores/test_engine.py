import os
import copy
import json
import tqdm
import numpy as np
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


def eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    # model forward
    model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict)
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)


def feature_extraction(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    not_eval=False,
    output_dir=None,
):
    """Inference and Evaluation the model"""

    assert output_dir is not None, "`output_dir` must be provided"

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    # model forward
    model.eval()
    # hack implementation to save GPU memory
    model.cpu()
    model.module.backbone.cuda()

    for i, data_dict in enumerate(test_loader):

        logger.info(f"Forward Pass for Batch {i}/{len(test_loader)}...")

        accumulated_results = []
        num_a_batch = data_dict["inputs"].shape[1]
        for a_batch_idx in range(num_a_batch):
            # accumulated batch computation
            a_batch = data_dict["inputs"][:, a_batch_idx, ...]
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    a_batch = a_batch.cuda()
                    results = model.module.backbone(a_batch) # [n_clip, n_chunk, n_frame, dim]
                    results = results.cpu().numpy()
                    accumulated_results.append(results.mean(2))

        _, a_batchsize, feat_dim = accumulated_results[-1].shape
        # stack results
        accumulated_results = np.stack(accumulated_results)
        accumulated_results = accumulated_results.reshape(-1, feat_dim)
        # compute chunk-level masks
        masks = data_dict["masks"].numpy().squeeze()
        masks = masks.reshape(num_a_batch * a_batchsize, -1)
        masks = masks.all(1)
        # Get feature index
        feature_idx = data_dict["metas"][0]["frame_inds"].squeeze()
        feature_idx = feature_idx.reshape(num_a_batch * a_batchsize, -1)
        feature_idx = feature_idx.mean(1).astype(int)

        # save features
        logger.info(f"Save Features...")
        filename = os.path.basename(data_dict["metas"][0]["filename"]).rsplit(".", 1)[0]
        feat_output_dir = os.path.join(output_dir, filename + "_feat")
        create_folder(feat_output_dir)
        for chunk_feat, mask, idx in zip(accumulated_results, masks, feature_idx):
            if not mask:
                # drop features from padded frames
                continue

            file_path = os.path.join(feat_output_dir, f"feat_{idx:>05}.npy")
            np.save(file_path, np.squeeze(chunk_feat))

        logger.info(f"Save Features Over...")
        logger.info(f"Forward Pass for Batch {i}/{len(test_loader)} Over...")

def gather_ddp_results(world_size, result_dict, post_cfg):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    # do nms for sliding window, if needed
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=class_idx[int(label.item())],
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
        result_dict = tmp_result_dict
    return result_dict

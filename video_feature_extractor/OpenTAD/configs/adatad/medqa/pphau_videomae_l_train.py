_base_ = [
    "../../_base_/datasets/medqa/medqa_train.py",  # dataset config
    "../../_base_/models/actionformer.py",  # model config
]

# Identical to the dataset settings
window_size = 1024
scale_factor = 8
accumulated_batchsize = 4
chunk_num = (
    window_size * scale_factor // accumulated_batchsize // 16
)  # max. 64 chunks, since videomae takes 16 frames as input and output chunk-level features

dataset = dict(
    test=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=2),
            dict(
                type="LoadFrames", num_clips=1, method="padding", trunc_len=window_size, scale_factor=scale_factor
            ),
            dict(
                type="mmaction.DecordDecode"
            ),  # Output size (480, 640, 3) * window_size
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks"],
                meta_keys=[
                    "filename",
                    "avg_fps",
                    "duration",
                    "total_frames",
                    "sample_stride",
                    "frame_inds",
                    "masks",
                    "num_clips",
                    "clip_len",
                ],
            ), # [1, 3, 4096, 224, 224]
            dict(type="Rearrange", keys=["inputs"], ops="n c (at t) h w -> at n c t h w", at=accumulated_batchsize), # [4, 1, 3, 1024, 224, 224]
        ],
    ),
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,  # enable activation checkpointing
            total_frames=window_size * scale_factor // accumulated_batchsize,
            adapter_index=list(range(24)),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b t1 t c", t1=chunk_num), # [1, 64, 8, 1024]
            ],
            freeze_backbone=True,  # unfreeze the backbone
        ),
    ),
    projection=dict(
        in_channels=1024,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=9),
        input_pdrop=0.2,
    ),
    rpn_head=dict(
        num_classes=20,
        prior_generator=dict(
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (2, 8), (4, 16), (8, 32), (16, 64), (32, 10000)],
        ),
        loss_normalizer=250,
        loss_weight=0.5,
    ),
)

solver = dict(
    test=dict(batch_size=1, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=8e-5, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=250)

inference = dict(
    load_from_raw_predictions=False,
    save_raw_prediction=False
)
post_processing = dict(
    pre_nms_topk=5000,
    nms=dict(
        use_soft_nms=True,
        sigma=0.4,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.75,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=19,
    end_epoch=35,
)

work_dir = "outputs/medqa/train"

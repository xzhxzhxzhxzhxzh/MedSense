annotation_path = "/root/project/videoqa/inputs/annotations/val_annotations.json"
data_path = "/root/project/videoqa/inputs/val_set"

window_size = 1024
scale_factor = 8
chunk_num = (
    window_size * scale_factor // 16
)  # max. 256 chunks, since videomae takes 16 frames as input and output chunk-level features

dataset = dict(
    test=dict(
        type="CumstomPaddingDataset",
        ann_file=annotation_path,
        subset_name="val",
        class_map="",
        data_path=data_path,
        test_mode=True,
        feature_stride=16,
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=8),
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
            ),
        ],
    ),
)

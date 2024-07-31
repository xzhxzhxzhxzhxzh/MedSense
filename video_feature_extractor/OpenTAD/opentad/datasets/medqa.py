import numpy as np
import json
from .base import ResizeDataset, PaddingDataset, SlidingWindowDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class CumstomPaddingDataset(PaddingDataset):
    def get_class_map(self, class_map_path):
        return None
    
    def get_gt(self, video_info, thresh=0.01):
        return None
    
    def get_dataset(self):
        with open(self.ann_file, "r") as f:
            ann = json.load(f)
            anno_database = ann["metadata"]

        self.data_list = []
        for video_name, video_info in anno_database.items():
            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
                self.subset_name = "val"
            else:
                raise ValueError("Only available for inference!")

            self.data_list.append([video_name, video_info, video_anno])
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        results = self.pipeline(
            dict(
                video_name=video_info["video_name"],
                data_path=self.data_path,
                snippet_stride=self.snippet_stride,
                duration=video_info["duration"],
                **video_anno,
            )
        )
        return results
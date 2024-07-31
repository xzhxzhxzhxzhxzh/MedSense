from collections import defaultdict
from itertools import chain
import json
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import pandas as pd
from scipy.interpolate import interp1d
import pickle


def collate_fn(batch):
    B = len(batch)
    feat_dim = batch[0][0].shape[1]
    b_feats, b_featstamps, b_acts, b_cap_tk, b_timestamps, b_duration, b_cap, b_key = (
        zip(*batch)
    )
    b_featstamps = list(chain(*b_featstamps))

    # get statistic information
    max_feat_len = max([x.shape[0] for x in b_feats])
    max_cap_len = max(chain(*[[len(cap) for cap in cap_tk] for cap_tk in b_cap_tk]))
    total_cap_num = sum([len(captions) for captions in b_cap_tk])
    max_cap_num = max(len(captions) for captions in b_cap_tk)

    # initialize tensors
    t_feats = torch.FloatTensor(B, max_feat_len, feat_dim).zero_()
    t_feats_mask = torch.BoolTensor(B, max_feat_len).zero_()
    t_length = torch.FloatTensor(B, 3).zero_()  # [max_feat_len, duration, n_caps]
    t_cap_tk = torch.LongTensor(total_cap_num, max_cap_len).zero_()
    t_cap_mask = torch.BoolTensor(total_cap_num, max_cap_len).zero_()
    t_cap_len = torch.LongTensor(total_cap_num).zero_()
    t_cap_gather_idx = torch.LongTensor(total_cap_num).zero_()
    t_bbox = torch.FloatTensor(B, max_cap_num, 2).zero_()

    curr_cap_idx = 0
    for idx in range(B):
        feat_len = b_feats[idx].shape[0]
        cap_num = len(b_cap_tk[idx])

        # assign the value to tensors
        t_feats[idx, :feat_len, :] = torch.tensor(b_feats[idx], dtype=torch.float32)
        t_length[idx, 0] = float(feat_len)
        t_length[idx, 1] = b_duration[idx]
        t_length[idx, 2] = cap_num
        t_feats_mask[idx, :feat_len] = True
        # centerize the bbox.
        t_bbox[idx, :cap_num] = torch.tensor(
            [
                [
                    (ts[1] + ts[0]) / (2 * b_duration[idx]),
                    (ts[1] - ts[0]) / b_duration[idx],
                ]
                for ts in b_timestamps[idx]
            ],
            dtype=torch.float32,
        )

        # specify to which batch the caption belongs
        t_cap_gather_idx[curr_cap_idx : curr_cap_idx + cap_num] = idx
        # assign captions
        for iidx, cap in enumerate(b_cap_tk[idx]):
            cap_len = len(cap)
            t_cap_len[curr_cap_idx + iidx] = cap_len
            t_cap_tk[curr_cap_idx + iidx, :cap_len] = torch.tensor(
                cap, dtype=torch.long
            )
            t_cap_mask[curr_cap_idx + iidx, :cap_len] = True
        curr_cap_idx += cap_num

    # filter out useless bbox if starttime == endtime == 0.
    t_bbox_mask = (t_bbox != 0).sum(2) > 0

    target = [
        {
            "boxes": torch.tensor(
                [
                    [
                        (ts[1] + ts[0]) / (2 * b_duration[i]),
                        (ts[1] - ts[0]) / b_duration[i],
                    ]
                    for ts in b_timestamps[i]
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor(
                b_acts[i], dtype=torch.long
            ),  # 0 if not specified in ann.
            "masks": None,
            "image_id": vid,
        }
        for i, vid in enumerate(b_key)
    ]
    batch_input = {
        "video": {
            "tensor": t_feats,  # [batch_size, feat_len, feat_dim]
            "length": t_length,  # [max_feat_len, duration, n_caps]
            "mask": t_feats_mask,  # [batch_size, feat_len]
            "key": list(b_key),
            "target": target,
        },
        "gt": {
            "featstamps": b_featstamps,  # [total_cap_num, 2]
            "timestamp": list(b_timestamps),  # [n_video, n_cap, 2]
            "gather_idx": t_cap_gather_idx,  # [total_cap_num]
            "boxes": t_bbox,  # [batch_size, n_cap, 2]
            "boxes_mask": t_bbox_mask,  # [batch_size, n_cap]
        },
        "cap": {
            "tensor": t_cap_tk,  # [total_cap_num, cap_len]
            "length": t_cap_len,  # [total_cap_num]
            "mask": t_cap_mask,  # [total_cap_num, cap_len]
            "raw": list(b_cap),  # [batch_size, n_cap]
        },
    }
    # flat the multi-level dict
    batch_input = {
        k1 + "_" + k2: v2 for k1, v1 in batch_input.items() for k2, v2 in v1.items()
    }
    return batch_input


class Translator(object):
    def __init__(self, translator_json, vocob_size):
        self.vocab_size = vocob_size
        self.vocab = json.load(open(translator_json, "r"))
        assert self.vocab_size == len(self.vocab["word_to_ix"].keys())
        self.vocab["word_to_ix"] = defaultdict(
            lambda: self.vocab_size - 1, self.vocab["word_to_ix"]
        )
        self.vocab["ix_to_word"] = defaultdict(
            lambda: "<unk>", self.vocab["ix_to_word"]
        )
        print("load translator, total_vocab: %d", len(self.vocab["ix_to_word"]))

    def translate(self, sentence, max_len):
        tokens = ['!', '@', '%', '^','*', '|', '#','[',']' ,'$',',', ':', '!', '_', ';', '.', '?', '"', '\\n', '\\', '.']
        for token in tokens:
            sentence = sentence.replace(token, ' ')
        sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
        res = np.array(
            [0] + [self.vocab['word_to_ix'][word] for word in sentence_split][:max_len - 2] + [0])
        return res

    def rtranslate(self, sent_ids):
        for i in range(len(sent_ids)):
            if sent_ids[i] == 0:
                sent_ids = sent_ids[:i]
                break
        if len(sent_ids):
            return ' '.join([self.vocab['ix_to_word'][str(idx)] for idx in sent_ids]) + '.'
        else:
            return ''

class ClassMap(object):
    def __init__(self, class_path):
        with open(class_path, 'r') as f:
            content = f.readlines()
        self.name2idx = {}
        self.idx2name = {}
        for idx, name in enumerate(content):
            name = name.strip('\n')
            self.name2idx[name] = idx
            self.idx2name[idx] = name

    def convert_name2idx(self, name):
        return self.name2idx[name]
    
    def convert_idx2name(self, idx):
        return self.idx2name[idx]

    def __len__(self):
        return len(self.name2idx)


class EDVCdataset(Dataset):

    def __init__(self, anno_file, feature_folder, translator_json, is_training, proposal_type, opt):

        super(EDVCdataset, self).__init__()
        opt.only_ft_class_head = vars(opt).get('only_ft_class_head', False)
        opt.train_with_split_anno = vars(opt).get('train_with_split_anno', False)
        self.train_with_split_anno = opt.train_with_split_anno
        self.translator = Translator(translator_json, opt.vocab_size)
        self.max_caption_len = opt.max_caption_len
        self.anno_path = anno_file
        with open(self.anno_path, 'r') as f:
            self.anno = json.load(f)
        self.keys = list(self.anno.keys())

        for json_path in opt.invalid_video_json:
            invalid_videos = json.load(open(json_path))
            self.keys = [k for k in self.keys if k[:13] not in invalid_videos]
        print('load captioning file, %d captioning loaded', len(self.keys))

        self.feature_folder = feature_folder
        self.feature_sample_rate = opt.feature_sample_rate
        self.opt = opt
        self.proposal_type = proposal_type
        self.is_training = is_training
        self.train_proposal_sample_num = opt.train_proposal_sample_num
        self.gt_proposal_sample_num = opt.gt_proposal_sample_num
        self.feature_dim = self.opt.feature_dim
        self.num_queries = opt.num_queries
        if self.opt.only_ft_class_head:
            self.name_map = ClassMap(opt.action_classes_path)


    def __len__(self):
        return len(self.keys)

    def process_time_step(self, duration, timestamps_list, feature_length):
        duration = np.array(duration)
        timestamps = np.array(timestamps_list)
        feature_length = np.array(feature_length)
        featstamps = feature_length * timestamps / duration
        featstamps = np.minimum(featstamps, feature_length - 1).astype('int')
        featstamps = np.maximum(featstamps, 0).astype('int')
        return featstamps.tolist()

    def __getitem__(self, idx):
        raise NotImplementedError()


class PropSeqDataset(EDVCdataset):

    def __init__(self, anno_file, feature_folder, translator_pickle, is_training, proposal_type,

                 opt):
        super(PropSeqDataset, self).__init__(anno_file,
                                             feature_folder, translator_pickle, is_training, proposal_type,
                                             opt)

    def load_feats(self, key):
        vf_types = self.opt.visual_feature_type
        rescale_method = 'fix_length'
        if type(vf_types) == list:
            assert type(self.feature_folder) == list and len(vf_types) == len(self.feature_folder)
            feats_dict = {}
            all_padding = True
            for vf_type, vf_folder in zip(vf_types, self.feature_folder):
                feats, is_padding = get_feats(key, vf_type, vf_folder)
                all_padding = is_padding & all_padding
                feats_dict[vf_type] = feats
                if self.opt.data_rescale:
                    if rescale_method == 'fix_length':
                        rescale_len = self.opt.frame_embedding_num
                    elif rescale_method.startswith('follow'):
                        follow_type = rescale_method.split('_')[1]
                        assert follow_type in vf_types
                        rescale_len = len(feats_dict[follow_type])
                    else:
                        raise AssertionError('rescale_method must be \"fix_length\" or "follow_*"')
                    if feats.shape[0] != rescale_len:
                        feats = resizeFeature(feats, rescale_len, 'nearest')
                else:
                    feats = feats[::self.opt.feature_sample_rate]
                feats_dict[vf_type] = feats
            if all_padding:
                print('all feature files of video {} do not exist'.format(key))
            out = np.concatenate([feats_dict[type_] for type_ in vf_types], axis=-1)
        else:
            out, is_padding = get_feats(key, vf_types, self.feature_folder, data_norm=self.opt.data_norm)
            if self.opt.data_rescale:
                out = resizeFeature(out, self.opt.frame_embedding_num, 'nearest')
        assert out.shape[1] == self.feature_dim, 'wrong value of feature_dim'
        return out

    def load_anno_for_single_video(self, key):
        duration = self.anno[key]['duration']
        captions = self.anno[key]['sentences']
        gt_timestamps = self.anno[key]['timestamps']  # [gt_num, 2]
        dataset = self.anno.get('dataset', 'none')
        action_labels = self.anno.get('action_labels', [0] * len(gt_timestamps))
        return duration, captions, gt_timestamps, action_labels, dataset 

    def __getitem__(self, idx):
        key = str(self.keys[idx])
        duration, captions, gt_timestamps, action_labels, dataset = self.load_anno_for_single_video(key)
        feat_key = key[3:] if self.train_with_split_anno else key
        feats = self.load_feats(feat_key)
        if self.opt.only_ft_class_head:
            action_labels = [self.name_map.convert_name2idx(_) for _ in action_labels]
            assert max(action_labels) <= self.opt.num_classes

        gt_sample_num = len(gt_timestamps) if (
                len(gt_timestamps) < self.gt_proposal_sample_num) else self.gt_proposal_sample_num
        random_ids = np.random.choice(list(range(len(gt_timestamps))), gt_sample_num, replace=False)

        captions = [captions[_] for _ in range(len(captions)) if _ in random_ids]
        gt_timestamps = [gt_timestamps[_] for _ in range(len(gt_timestamps)) if _ in random_ids]
        action_labels = [action_labels[_] for _ in range(len(action_labels)) if _ in random_ids]

        caption_label = [np.array(self.translator.translate(sent, self.max_caption_len)) for sent in captions]
        gt_featstamps = self.process_time_step(duration, gt_timestamps, feats.shape[0])

        return feats, gt_featstamps, action_labels, caption_label, gt_timestamps, duration, captions, key


class CustomDataset(Dataset):
    def __init__(
        self,
        ann_file,
        feature_dir,
        vocabulary_dict,
        opt,
        pooling=None,
        transform=None,
    ):
        super().__init__()

        self.ann_file = ann_file
        with open(self.ann_file, "r") as f:
            ann = json.load(f)
        self.metadata = {k: v for k, v in ann["metadata"].items()}
        self.seq_list = list(self.metadata.keys())

        self.ann = ann["annotations"]
        self.feature_dir = feature_dir
        self.translator = Translator(vocabulary_dict, opt.vocab_size)
        self.opt = opt
        self.pooling = pooling
        self.transform = transform

        self.frame_embedding_num = opt.frame_embedding_num  # 200
        self.feature_sample_rate = opt.feature_sample_rate  # 1
        self.train_proposal_sample_num = opt.train_proposal_sample_num  # 24
        self.gt_proposal_sample_num = opt.gt_proposal_sample_num  # 30
        self.max_caption_len = opt.max_caption_len  # 30
        self.feature_dim = opt.feature_dim
        self.num_queries = opt.num_queries

        opt.only_ft_class_head = vars(opt).get("only_ft_class_head", False)  # false
        opt.train_with_split_anno = vars(opt).get(
            "train_with_split_anno", False
        )  # false
        if opt.only_ft_class_head:
            self.name_map = ClassMap(opt.action_classes_path)
        self.train_with_split_anno = opt.train_with_split_anno  # false

    def __len__(self):
        return len(self.metadata)

    def load_single_video_ann(self, key):
        duration = self.metadata[key]["duration"]
        captions = [random.choice(v["cap"]) for v in self.ann[key].values()]
        gt_timestamps = [v["z"] for v in self.ann[key].values()]
        action_labels = [0] * len(captions)  # useless
        return duration, captions, gt_timestamps, action_labels

    def load_feats(self, vid_name):
        feat_dir = os.path.join(self.feature_dir, f"{vid_name}_feat")
        feat_list = os.listdir(feat_dir)
        feat_list.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))

        feat_array = []
        for feat in feat_list:
            feat_array.append(np.load(os.path.join(feat_dir, feat)))
        feat_array = np.stack(feat_array)
        if self.pooling:
            feat_array = feat_array.reshape((-1, self.pooling, feat_array.shape[-1]))
            feat_array = feat_array.mean(axis=1)

        feat_array = resizeFeature(
            feat_array, self.frame_embedding_num, sample_method="cubic"
        )

        assert feat_array.shape[1] == self.feature_dim, "Wrong feature dimension."
        return feat_array

    def process_time_step(self, duration, timestamps, feature_length):
        duration = np.array(duration)
        timestamps = np.array(timestamps)
        feature_length = np.array(feature_length)
        featstamps = feature_length * timestamps / duration
        featstamps = np.clip(featstamps, 0, feature_length - 1)
        featstamps = featstamps.astype("int").tolist()
        return featstamps

    def __getitem__(self, idx):
        while True:
            try:
                key = self.seq_list[idx]
                duration, cap, timestamps, actions = self.load_single_video_ann(key)
                feats = self.load_feats(self.metadata[key]["video_name"])
                if self.transform:
                    augmented = self.transform(image=feats[None, :])
                    feats = augmented["image"].squeeze()

                if self.opt.only_ft_class_head:
                    raise NotImplementedError

                gt_sample_num = min(len(timestamps), self.gt_proposal_sample_num)
                cap = random.sample(cap, gt_sample_num)
                timestamps = random.sample(timestamps, gt_sample_num)
                actions = random.sample(actions, gt_sample_num)
                cap_tk = list(
                    map(lambda x: self.translator.translate(x, self.max_caption_len), cap)
                )
                featstamps = self.process_time_step(duration, timestamps, feats.shape[0])
                timestamps = np.array(timestamps, dtype=int).tolist()

            except FileNotFoundError:
                print(f"LOADING {key}...FAILED!")
                idx = random.randint(0, len(self.seq_list) - 1)
            else:
                break

        return feats, featstamps, actions, cap_tk, timestamps, duration, cap, key


def iou(interval_1, interval_2):
    interval_1, interval_2 = map(np.array, (interval_1, interval_2))
    start, end = interval_2[None, :, 0], interval_2[None, :, 1]
    start_i, end_i = interval_1[:, None, 0], interval_1[:, None, 1]
    intersection = np.minimum(end, end_i) - np.maximum(start, start_i)
    union = np.minimum(np.maximum(end, end_i) - np.minimum(start, start_i), end - start + end_i - start_i)
    iou = intersection.clip(0) / (union + 1e-8)
    return iou


def sort_events(proposal_data):
    for vid in proposal_data.keys():
        v_data = proposal_data[vid]
        v_data = [p for p in v_data if p['score'] > 0]
        tmp = sorted(v_data, key=lambda x: x['segment'])
        proposal_data[vid] = tmp
    return proposal_data


def read_file(path, feat_dim, MEAN=0., VAR=1., data_norm=False):
    if os.path.exists(path):
        ext = path.split('.')[-1]
        if ext == 'npy':
            feats = np.load(path)
        elif ext == 'csv':
            feats = pd.read_csv(path).values
        elif ext == 'pkl':
            with open(path, 'rb') as f:
                feats = pickle.load(f)
        else:
            raise NotImplementedError

        padding = False
    else:
        print('{} not exists, use zero padding. '.format(path))
        feats = np.zeros((100, feat_dim))
        padding = True
    if data_norm:
        feats = (feats - MEAN) / np.sqrt(VAR)
    return feats, padding


def get_feats(key, vf_type, vf_folder, data_norm=False):
    MEAN = VAR = 0
    if vf_type == 'c3d':
        feat_dim = 500
        MEAN = -0.001915027447565527
        VAR = 1.9239444588254049
        path = os.path.join(vf_folder, key[0:13] + '.npy')

    elif vf_type == 'c3d4096':
        feat_dim = 4096
        path = os.path.join(vf_folder, key + '.npy')

    elif vf_type == 'resnet':
        feat_dim = 2048
        MEAN = 0.41634243404998694
        VAR = 0.2569392081183313
        path = os.path.join(vf_folder, key[2:13] + '_resnet.npy')
    elif vf_type == 'bn':
        feat_dim = 1024
        MEAN = 0.8945046635916155
        VAR = 3.6579982046018844
        path = os.path.join(vf_folder, key[2:13] + '_bn.npy')
    elif vf_type == 'tsn_100':
        feat_dim = 400
        path = os.path.join(vf_folder, key[0:13] + '.csv')
    elif vf_type == 'i3d_rgb':
        feat_dim = 1024
        path = os.path.join(vf_folder, key[:13] + '_rgb.npy')
    elif vf_type == 'i3d_flow':
        feat_dim = 1024
        path = os.path.join(vf_folder, key[:13] + '_flow.npy')
    elif vf_type == 'tsp':
        feat_dim = 512
        path = os.path.join(vf_folder, key[0:13] + '.npy')
    elif vf_type == 'swin':
        feat_dim = 1024
        path = os.path.join(vf_folder, key[0:13] + '.npy')
    elif vf_type == 'vggish':
        feat_dim = 128
        path = os.path.join(vf_folder, key[0:13] + '.npy')
    elif vf_type == 'clip_pkl':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    elif vf_type == 'clip':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:13] + '.npy')
    else:
        raise AssertionError('feature type error: {}'.format(vf_type))

    feats, padding = read_file(path, feat_dim, MEAN, VAR, data_norm)

    if len(feats.shape) == 1:
        assert feats.shape[0] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)

    assert feats.shape[1] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)
    return feats, padding


def resizeFeature(inputData, newSize, sample_method):
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    # print originalSize
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = interp1d(x, inputData, axis=0, kind=sample_method)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new

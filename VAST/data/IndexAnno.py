
import os
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from toolz.sandbox import unzip
from torch.utils.data import Dataset
from utils.logger import LOGGER
from .vision_mapper import VisionMapper
from .audio_mapper import AudioMapper

from torch.utils.data import ConcatDataset








class AnnoIndexedDataset(Dataset):
    def __init__(self, d_cfg, args):
        self.vision_mapper = VisionMapper(d_cfg, args) if 'vision' in d_cfg else None
        self.audio_mapper = AudioMapper(d_cfg, args) if 'audio' in d_cfg else None
        self.annos = json.load(open(d_cfg['txt']))
        self.idx = list(range(len(self.annos)))
        self.dataset_name = d_cfg['name']
        self.training = d_cfg.training

        self.worker_init_fn =  None
        self.use_sampler = True
        self.collate_fn = annoindexedcollate
         
        self.annfile = getattr(d_cfg,'annfile',None)
        self.make_submission = getattr(d_cfg,'make_submission',False)
        self.multi_evaluation = getattr(d_cfg,'multi_evaluation',False)
        self.vqa_anno_file = getattr(d_cfg,'vqa_anno_file',None)
        self.vqa_question_file = getattr(d_cfg,'vqa_question_file',None)
        self.box_noise_scale = getattr(d_cfg,'box_noise_scale',0.5)
        
        
    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        anno = self.annos[i]

        for key in ['video_id','image_id','image','id']:
            if key in anno: 
                id_ = anno[key]
                break

        raw_captions = None
        raw_subtitles = None
        question_id = None
        question = None
        answer = None
        id_txt = None
        vision_pixels = None
        audio_spectrograms = None 
 


      

        
        raw_captions = anno['desc'] if 'desc' in anno else anno['caption']
        if isinstance(raw_captions, list):
            raw_captions = random.choice(raw_captions)
        num_samples = len(raw_captions) if isinstance(raw_captions, list) else 1
        id_txt = [id_] * num_samples
        timestamp = anno['timestamp']
        if self.training:
            start, end = timestamp
            half_width = (end - start) / 2

            start += random.uniform(-1, 1) * half_width * self.box_noise_scale
            end += random.uniform(-1, 1) * half_width * self.box_noise_scale
            if start < end:
                timestamp = [max(start, 0), min(end, anno['frame_count'])]

        id_, event_id = id_.split("@")


        if 'subtitle' in anno:
            raw_subtitles = anno['subtitle']

        if 'question' in anno:

            if self.training:
                question = anno['question']
                if isinstance(anno['answer'],list): #### vqav2
                    answer = random.choice(anno['answer'])
                else:
                    answer = anno['answer']

            else:
                question = anno['question']
                answer = anno['answer']
                if 'question_id' in anno:
                    question_id = anno['question_id']
                
           
        if self.vision_mapper:
            if self.vision_mapper.vision_format == 'video_feats':
                vision_feats = self.vision_mapper.read(id_)

            else:
                vision_pixels = self.vision_mapper.read(id_, timestamp)
                if vision_pixels is None: ###wrong img/video, resample when training and raise error when testing
                    if self.training: 
                        resample_idx = random.choice(self.idx)
                        LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
                        return self.__getitem__(resample_idx)
                    else:
                        resample_idx = random.choice(self.idx)
                        LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video,!!!!!!!!!!!!!!!!!!!!!!!! use {resample_idx} instead.')
                        return self.__getitem__(resample_idx)
                        # raise ValueError

        if  self.audio_mapper:   
            audio_spectrograms = self.audio_mapper.read(id_)
            if audio_spectrograms is None: ### wrong audio, resample when training and raise error when testing
                if self.training:
                    resample_idx = random.choice(self.idx)
                    LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
                    return self.__getitem__(resample_idx)
                else:
                    raise ValueError                

        id_ = f"{id_}@{event_id}"
        return id_, raw_captions, vision_pixels, id_txt, question, answer, question_id, \
        audio_spectrograms, raw_subtitles



def annoindexedcollate(inputs):
    
    batch = {}
    all_data = map(list, unzip(inputs))
    keys = ['ids', 
            'raw_captions', 
            'vision_pixels', 
            'ids_txt', 
            'raw_questions', 
            'raw_answers', 
            'question_ids', 
            'audio_spectrograms',
            'raw_subtitles']

    for key, data in zip(keys, all_data):
  
        if data[0] is None:
            continue 
        elif isinstance(data[0], torch.Tensor):
            batch[key] = torch.stack(data, dim=0).float()
      
        else:
            batch[key] = data

      

    return batch



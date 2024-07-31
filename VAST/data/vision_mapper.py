import os
import decord
import random
import torch
import h5py
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.transforms import *
from PIL import Image
from utils.logger import LOGGER
from utils.tool import split




class VisionMapper(object):
    def __init__(self, d_cfg, args):

        self.vision = d_cfg.vision
        self.name = d_cfg.name
        self.training = d_cfg.training
        self.vision_format = d_cfg.vision_format
        ### for feat extraction
        self.dense_extraction = getattr(d_cfg,'dense_extraction',False)
        self.extract_fps = getattr(d_cfg,'extract_fps',None)
        self.frame_fps = getattr(d_cfg,'frame_fps',None)


        # if self.vision_format == 'video_feats':
        #     self.num_pre_clips = args.model_cfg.NUM_PRE_CLIPS
        #     self.num_clips = args.model_cfg.NUM_CLIPS

        # else:


        # self.num_pre_clips = args.model_cfg.NUM_PRE_CLIPS
        # self.num_clips = args.model_cfg.NUM_CLIPS

        if self.vision_format.startswith('video'):        
            self.sample_num = d_cfg.vision_sample_num 
        
        
        self.resolution = args.model_cfg.vision_resolution

        if args.model_cfg.vision_encoder_type.startswith('clip') or args.model_cfg.vision_encoder_type.startswith('evaclip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
            LOGGER.info(f'{self.name} Using clip mean and std.')
        else:    
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
            LOGGER.info(f'{self.name} Using imagenet mean and std.')

        self.vision_transforms =  d_cfg.get('vision_transforms','none')
        if self.vision_transforms == 'none':
            if self.training:
                self.transforms = Compose([
                                                Resize((self.resolution,self.resolution)),
                                                Normalize(self.mean,self.std)])
            else:
                self.transforms = Compose([
                                                Resize((self.resolution,self.resolution)),
                                                Normalize(self.mean,self.std)])
        elif self.vision_transforms == 'crop_flip':
            if self.training:
                self.transforms = Compose([
                                                RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                RandomHorizontalFlip(),
                                                Normalize(self.mean,self.std)])
            else:
                self.transforms = Compose([
                                                Resize(self.resolution),
                                                CenterCrop(self.resolution),
                                                Normalize(self.mean,self.std)])
            
        else:
            raise NotImplementedError
        LOGGER.info(f'{self.name} transforms {self.vision_transforms}')
    



    def read(self, id_, timestamp=None):
      
        if self.vision_format.startswith('video'):
            if self.vision_format == 'video_feats':

                if self.vision.endswith('hdf5'):
                    with h5py.File(self.vision, 'r') as f:
                        try:
                            feat = f[id_]['c3d_features'][:]
                        except:
                            feat = f[id_][:]
                        feat = F.normalize(torch.from_numpy(feat), dim=1)
                else:
                    feat = np.load(os.path.join(self.vision,f'{id_}.npy'))
                    feat = F.normalize(torch.from_numpy(feat).float(), dim=1)
                # with h5py.File(self.vision, 'r') as f:
                #     feat = f[id_]['c3d_features'][:]
                #     feat = F.normalize(torch.from_numpy(feat), dim=1)
                num_pre_clips = self.num_pre_clips
                num_src_clips = feat.size(0)
                idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
                idxs = idxs.round().long().clamp(max=num_src_clips-1)
                # To prevent a empty selection, check the idxs
                meanfeats = []
                for i in range(num_pre_clips):
                    s, e = idxs[i], idxs[i+1]
                    if s < e:
                        meanfeats.append(feat[s:e].mean(dim=0))
                    else:
                        meanfeats.append(feat[s])
                return torch.stack(meanfeats)





            else:
                vision_pixels = []        
                sample_num = self.sample_num
            
                try:




                    if self.vision_format == 'video_rawvideo':
                        video_path = os.path.join(self.vision, str(id_))
                        if not os.path.exists(video_path):
                            video_path = video_path+'.mp4'
                        if not os.path.exists(video_path):
                            video_path = video_path.replace('.mp4','.avi')
                        if not os.path.exists(video_path):
                            video_path = video_path.replace('.avi','.webm')
                        if not os.path.exists(video_path):
                            video_path = video_path.replace('.webm','.mkv')

                        container = decord.VideoReader(video_path)
                        if timestamp:
                            frames_ids = list(range(int(timestamp[0]), int(timestamp[1])))
                        else:
                            frames_ids = list(range(len(container)))
                        if self.dense_extraction: ### for feature extraction
                            fps = container.get_avg_fps()
                            sample_num = int(len(container)* self.extract_fps / fps)
                        frames_splited = split(frames_ids, sample_num)
                        if self.training:
                            sample_idx = [random.choice(i) for i in frames_splited]
                        else:
                            sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited] 
                        frames = container.get_batch(sample_idx).asnumpy()
                        vision_pixels = torch.from_numpy(frames.transpose(0,3,1,2)/255.0)  ### nX3xHxW
                        vision_pixels = self.transforms(vision_pixels)    
         
                        return vision_pixels

                    elif self.vision_format == 'video_frame':
                        frame_path = os.path.join(self.vision, str(id_))
                        frames = os.listdir(frame_path)
                        frames.sort()   ### ['img_0001.jpg','img_0002.jpg',...]
                        sample_num = self.sample_num
                        if self.dense_extraction: ### for feature extraction
                            sample_num = int(len(frames)* self.extract_fps / self.frame_fps)
                        frames_splited = split(frames,sample_num)    
                        if self.training:
                            sample_idx = [random.choice(i) for i in frames_splited]
                        else:
                            sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]
                        for i in range(sample_num):
                            frame = Image.open(os.path.join(frame_path,sample_idx[i]))
                            frame = np.array(frame).transpose(2,0,1)/255.0
                            vision_pixels.append(frame)

                        vision_pixels = torch.from_numpy(np.stack(vision_pixels,axis=0))   ### nX3xHxW
                        vision_pixels = self.transforms(vision_pixels)        
                        return vision_pixels


                except Exception as e:
                    print(e)
                    print(id_)
                    return None



        elif self.vision_format.startswith('image'):
            
            try:
              
             
                if self.vision_format=='image_rawimage':
                    img_path = os.path.join(self.vision, id_)
                    if not os.path.exists(img_path):
                        img_path += '.jpg'
                    if not os.path.exists(img_path):
                        img_path =  img_path.replace('.jpg','.JPEG')
                    if not os.path.exists(img_path):
                        print('not have im', id_)
                        assert self.name == 'llava_v1_5_mix665k'
                        return torch.zeros(1, 3, self.resolution, self.resolution)
                img = Image.open(img_path).convert('RGB')
                img = np.array(img).transpose(2,0,1)/255.0
                img = torch.from_numpy(img) 
                img = self.transforms(img)   
                img = img.unsqueeze(0)
                return img

            except Exception as e:
                print(e)
                return None

        else:
            raise NotImplementedError()

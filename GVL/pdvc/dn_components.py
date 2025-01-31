# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
import torch.nn.functional as F
from misc.detr_utils.misc import inverse_sigmoid


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        # used to create group indice
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 10:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1) # useless
        known_labels = labels.repeat(2 * dn_number, 1).view(-1) # [gt_num * 2 * dn_num]
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1) # [gt_num * 2 * dn_num]
        known_bboxs = boxes.repeat(2 * dn_number, 1) # [gt_num * 2 * dn_num, 2]
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes) + 1 # assign the class labels to the chosen indice, 1 is neg (noisy labels)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))
        
        # num of cdn
        pad_size = int(single_pad * 2 * dn_number)

        # specify the pos/neg idx by groups
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            # instead of moving center point and width, noise is added to the left/right boundary
            known_bbox_[:, 0] = known_bboxs[:, 0] - known_bboxs[:, 1] / 2
            known_bbox_[:, 1] = known_bboxs[:, 0] + known_bboxs[:, 1] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, 0] = known_bboxs[:, 1] / 2
            diff[:, 1] = known_bboxs[:, 1] / 2

            # randomly generate +/-1 signs
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0 # pos cases in (0, 1), neg cases in (1, 2)
            rand_part *= rand_sign
            # for pos cases, boundaries movment is limited within -/+50% gt width
            # for neg cases, boundaries movment exceed -/+50% gt width
            # but those noise will be modulated by `box_noise_scale`
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, 0] = (known_bbox_[:, 0] + known_bbox_[:, 1]) / 2
            known_bbox_expand[:, 1] = known_bbox_[:, 1] - known_bbox_[:, 0]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 2).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries # cdn queries + real queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    # ! note that `input_query_label` and `input_query_bbox` only contains cdn part
    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(hs, references, dn_meta):
    if dn_meta and dn_meta["pad_size"] > 0:
        # cdn part
        cdn_hs = hs[:, :, :dn_meta["pad_size"], :]
        cdn_references = references[:, :, :dn_meta["pad_size"], :]
        # true prediction
        hs = hs[:, :, dn_meta["pad_size"]:, :]
        references = references[:, :, dn_meta["pad_size"]:, :]

        dn_meta["output"] = {
            "hs": cdn_hs,
            "references": cdn_references
        }
    return hs, references



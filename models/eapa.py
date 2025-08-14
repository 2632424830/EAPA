import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from utils import read_json
from functools import partial
from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from models.vit import VisionTransformer, interpolate_pos_embed
from models.bert import BertModel, BertConfig
from models.resnet import resnet50, resnet101
from torchvision.models import vgg16, vgg19_bn
from torchvision import models
from torch.autograd import Variable

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply


def build_vision_encoder(config, load_vision_params=False):

    num_patches = (config['image_res'] // config['patch_size']) ** 2
    if config['use_swin']:
        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False)

        if load_vision_params:
            state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

            for k in list(state_dict.keys()):
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]

    else:
        assert config['patch_size'] == 16
        vision_width = 384

        vision_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=config['patch_size'], embed_dim=384, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            local_attn_depth=4)

        if load_vision_params:
            state_dict = torch.load("data/deit_small_patch16_224-cd65a155.pth", map_location="cpu")["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed'] = pos_embed_reshaped


    if load_vision_params:
        if config['use_swin']:
            print("### Load Trans-Encoder[SWin-T]: ", flush=True)
        else:
            print("### Load Trans-Encoder[ViT]: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)

    return vision_encoder, vision_width


def build_conv_encoder(config, load_vision_params=False, ins='resnet'):
    resnet_ckpt = config['resnet_ckpt']
    finetune_conv = config['finetune_conv']
    if ins == 'resnet':
        resnet_with_last = nn.Sequential(*list(resnet50(num_classes=30).children())[:-1])
        conv_width = 2048  
    elif ins == 'vgg':
        vgg = vgg19_bn(num_classes=30)
        vgg.classifier[6] = nn.Linear(4096, 2048)
        resnet_with_last = vgg
        conv_width = 2048  
    else:
        raise ValueError

    if load_vision_params:
        print("### Load Conv-Encoder[ResNet-50]: ", flush=True)
        state_dict = torch.load(resnet_ckpt, map_location="cpu")
        if len(state_dict) < 10:
            state_dict = state_dict['model']
        if ins == 'vgg':
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        resnet_with_last.load_state_dict(state_dict, strict=False)

        for child in resnet_with_last.children():
            for param in child.parameters():
                param.requires_grad = finetune_conv
    return resnet_with_last, conv_width


def build_text_encoder(config, load_text_params=False):

    text_config = read_json(config['text_config'])
    text_width = text_config['hidden_size']

    bert_config = BertConfig.from_json_file(config['text_config'])
    text_encoder = BertModel(bert_config)

    if load_text_params:

        print("### Load Trans-Encoder[Bert-B]: ", flush=True)
        init_checkpoint = config['text_encoder'] + '/pytorch_model.bin'
        state_dict = torch.load(init_checkpoint, map_location='cpu')
        text_encoder.load_state_dict(state_dict, strict=False)

        for child in text_encoder.children():
            for param in child.parameters():
                param.requires_grad = True

    return text_encoder, text_width


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def load_pretrained_eapa(ckpt_rpath, config, is_eval=False, load_text=False):

    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2

    print("### Loading pretrained vision encoder", flush=True)

    window_size = read_json(config['vision_config'])['window_size']

    for k in list(state_dict.keys()):
        if 'relative_position_bias_table' in k:
            dst_num_pos = (2 * window_size - 1) ** 2
            state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
        elif ('relative_position_index' in k) or ('attn_mask' in k):
            del state_dict[k]

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if 'text_encoder.' in key:
                if 'bert.' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

    return state_dict


class EAPABase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=True,
                 use_contrastive_loss=False, use_affil_loss=False):
        super().__init__()
        if config['is_eapa']:
            self.embed_dim = config['embed_dim']
            self.temp = nn.Parameter(torch.ones([]) * config['temp1'])

    def load_pretrained_eapa(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained_eapa(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def get_vision_embeds(self, image):
        return F.normalize(self.vision_proj(self.vision_encoder(image))[:, 0, :])

    def get_text_embeds(self, text_ids):
        return F.normalize(self.text_proj(self.text_encoder(text_ids))[:, 0, :])

    def get_le_contr_loss(self, image_le, text_le, image_pseudo_label, text_pseudo_label, idx=None, label=None, config=None):

        image_le_all = allgather(image_le.contiguous(), torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_le_all = allgather(text_le.contiguous(), torch.distributed.get_rank(), torch.distributed.get_world_size())
        
        image_le_all = image_le_all.transpose(1, 0)
        text_le_all = text_le_all.transpose(1, 0)
        
        le_logits = (torch.bmm(image_le_all, text_le_all.transpose(1, 2))/image_le_all.size(-1)).sum(0)
        
        bsz = image_le_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(le_logits, labels)
            loss_t2i = F.cross_entropy(le_logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_le.size(0)

            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(le_logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(le_logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2


    def get_contr_loss(self, image_feat, text_feat, idx=None, label=None, config=None):

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)

            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2
    
    def get_cls_loss(self, image_pred, text_pred, text_pseudo_label, image_pseudo_label, idx=None, label=None, config=None):
        
        image_pred_all = allgather(image_pred, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_pred_all = allgather(text_pred, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_pseudo_label_all = allgather(text_pseudo_label, torch.distributed.get_rank(), torch.distributed.get_world_size())
        image_pseudo_label_all = allgather(image_pseudo_label, torch.distributed.get_rank(), torch.distributed.get_world_size())
        
        text_mask = (text_pseudo_label_all != -1).float()
        image_mask = (image_pseudo_label_all != -1).float()
        loss_image = F.binary_cross_entropy_with_logits(image_pred_all, image_pseudo_label_all, weight=image_mask, reduction='sum')
        loss_text = F.binary_cross_entropy_with_logits(text_pred_all, text_pseudo_label_all, weight=text_mask, reduction='sum')
        loss_image = loss_image / image_mask.sum()
        loss_text = loss_text / text_mask.sum()
        
        
        with torch.no_grad():
            image_pred_sigmoid = torch.sigmoid(image_pred_all)
            text_pred_sigmoid = torch.sigmoid(text_pred_all)
            
            image_pred_binary = (image_pred_sigmoid >= 0.5).float()
            text_pred_binary = (text_pred_sigmoid >= 0.5).float()
            
            correct_image = (image_pred_binary == image_pseudo_label_all).float() * image_mask
            correct_text = (text_pred_binary == text_pseudo_label_all).float() * text_mask
            
            acc_image = correct_image.sum() / image_mask.sum()
            acc_text = correct_text.sum() / text_mask.sum()

        return loss_image, loss_text, acc_image, acc_text


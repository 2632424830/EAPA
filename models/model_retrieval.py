import torch
import torch.nn as nn
from models import EAPABase, load_pretrained_eapa
import torch.nn.functional as F

import torch
# from torchinfo import summary
from PIL import Image
import open_clip
# from inference_tool import get_preprocess
from open_clip import tokenizer

from collections import OrderedDict

from models import CLSTransformer, GroupWiseLinear, PositionEmbeddingSine

# clip, _, _ = open_clip.create_model_and_transforms("ViT-B/32")
# checkpoint = torch.load(ckpt_path, map_location="cpu")
# msg = clip.load_state_dict(checkpoint, strict=False)
# print("Missing keys: ", msg.missing_keys)
# print("Unexpected keys: ", msg.unexpected_keys)


class EAPA(EAPABase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True, use_contrastive_loss=True, \
                         use_affil_loss=False)
        self.config = config
        self.use_affil_loss = config['use_affil_loss']
        self.use_triplet_loss = config['use_triplet_loss']
        self.use_cls_loss = True
        self.create_and_load_pretrained(config)
        self.align_before = False
        

        # trainable!!!
        #self.cls_head = nn.Linear(config['embed_dim'], 33)
        self.cls_transformer = CLSTransformer(
            d_model=config['embed_dim'],
            dropout=0.1,
            nhead=4,
            dim_feedforward=4*config['embed_dim'],
            num_encoder_layers=2,
            num_decoder_layers=2,
            normalize_before=False,
            return_intermediate_dec=False,
            rm_self_attn_dec=True, 
            rm_first_self_attn=True,
        )
        self.num_class = config['class_num']
        hidden_dim = self.cls_transformer.d_model
        # trainable!!!
        self.input_proj = nn.Linear(config['embed_dim'], hidden_dim)
        # trainable!!!
        self.query_embed = nn.Embedding(self.num_class, hidden_dim)
        # trainable!!!
        self.cls_fc = GroupWiseLinear(self.num_class, hidden_dim, bias=True)
        

    def create_and_load_pretrained(self, config):
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B/32")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters in self.model: {total_params:,} ({total_params/1e6:.2f}M)")
        
        config['adapter_share_layer'] = 1
        self.model.visual.transformer.adapter_prompt = nn.Embedding((config['adapter_last_layer']-config['adapter_share_layer']) * config['adapter_len'], 768)
        self.model.visual.transformer.adapter_last_layer = config['adapter_last_layer']
        self.model.visual.transformer.adapter_len = config['adapter_len']
        self.model.transformer.adapter_prompt = nn.Embedding((config['adapter_last_layer']-config['adapter_share_layer']) * config['adapter_len'], 512)
        self.model.transformer.adapter_last_layer = config['adapter_last_layer']
        self.model.transformer.adapter_len = config['adapter_len']
        
        adapter_share_prompt = nn.Embedding(config['adapter_share_layer'] * config['adapter_len'], 512)
        self.model.visual.transformer.adapter_share_prompt = adapter_share_prompt
        self.model.transformer.adapter_share_prompt = adapter_share_prompt
        self.model.visual.transformer.adapter_share_layer = config['adapter_share_layer']
        self.model.transformer.adapter_share_layer = config['adapter_share_layer']
        self.model.visual.transformer.adapter_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(512, 768*4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(768*4, 768))
        ]))
        self.model.transformer.adapter_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(512, 512*4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(512*4, 512))
        ]))
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters in self.model: {total_params:,} ({total_params/1e6:.2f}M)")

    def get_vis_emb(self, image, idx=None, label=None):
        if self.align_before:
            img_emb,img_seq_emb,feas_vis = self.model.encode_image(image,normalize=True)
            return img_emb,img_seq_emb,feas_vis
        else:
            img_emb,img_seq_emb = self.model.encode_image(image,normalize=True)
        return img_emb,img_seq_emb
        
    def get_txt_emb(self, text_ids, idx=None, label=None):
        if self.align_before:
            txt_emb,txt_seq_emb,feas_txt = self.model.encode_text(text_ids,normalize=True)
            return txt_emb,txt_seq_emb,feas_txt
        else:
            txt_emb,txt_seq_emb = self.model.encode_text(text_ids,normalize=True)
        return txt_emb,txt_seq_emb
        

    def forward(self, image, text_ids, idx=None, label=None, text_pseudo_label=None, image_pseudo_label=None):
        ## Baseline(Swin-T+Bert-B)
        if self.align_before:
            img_emb,feas_vis = self.get_vis_emb(image)
            txt_emb,feas_txt = self.get_txt_emb(text_ids)
        else:
            img_emb,img_seq_emb = self.get_vis_emb(image)
            txt_emb,txt_seq_emb=self.get_txt_emb(text_ids)

        if self.use_affil_loss:
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_affil = self.get_affil_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            return loss_contr, loss_affil
        elif self.use_triplet_loss:
            loss_triplet = self.get_triplet_loss(img_emb, txt_emb)
            return loss_triplet
        else:
            loss_before_contr = []
            if self.align_before:
                for i in range(len(feas_vis)):
                    # print("vis",feas_vis[i].shape)
                    loss_contr = self.get_contr_loss(feas_vis[i],feas_txt[i], idx=idx, label=label, config=self.config)
                    loss_before_contr.append(loss_contr)
                total_loss_before = sum(loss_before_contr)
            #loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            if self.align_before:
                return loss_contr,loss_triplet,total_loss_before
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            if self.use_cls_loss:
                #pos = self.position_embedding(img_seq_emb)
                query_input = self.query_embed.weight
                
                image_hs = self.cls_transformer(self.input_proj(img_seq_emb), query_input, None, seq_type=2)[0][-1] # B,K,d
                image_pred = self.cls_fc(image_hs)
                
                eos_pos = torch.argmax(text_ids, dim=-1)
                text_hs = self.cls_transformer(self.input_proj(txt_seq_emb), query_input, None, eos_pos=eos_pos)[0][-1] # B,K,d
                text_pred = self.cls_fc(text_hs)
                
                loss_cls_image, loss_cls_text, loss_kd, acc_image, acc_text = self.get_cls_loss(image_pred, text_pred, text_pseudo_label=text_pseudo_label, image_pseudo_label=image_pseudo_label, idx=idx, label=label, config=self.config)
                loss_led = self.get_le_contr_loss(image_hs, text_hs, image_pseudo_label=image_pseudo_label, text_pseudo_label=text_pseudo_label, idx=idx, label=label, config=self.config)
                return loss_contr,loss_cls_image,loss_cls_text,loss_kd,acc_image,acc_text,loss_led

            return loss_contr
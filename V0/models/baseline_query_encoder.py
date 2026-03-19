

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

CLIP_LOCAL_PATH = "/home/caidongao/hf_models/openai_clip_vit_large_patch14"

class BaselineQueryEncoder(nn.Module):
    def __init__(self, model_name=CLIP_LOCAL_PATH, device="cpu"):
        super().__init__()
        self.device = device

        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=True,
        )
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            use_fast=False,
            local_files_only=True,
        )


        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)


        self.embed_dim = self.model.config.projection_dim


        self.text_mlp = nn.Sequential(


            nn.Linear(self.embed_dim, self.embed_dim),


            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        ).to(self.device)


    def forward(self, images, texts):

        inputs = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            ref_emb = self.model.get_image_features(inputs["pixel_values"]).pooler_output


            text_emb = self.model.get_text_features(inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output


        ref_emb = F.normalize(ref_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)


        delta = self.text_mlp(text_emb)


        pred_emb = F.normalize(ref_emb + delta, dim=-1)

        return pred_emb

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

CLIP_LOCAL_PATH = "/home/caidongao/hf_models/openai_clip_vit_large_patch14"

class AttentionQueryEncoder(nn.Module):
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


        self.type_embed = nn.Parameter(torch.randn(2, self.embed_dim) * 0.02)


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=self.embed_dim,


            nhead=8,


            dim_feedforward=self.embed_dim * 4,

            dropout=0.1,


            batch_first=True,

            activation="gelu",
        )


        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)


        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )


        self.logit_scale = nn.Parameter(torch.tensor(2.6593))

    def get_logit_scale(self):


        return self.logit_scale.exp().clamp(max=100.0)


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


        tokens = torch.stack([ref_emb, text_emb], dim=1)


        tokens = tokens + self.type_embed.unsqueeze(0)


        fused_tokens = self.fusion(tokens)


        fused = fused_tokens[:, 0, :]


        pred_emb = self.output_proj(fused)


        pred_emb = F.normalize(pred_emb, dim=-1)

        return pred_emb

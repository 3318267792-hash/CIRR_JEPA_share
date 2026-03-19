
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

CLIP_LOCAL_PATH = "/home/caidongao/hf_models/openai_clip_vit_large_patch14"


class CLIPTargetEncoder(nn.Module):


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

    def forward(self, images):


        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )


        inputs = {k: v.to(self.device) for k, v in inputs.items()}


        with torch.no_grad():


            image_features = self.model.get_image_features(**inputs)


        return F.normalize(image_features.pooler_output, dim=-1)

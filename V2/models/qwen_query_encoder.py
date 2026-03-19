import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import AutoProcessor, Qwen3VLModel


QWEN_LOCAL_PATH = "/home/caidongao/hf_models/Qwen3-VL-2B-Instruct"


class QwenQueryEncoder(nn.Module):
    def __init__(self, model_name=QWEN_LOCAL_PATH, device="cpu"):


        super().__init__()
        self.device = device


        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )


        self.model = Qwen3VLModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )


        self.backbone = self.model


        self.backbone.eval()


        for param in self.backbone.parameters():
            param.requires_grad = False


        if hasattr(self.backbone.config, "text_config"):
            hidden_size = self.backbone.config.text_config.hidden_size
        else:
            hidden_size = self.backbone.config.hidden_size


        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 768),
        )


        self.logit_scale = nn.Parameter(torch.tensor(2.6593))


        self.to(self.device)

    def get_logit_scale(self):

        return self.logit_scale.exp().clamp(max=100.0)


    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.model.eval()
        return self

    def forward(self, images, texts):


        conversations = []

        for image, text in zip(images, texts):
            prompt = f"Given the reference image, retrieve the target image that matches this modification: {text}"
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )


        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            )


        inputs.pop("token_type_ids", None)


        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }


        with torch.no_grad():
            outputs = self.backbone(
                input_ids=inputs.get("input_ids", None),
                attention_mask=inputs.get("attention_mask", None),
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                mm_token_type_ids=inputs.get("mm_token_type_ids", None),
                use_cache=False,
                return_dict=True,
            )


        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]


        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)


        pooled = (last_hidden_state * mask).sum(dim=1)


        pooled = pooled / mask.sum(dim=1).clamp(min=1.0)


        pooled = pooled.float()


        pred_emb = self.output_proj(pooled)


        pred_emb = F.normalize(pred_emb, dim=-1)

        return pred_emb

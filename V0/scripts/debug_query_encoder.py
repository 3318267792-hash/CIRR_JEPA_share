from datasets.cirr_dataset import CIRRDataset
from models.baseline_query_encoder import BaselineQueryEncoder

import torch

transform = None

dataset = CIRRDataset(
    caption_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/captions/cap.rc2.train.json",
    split_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/image_splits/split.rc2.train.json",
    image_root="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/img_raw",
    transform=transform
)

samples = [dataset[i] for i in range(4)]

pair_ids = [sample["pair_id"] for sample in samples]
captions = [sample["caption"] for sample in samples]
ref_images = [sample["ref_img"] for sample in samples]

print("batch size =", len(samples))
print("pair_ids =", pair_ids)
print("captions =", captions)
print("ref_img type =", type(ref_images[0]))

device = "mps" if torch.backends.mps.is_available() else "cpu"

query_encoder = BaselineQueryEncoder(device=device)

pred_emb = query_encoder(ref_images, captions)

print("pred_emb shape =", pred_emb.shape)

import json
import os
from PIL import Image
from torch.utils.data import Dataset


class CIRRDataset(Dataset):
    def __init__(self, caption_json, split_json, image_root, transform=None):
        self.caption_json = caption_json
        self.split_json = split_json
        self.image_root = image_root
        self.transform = transform

        with open(caption_json, "r", encoding="utf-8") as f:
            self.caption_data = json.load(f)
        with open(split_json, "r", encoding="utf-8") as f:
            self.split_map = json.load(f)
    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, index):


        sample = self.caption_data[index]
        pair_id = sample.get("pairid")
        ref_id = sample.get("reference")
        target_id = sample.get("target_hard")
        caption = sample.get("caption")


        ref_rel_path = self.split_map.get(ref_id)


        if ref_rel_path is None:
            raise KeyError(f"reference id {ref_id} not found in split_map")

        target_rel_path = self.split_map.get(target_id)

        ref_img_path = os.path.join(self.image_root,ref_rel_path)

        target_img_path = None
        if target_rel_path is not None:
            target_img_path = os.path.join(self.image_root, target_rel_path)


        ref_img = Image.open(ref_img_path).convert("RGB")


        target_img = None
        if target_rel_path is not None:
            target_img = Image.open(target_img_path).convert("RGB")

        if self.transform is not None:
            ref_img = self.transform(ref_img)
            if target_img is not None:
              target_img = self.transform(target_img)

        img_set = sample.get("img_set", {})
        img_set_members = img_set.get("members", [])

        return {
            "pair_id": pair_id,
            "reference_id": ref_id,
            "target_id": target_id,
            "ref_img": ref_img,
            "target_img": target_img,
            "caption": caption,
            "img_set_members": img_set_members
        }

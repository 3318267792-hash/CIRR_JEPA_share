import json
import os
from PIL import Image
from torch.utils.data import Dataset

class GalleryDataset(Dataset):

    def __init__(self, split_json, image_root, transform=None):
       
        with open(split_json, "r", encoding="utf-8") as f:
            split_map = json.load(f)

        self.items = sorted(split_map.items(), key=lambda x: x[0])
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, rel_path = self.items[index]

        rel_path = rel_path.lstrip("./")
        image_path = os.path.join(self.image_root, rel_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found for id {image_id}: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {"image_id": image_id, "image": image}
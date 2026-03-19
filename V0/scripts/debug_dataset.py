from torchvision import transforms

from datasets.cirr_dataset import CIRRDataset


transform = transforms.ToTensor()


dataset = CIRRDataset(
    caption_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/captions/cap.rc2.train.json",
    split_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/image_splits/split.rc2.train.json",
    image_root="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/img_raw",
    transform=transform
)

print("dataset length =", len(dataset))


sample = dataset[0]
print(sample.keys())
print("pair_id =", sample["pair_id"])
print("caption =", sample["caption"])
print("ref_img shape =", sample["ref_img"].shape)

if sample["target_img"] is not None:
    print("target_img shape =", sample["target_img"].shape)

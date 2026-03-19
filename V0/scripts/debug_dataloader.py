from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.cirr_dataset import CIRRDataset


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
dataset = CIRRDataset(
    caption_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/captions/cap.rc2.train.json",
    split_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/image_splits/split.rc2.train.json",
    image_root="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/img_raw",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
    )

step = 0
for batch in loader:
    print(batch.keys())
    print("pair_id =", batch["pair_id"])
    print("caption =", batch["caption"])
    print("ref_img shape =", batch["ref_img"].shape)

    step += 1


    if batch["target_img"] is not None:
        print("target_img shape =", batch["target_img"].shape)
    if step >= 2:
        break

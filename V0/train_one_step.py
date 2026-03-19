

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.cirr_dataset import CIRRDataset
from models.baseline_query_encoder import BaselineQueryEncoder
from models.clip_target_encoder import CLIPTargetEncoder
from utils import CIRRCollateFn


def main():
    device = "mps" if torch.backends.cuda.is_available() else "cpu"
    print("device =", device)

    dataset = CIRRDataset(
        caption_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/captions/cap.rc2.train.json",
        split_json="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/image_splits/split.rc2.train.json",
        image_root="/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/img_raw",
        transform=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=CIRRCollateFn()
    )

    query_encoder = BaselineQueryEncoder(device=device).to(device)
    target_encoder = CLIPTargetEncoder(device=device).to(device)

    query_encoder.train()
    target_encoder.eval()


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, query_encoder.parameters()),


        lr=1e-4
    )


    batch = next(iter(dataloader))

    captions = batch["captions"]
    ref_images = batch["ref_images"]
    target_images = batch["target_images"]


    pred_emb = query_encoder(ref_images, captions)
    target_emb = target_encoder(target_images)

    print("pred_emb shape =", pred_emb.shape)
    print("target_emb shape =", target_emb.shape)


    sim = pred_emb @ target_emb.T
    print("sim shape =", sim.shape)
    print("sim =", sim)


    batch_size = sim.size(0)
    labels = torch.arange(batch_size, device=sim.device)

    loss = F.cross_entropy(sim, labels)


    print("loss =", loss.item())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("one training step done.")


if __name__ == "__main__":
    main()

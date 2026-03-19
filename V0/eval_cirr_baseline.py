
import torch
from torch.utils.data import DataLoader
import csv
from models.baseline_query_encoder import BaselineQueryEncoder
from models.clip_target_encoder import CLIPTargetEncoder
from datasets.cirr_dataset import CIRRDataset
from utils import CIRRCollateFn, GalleryCollateFn
from datasets.gallery_dataset import GalleryDataset
import os

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)


    max_gallery_batches = 2
    max_query_batches = 2


    query_encoder = BaselineQueryEncoder(device=device).to(device)
    target_encoder = CLIPTargetEncoder(device=device).to(device)


    checkpoint_path = "checkpoints/last.pt"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        query_encoder.load_state_dict(checkpoint["query_encoder"])

        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    query_encoder.eval()
    target_encoder.eval()


    gallery_dataset = GalleryDataset(
        split_json="data/cirr/image_splits/split.rc2.val.json",
        image_root="data/cirr/img_raw",
        transform=None
    )
    gallery_dataloader = DataLoader(
        gallery_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=GalleryCollateFn()
    )
    gallery_embs_list = []
    gallery_names = []
    with torch.no_grad():
        for gallery_batch_idx, batch in enumerate(gallery_dataloader):
            images = batch["images"]
            image_ids = batch["image_ids"]

            embs = target_encoder(images)
            gallery_embs_list.append(embs.cpu())
            gallery_names.extend(image_ids)

            if max_gallery_batches is not None and (gallery_batch_idx + 1) >= max_gallery_batches:
                break


    gallery_embs = torch.cat(gallery_embs_list, dim=0).to(device)


    assert len(gallery_names) == len(set(gallery_names)), "gallery_names 中有重复的 image_id！"
    assert gallery_embs.size(0) == len(gallery_names), "gallery_embs 行数和 gallery_names 长度不一致！"


    name_to_idx = {name: idx for idx, name in enumerate(gallery_names)}


    query_dataset = CIRRDataset(
        caption_json="data/cirr/captions/cap.rc2.val.json",
        split_json="data/cirr/image_splits/split.rc2.val.json",
        image_root="data/cirr/img_raw",
        transform=None
    )
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=CIRRCollateFn()
    )


    total = 0
    recall_1_count = 0
    recall_5_count = 0
    recall_10_count = 0
    recall_50_count = 0
    recall_sub1_count = 0
    recall_sub2_count = 0
    recall_sub3_count = 0


    with torch.no_grad():
      for query_batch_idx, batch in enumerate(query_dataloader):


        ref_images = batch["ref_images"]
        captions = batch["captions"]
        reference_ids = batch["reference_ids"]
        target_ids = batch["target_ids"]
        img_set_members = batch["img_set_members"]

        pred_embs = query_encoder(ref_images, captions)


        scores = pred_embs @ gallery_embs.T


        for i in range(scores.shape[0]):
          reference_id = reference_ids[i]
          target_id = target_ids[i]
          img_set_member = img_set_members[i]


          ref_idx = name_to_idx[reference_id]
          scores[i, ref_idx] = float("-inf")


          topk_idx = torch.topk(scores[i], k=50).indices.cpu().tolist()


          topk_names = [gallery_names[idx] for idx in topk_idx]


          total += 1
          if target_id in topk_names[:1]:
            recall_1_count += 1
          if target_id in topk_names[:5]:
            recall_5_count += 1
          if target_id in topk_names[:10]:
            recall_10_count += 1
          if target_id in topk_names[:50]:
            recall_50_count += 1


          subset_embs = []
          subset_names = []
          for member_id in img_set_member:
            if member_id == reference_id:
              continue
            idx = name_to_idx[member_id]
            subset_embs.append(gallery_embs[idx])
            subset_names.append(member_id)


          subset_embs = torch.stack(subset_embs)


          scores_subset = pred_embs[i] @ subset_embs.T


          topk_sub_idx = torch.topk(scores_subset, k=3).indices.cpu().tolist()


          topk_sub_names = [subset_names[k] for k in topk_sub_idx]


          if target_id in topk_sub_names[:1]:
            recall_sub1_count += 1
          if target_id in topk_sub_names[:2]:
            recall_sub2_count += 1
          if target_id in topk_sub_names[:3]:
            recall_sub3_count += 1

        if max_query_batches is not None and (query_batch_idx + 1) >= max_query_batches:
          break


    if total == 0:
        print("No query was evaluated. Please check max_query_batches or dataset settings.")
        return

    recall_1 = recall_1_count / total
    recall_5 = recall_5_count / total
    recall_10 = recall_10_count / total
    recall_50 = recall_50_count / total
    recall_sub1 = recall_sub1_count / total
    recall_sub2 = recall_sub2_count / total
    recall_sub3 = recall_sub3_count / total


    os.makedirs("checkpoints", exist_ok=True)
    eval_log_path = "checkpoints/eval_log.csv"
    write_header = not os.path.exists(eval_log_path)
    with open(eval_log_path, "a", newline="", encoding="utf-8") as eval_log_file:
      eval_log_writer = csv.writer(eval_log_file)
      if write_header:
        eval_log_writer.writerow([
          "checkpoint_path",
          "max_gallery_batches",
          "max_query_batches",
          "total_queries",
          "recall_at_1",
          "recall_at_5",
          "recall_at_10",
          "recall_at_50",
          "recall_subset_at_1",
          "recall_subset_at_2",
          "recall_subset_at_3",
        ])
      eval_log_writer.writerow([
        checkpoint_path,
        max_gallery_batches,
        max_query_batches,
        total,
        f"{recall_1:.6f}",
        f"{recall_5:.6f}",
        f"{recall_10:.6f}",
        f"{recall_50:.6f}",
        f"{recall_sub1:.6f}",
        f"{recall_sub2:.6f}",
        f"{recall_sub3:.6f}",
      ])

    print(f"\n===== CIRR Baseline Evaluation Results =====")
    print(f"Total queries: {total}")
    print(f"Recall@1:  {recall_1:.4f}")
    print(f"Recall@5:  {recall_5:.4f}")
    print(f"Recall@10: {recall_10:.4f}")
    print(f"Recall@50: {recall_50:.4f}")
    print(f"Recall_subset@1: {recall_sub1:.4f}")
    print(f"Recall_subset@2: {recall_sub2:.4f}")
    print(f"Recall_subset@3: {recall_sub3:.4f}")
    print(f"Eval log saved to {eval_log_path}")


if __name__ == "__main__":
    main()

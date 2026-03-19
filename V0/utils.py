class CIRRCollateFn:
    def __call__(self, batch):
        pair_ids = [sample["pair_id"] for sample in batch]
        captions = [sample["caption"] for sample in batch]
        ref_images = [sample["ref_img"] for sample in batch]
        target_images = [sample["target_img"] for sample in batch]
        reference_ids = [sample["reference_id"] for sample in batch]
        target_ids = [sample["target_id"] for sample in batch]
        img_set_members = [sample["img_set_members"] for sample in batch]


        return {
            "pair_ids": pair_ids,
            "captions": captions,
            "ref_images": ref_images,
            "target_images": target_images,
            "reference_ids": reference_ids,
            "target_ids": target_ids,
            "img_set_members": img_set_members

        }


import torch


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


class GalleryCollateFn:
    def __call__(self, batch):
        return {
            "image_ids": [x["image_id"] for x in batch],
            "images": [x["image"] for x in batch],
        }

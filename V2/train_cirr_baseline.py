import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import csv


from datasets.cirr_dataset import CIRRDataset
from models.qwen_query_encoder import QwenQueryEncoder
from models.clip_target_encoder import CLIPTargetEncoder
from utils import CIRRCollateFn, move_optimizer_to_device


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)


    max_steps = None


    os.makedirs("checkpoints", exist_ok=True)


    dataset = CIRRDataset(
        caption_json="data/cirr/captions/cap.rc2.train.json",
        split_json="data/cirr/image_splits/split.rc2.train.json",
        image_root="data/cirr/img_raw",
        transform=None
    )
    print("len(dataset) =", len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=CIRRCollateFn()
    )

    query_encoder = QwenQueryEncoder(device=device).to(device)
    target_encoder = CLIPTargetEncoder(device=device).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, query_encoder.parameters()),
        lr=1e-4
    )


    resume_path = None

    num_epochs = 3
    start_epoch = 0
    global_step = 0


    if resume_path is not None:

        ckpt = torch.load(resume_path, map_location=device)


        query_encoder.load_state_dict(ckpt["query_encoder"])


        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_to_device(optimizer, device)


        start_epoch = ckpt["epoch"] + 1

        global_step = ckpt["global_step"]

        print(f"resumed from {resume_path}, start_epoch={start_epoch}, global_step={global_step}")


    log_path = "checkpoints/train_log.csv"

    write_header = (start_epoch == 0) or not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "step", "step_loss", "avg_epoch_loss"])
        log_file.flush()

    for epoch in range(start_epoch, num_epochs):

        query_encoder.train()
        query_encoder.model.eval()
        target_encoder.eval()

        epoch_loss = 0.0
        steps_this_epoch = 0
        reached_max_steps = False

        for batch in dataloader:
            captions = batch["captions"]
            ref_images = batch["ref_images"]
            target_images = batch["target_images"]

            pred_emb = query_encoder(ref_images, captions)


            with torch.no_grad():
                target_emb = target_encoder(target_images)


            logit_scale = query_encoder.get_logit_scale()
            sim = logit_scale * (pred_emb @ target_emb.T)
            batch_size = sim.size(0)
            labels = torch.arange(batch_size, device=sim.device)

            loss = F.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps_this_epoch += 1
            global_step += 1

            if global_step % 10 == 0:
                print(f"epoch = {epoch}, step = {global_step}, loss = {loss.item():.4f}, logit_scale = {query_encoder.get_logit_scale().item():.4f}")

                log_writer.writerow([epoch, global_step, f"{loss.item():.6f}", ""])
                log_file.flush()


            if max_steps is not None and global_step >= max_steps:
                reached_max_steps = True
                break


        avg_loss = epoch_loss / steps_this_epoch
        print(f"epoch = {epoch}, avg_loss = {avg_loss:.4f}")

        log_writer.writerow([epoch, global_step, "", f"{avg_loss:.6f}"])
        log_file.flush()


        ckpt_path = f"checkpoints/cirr_baseline_epoch_{epoch}.pt"

        state = {
            "epoch": epoch,
            "global_step": global_step,

            "query_encoder": query_encoder.state_dict(),

            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)

        torch.save(state, "checkpoints/last.pt")
        print(f"saved checkpoint to {ckpt_path} + last.pt")

        if reached_max_steps:
            break

    log_file.close()
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()

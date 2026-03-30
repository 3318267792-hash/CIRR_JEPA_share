import os
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.cirr_dataset import CIRRDataset
from models.qwen_embedding_encoder import QwenEmbeddingEncoder
from utils import CIRRCollateFn, move_optimizer_to_device


def freeze_backbone_keep_mapper_trainable(query_encoder):
    # ===== 新增：冻结 Qwen backbone，仅训练 query mapper =====
    # Qwen 本体参数量巨大且已经很强，全量训练既浪费算力又容易过拟合。
    # 所以把 backbone 全部冻住（requires_grad=False 且 eval 模式关闭 dropout/BN 更新），
    # 只让 query_mapper 这个轻量 MLP 接收梯度、参与训练。
    # hasattr(obj, "attr") 检查对象是否有某个属性，返回 True/False。
    # 这里做防御性检查：确认 query_encoder 内部确实有 embedder.model 这条链路，
    # 才去冻结它，避免结构不同时直接报 AttributeError。
    if hasattr(query_encoder, "embedder") and hasattr(query_encoder.embedder, "model"):
        for p in query_encoder.embedder.model.parameters():
            p.requires_grad = False     # 冻结：不计算也不存梯度，省显存
        query_encoder.embedder.model.eval()  # eval 模式：关闭 dropout 等随机行为

    for p in query_encoder.query_mapper.parameters():
        p.requires_grad = True  # 只有 mapper 的参数可训练


def compute_loss(pred_emb, target_emb, mse_weight=1.0, ce_weight=1.0, ce_temperature=0.07):
    # ===== 新增：MSE + CE loss（双目标联合训练） =====
    #
    # 1) MSE loss（绝对距离）：逐元素对齐 query embedding 和 target embedding，
    #    让两者在向量空间中尽可能重合。
    loss_mse = F.mse_loss(pred_emb, target_emb)

    # 2) CE loss（对比损失 / 相对排序）：
    #    - pred_emb @ target_emb.T 得到 [B, B] 的相似度矩阵
    #    - 除以 temperature（默认 0.07，值越小 softmax 越尖锐，正负样本区分越极端）
    #    - 对角线是正样本对（query_i ↔ target_i），其余都是 batch 内负样本
    #    - cross_entropy 要求正样本的相似度在 softmax 后占绝对主导
    logits = (pred_emb @ target_emb.T) / ce_temperature
    labels = torch.arange(logits.size(0), device=logits.device)  # [0,1,2,...,B-1]
    loss_ce = F.cross_entropy(logits, labels)

    # 两个 loss 加权求和：MSE 管"绝对靠近"，CE 管"相对排序"
    loss = mse_weight * loss_mse + ce_weight * loss_ce
    return loss, loss_mse, loss_ce


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    max_steps = None
    num_epochs = 5
    batch_size = 16

    # lr（学习率）：每一步参数更新的步幅大小。
    #   太大 → 跳过最优解，loss 震荡甚至发散；太小 → 收敛极慢。
    #   1e-4 是 fine-tune 预训练模型时的常用起点（比从头训练的 1e-3 小一个量级），
    #   因为预训练权重已经很好，只需要小步微调。
    lr = 1e-4

    # weight_decay（权重衰减）：不让 mapper 的参数长得太大、太激进。
    #   数学上相当于给 loss 加了一项惩罚：loss_total = loss + λ * ||W||²
    #   参数越大，额外惩罚越大 → 模型更倾向于学一个小而稳的修正。
    #   和当前任务特别契合：backbone 已经 frozen 且很强，mapper 的角色是"小修小补"，
    #   不是大幅改空间。weight_decay 就像一个"刹车"，防止 mapper 学得太猛把
    #   原本很好的 frozen embedding 空间拉坏（表现为：训练 loss 降了但 val recall 反而掉）。
    #   1e-4 是常见默认值：不至于强到把学习压死，也不至于弱到没效果。
    #   如果后面发现过拟合，可以试着提到 5e-4 或 1e-3。
    weight_decay = 1e-4

    # mse_weight / ce_weight：两个 loss 分量的加权系数。
    #   最终 loss = mse_weight * MSE + ce_weight * CE
    #   都设为 1.0 表示两个 loss 等权相加，不偏向任何一方。
    #   如果发现 MSE 的数值量级远大于 CE（或反过来），可以调这两个值来平衡。
    mse_weight = 1.0
    ce_weight = 1.0

    # ce_temperature（对比损失的温度系数）：控制 softmax 的"尖锐程度"。
    #   logits = (pred @ target.T) / temperature
    #   temperature 越小 → softmax 输出越接近 one-hot，正负样本区分越极端，
    #   模型被迫把正样本的相似度拉得远高于所有负样本。
    #   0.07 是 CLIP 论文的经典值，已被大量对比学习工作验证有效。
    #   太小（如 0.01）→ 梯度爆炸，训练不稳定；太大（如 1.0）→ 区分度不够，学不动。
    ce_temperature = 0.07

    os.makedirs("checkpoints", exist_ok=True)

    dataset = CIRRDataset(
        caption_json="data/cirr/captions/cap.rc2.train.json",
        split_json="data/cirr/image_splits/split.rc2.train.json",
        image_root="data/cirr/img_raw",
        transform=None,
    )
    print("len(dataset) =", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=CIRRCollateFn(),
    )

    query_encoder = QwenEmbeddingEncoder(
        device=device,
        use_query_mapper=True,
    ).to(device)

    freeze_backbone_keep_mapper_trainable(query_encoder)

    optimizer = torch.optim.AdamW(
        query_encoder.query_mapper.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    resume_path = None
    start_epoch = 0
    global_step = 0

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)

        query_encoder.load_state_dict(ckpt["query_encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_to_device(optimizer, device)

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]

        freeze_backbone_keep_mapper_trainable(query_encoder)

        print(f"resumed from {resume_path}, start_epoch={start_epoch}, global_step={global_step}")

    log_path = "checkpoints/train_log_qwen_mlp.csv"
    write_header = (start_epoch == 0) or (not os.path.exists(log_path))
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)

    if write_header:
        log_writer.writerow([
            "epoch",
            "step",
            "step_loss",
            "step_loss_mse",
            "step_loss_ce",
            "avg_epoch_loss",
            "avg_epoch_loss_mse",
            "avg_epoch_loss_ce",
        ])
        log_file.flush()

    for epoch in range(start_epoch, num_epochs):
        query_encoder.train()

        # ===== 新增：保持 backbone eval，仅 mapper 训练 =====
        # query_encoder.train() 会把整个 Module 切到训练模式（包括子模块），
        # 但我们只想让 mapper 处于 train 模式，backbone 必须保持 eval。
        # 所以这里紧接着把 backbone 强制切回 eval，确保其 dropout/BN 行为不变。
        if hasattr(query_encoder, "embedder") and hasattr(query_encoder.embedder, "model"):
            query_encoder.embedder.model.eval()

        epoch_loss = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_ce = 0.0
        steps_this_epoch = 0
        reached_max_steps = False

        for batch in dataloader:
            captions = batch["captions"]
            ref_images = batch["ref_images"]
            target_images = batch["target_images"]

            pred_emb = query_encoder.encode_queries(
                ref_images=ref_images,
                captions=captions,
                use_mapper=True,
            )

            with torch.no_grad():
                target_emb = query_encoder.encode_images(target_images)

            loss, loss_mse, loss_ce = compute_loss(
                pred_emb=pred_emb,
                target_emb=target_emb,
                mse_weight=mse_weight,
                ce_weight=ce_weight,
                ce_temperature=ce_temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_mse += loss_mse.item()
            epoch_loss_ce += loss_ce.item()
            steps_this_epoch += 1
            global_step += 1

            if global_step % 10 == 0:
                print(
                    f"epoch = {epoch}, step = {global_step}, "
                    f"loss = {loss.item():.4f}, "
                    f"mse = {loss_mse.item():.4f}, "
                    f"ce = {loss_ce.item():.4f}"
                )

                log_writer.writerow([
                    epoch,
                    global_step,
                    f"{loss.item():.6f}",
                    f"{loss_mse.item():.6f}",
                    f"{loss_ce.item():.6f}",
                    "",
                    "",
                    "",
                ])
                log_file.flush()

            if max_steps is not None and global_step >= max_steps:
                reached_max_steps = True
                break

        avg_loss = epoch_loss / max(steps_this_epoch, 1)
        avg_loss_mse = epoch_loss_mse / max(steps_this_epoch, 1)
        avg_loss_ce = epoch_loss_ce / max(steps_this_epoch, 1)

        print(
            f"epoch = {epoch}, "
            f"avg_loss = {avg_loss:.4f}, "
            f"avg_mse = {avg_loss_mse:.4f}, "
            f"avg_ce = {avg_loss_ce:.4f}"
        )

        log_writer.writerow([
            epoch,
            global_step,
            "",
            "",
            "",
            f"{avg_loss:.6f}",
            f"{avg_loss_mse:.6f}",
            f"{avg_loss_ce:.6f}",
        ])
        log_file.flush()

        ckpt_path = f"checkpoints/cirr_qwen_mlp_epoch_{epoch}.pt"
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "query_encoder": query_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(state, ckpt_path)
        torch.save(state, "checkpoints/last_qwen_mlp.pt")
        print(f"saved checkpoint to {ckpt_path} + last_qwen_mlp.pt")

        if reached_max_steps:
            break

    log_file.close()
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
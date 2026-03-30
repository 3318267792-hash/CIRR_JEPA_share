import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/caidongao/hf_models")
from Qwen3_VL_Embedding_2B.scripts.qwen3_vl_embedding import Qwen3VLEmbedder


QWEN_EMBED_LOCAL_PATH = "/home/caidongao/hf_models/Qwen3_VL_Embedding_2B"


class QwenEmbeddingEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = QWEN_EMBED_LOCAL_PATH,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation: str | None = None,
        max_length: int = 8192,
        min_pixels: int = 4096,
        max_pixels: int = 1843200,
        default_instruction: str = "Represent the user's input.",
        normalize: bool = True,
        embed_dim: int = 2048,
        mapper_hidden_dim: int = 4096,
        mapper_dropout: float = 0.0,
        use_query_mapper: bool = False,
    ):
        super().__init__()
        self.device = device
        self.normalize = normalize
        self.embed_dim = embed_dim
        self.use_query_mapper = use_query_mapper

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_name,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            default_instruction=default_instruction,
        )

        # ===== 新增：query-side MLP mapper =====
        # 两层 MLP（2048→4096→2048），输入输出维度相同，是 V6 唯一的可训练参数。
        # 背景：Qwen embedding encoder 本体已经很强，冻结不动（不参与梯度更新）。
        # 作用：只对 query embedding 做轻量映射，让它在共享的 embedding 空间里
        #       更靠近对应的 target embedding，从而提升检索匹配精度。
        # 为什么只改 query 侧：target 侧用的是同一个冻结 encoder 的 encode_images，
        #       gallery embedding 可以离线预算一次；只动 query 侧就不用每次重算 gallery。
        self.query_mapper = nn.Sequential(
            # 第1层 Linear：2048→4096，升维。
            # 把 embedding 投射到更高维空间，给模型更多"自由度"去学习变换。
            # 参数量 = 2048×4096 + 4096(bias) ≈ 8.4M，是整个 mapper 的主要参数。
            nn.Linear(embed_dim, mapper_hidden_dim),
            # GELU 激活函数：引入非线性。
            # 如果没有激活函数，两层 Linear 叠起来等价于一层 Linear（矩阵乘法可合并），
            # 那 MLP 就退化成单层线性映射，表达能力大打折扣。
            # GELU 比 ReLU 更平滑，梯度不会在零点硬截断，训练更稳定。
            nn.GELU(),
            # Dropout：训练时随机丢弃一部分神经元输出（置零），防止过拟合。
            # mapper_dropout 默认 0.0（不丢弃），需要时可调大。
            nn.Dropout(mapper_dropout),
            # 第2层 Linear：4096→2048，降回原维度。
            # 保证输出和输入维度一致，这样才能做残差连接（emb + mapper(emb)），
            # 也能直接和 target embedding 在同一个 2048 维空间里算相似度。
            nn.Linear(mapper_hidden_dim, embed_dim),
        )

        # ===== 新增：让 residual mapper 初始近似 identity =====
        #
        # 【问题】为什么 PyTorch 默认不把权重初始化为 0？
        #   举个具体例子：假设第1层 Linear 有 4096 个神经元，权重全是 0。
        #   → 同一个输入进来，4096 个神经元全部算出相同的结果（都是 0）
        #   → loss 反传时，4096 个神经元收到的梯度也完全一样
        #   → 权重更新后，4096 个神经元的值还是一模一样
        #   → 不管训练多久，4096 个神经元永远在做同一件事，等于只有 1 个神经元
        #   这就是"对称性问题"：全零让所有神经元无法分化。
        #   所以 PyTorch 默认用 Kaiming Uniform（从一个随机范围里采样非零值），
        #   让每个神经元从第一步起就不一样，逐渐学出不同的"职责"。
        #
        # 【那为什么这里反而要零初始化？】
        #   因为这个 MLP 接在残差连接后面：output = emb + mapper(emb)
        #   用一个具体的前传和反传过程来说明为什么不会卡死：
        #
        #   前传（第一步，最后一层权重 = 0）：
        #     mapper(emb) = 0（最后一层全零，输出必然是零向量）
        #     output = emb + 0 = emb（完全等于原始 Qwen embedding，这正是我们想要的起点）
        #
        #   反传（第一步）：
        #     loss 对 output 求梯度，得到 grad_output（一个非零向量）
        #     output = emb + mapper(emb)，所以 grad_output 同时流向两条路：
        #       路1：直接传给 emb（但 emb 是 frozen 的，不更新，这条路只是"路过"）
        #       路2：传给 mapper(emb)，再链式传进 mapper 内部
        #     关键：路2 的梯度 = grad_output × 1（加法的梯度就是 1），所以是非零的。
        #     这个非零梯度继续往 mapper 里传，最后一层权重虽然是 0，但它收到了非零梯度，
        #     于是权重被更新成非零值。第二步开始，各神经元就不再相同了。
        #
        #   对比：如果没有残差连接（output = mapper(emb)），最后一层全零 → 输出全零 →
        #     loss 对全零输出的梯度传回去时，前面各层也收到零梯度 → 永远更新不动。
        #
        # 【好处】训练起点 = frozen baseline 的原始性能，MLP 从"什么都不改"渐进学修正量。
        # 【业界同类做法】LoRA 的 B 矩阵、GPT 的 residual 投影层、
        #   LeWorldModel 的 AdaLN zero init 都是同一思路。
        nn.init.zeros_(self.query_mapper[-1].weight)
        nn.init.zeros_(self.query_mapper[-1].bias)

    def _post_process(self, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.float()
        if self.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    # 这里用 no_grad 而不是 inference_mode，原因：
    #   两者都不计算梯度，但 inference_mode 产出的 tensor 是"只读"的特殊类型，
    #   PyTorch 禁止它出现在任何需要 backward 的计算里。
    #   而训练时 target_emb 虽然自身不需要梯度，但它会作为"常量"参与 mse_loss / cross_entropy，
    #   autograd 需要把它保存下来给 pred_emb 算梯度 → 如果是 inference tensor 就会报错。
    #   no_grad 产出的是普通 tensor，可以安全参与后续的 autograd 计算。
    @torch.no_grad()
    def encode_images(self, images):
        inputs = [{"image": img} for img in images]
        emb = self.embedder.process(inputs)
        emb = self._post_process(emb)
        return emb

    def encode_queries(self, ref_images, captions, instruction=None, use_mapper=None):
        assert len(ref_images) == len(captions), (
            f"len(ref_images)={len(ref_images)} != len(captions)={len(captions)}"
        )

        if instruction is None:
            instruction = (
                "Represent the composed image retrieval query formed by "
                "a reference image and a modification caption."
            )

        if use_mapper is None:
            use_mapper = self.use_query_mapper

        inputs = []
        for img, cap in zip(ref_images, captions):
            inputs.append(
                {
                    "image": img,
                    "text": cap,
                    "instruction": instruction,
                }
            )

        with torch.no_grad():
            emb = self.embedder.process(inputs)
            emb = emb.float()

        # ===== 新增：query embedding residual mapping =====
        # 残差连接：MLP 只学一个"小修正量"加回原始 embedding，
        # 而非从零学全新表示。训练初期 MLP 参数接近零时输出≈原始 embedding，起点稳定。
        if use_mapper:
            emb = emb + self.query_mapper(emb)

        if self.normalize:
            emb = F.normalize(emb, dim=-1)

        return emb

    def forward(self, ref_images, captions, instruction=None, use_mapper=None):
        return self.encode_queries(
            ref_images=ref_images,
            captions=captions,
            instruction=instruction,
            use_mapper=use_mapper,
        )
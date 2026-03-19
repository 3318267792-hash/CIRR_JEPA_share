from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="./openai_clip_vit_large_patch14"
)

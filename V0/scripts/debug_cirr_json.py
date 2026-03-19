import json
from pprint import pprint


CAPTION_JSON = "/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/captions/cap.rc2.train.json"
SPLIT_JSON = "/Users/danielcai/Desktop/Research/JEPA/任务/CIRR_JEPA/data/cirr/image_splits/split.rc2.train.json"


with open(CAPTION_JSON, "r", encoding="utf-8") as f:
    captions_data = json.load(f)


with open(SPLIT_JSON, "r", encoding="utf-8") as f:
    split_map = json.load(f)

print("=== 基本信息 ===")

print("captions_data type:", type(captions_data))


print("captions_data len :", len(captions_data))


print("split_map type    :", type(split_map))
print("split_map len     :", len(split_map))

print("\n=== 第一条样本 ===")


sample = captions_data[0]


pprint(sample)

print("\n=== 第一条样本的字段 ===")


print(sample.keys())

print("\n=== 关键字段 ===")


pair_id = sample.get("pairid")
ref_id = sample.get("reference")
caption = sample.get("caption")
target_id = sample.get("target_hard")

print("pairid   =", pair_id)
print("reference=", ref_id)
print("caption  =", caption)
print("target   =", target_id)

print("\n=== 路径映射 ===")


ref_rel_path = split_map.get(ref_id)


target_rel_path = split_map.get(target_id) if target_id is not None else None

print("ref_rel_path   =", ref_rel_path)
print("target_rel_path=", target_rel_path)

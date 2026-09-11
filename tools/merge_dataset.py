"""把规则侧与 LLM 侧的产出合并成一套可直接喂 LLaMA-Factory 的文件, 并生成 dataset_info.json。

    python tools/merge_dataset.py --in data/vqa_rule data/vqa_llm --out data/vqa

合并规则:
  - 图像样本与视频样本**始终分文件**(train.json / train_video.json), 因为在
    LLaMA-Factory 里它们是两个数据集条目, columns 分别映射 images 与 videos;
    混在一个文件里, 视频那几条会因为找不到 images 字段直接加载失败。
  - LLM 侧只产一个 all.json(没切分), 按 image_id 归到规则侧同一张图所在的切分里 ——
    同一张图的规则题和描述题必须在同一边, 否则测试集的图在训练集里见过。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = ("train", "val", "test")


def _load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else []


def main() -> None:
    ap = argparse.ArgumentParser(description="合并规则侧/LLM 侧产出并生成 dataset_info.json")
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", default="data/vqa")
    ap.add_argument("--name", default="military_anomaly", help="dataset_info 里的条目名前缀")
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="LLM 侧未切分样本的落位比例")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = {k: [] for k in SPLITS}
    where: dict[str, str] = {}                 # image_id -> split, 保证同图不跨切分
    loose: list[dict] = []

    for d in args.inp:
        d = Path(d)
        for sp in SPLITS:
            for suffix in ("", "_video"):
                for row in _load(d / f"{sp}{suffix}.json"):
                    buckets[sp].append(row)
                    where.setdefault((row.get("extra") or {}).get("image_id", ""), sp)
        for suffix in ("", "_video"):
            loose += _load(d / f"all{suffix}.json")

    r = [float(x) for x in args.ratios.split(",")]
    n_assigned = 0
    for i, row in enumerate(loose):
        img = (row.get("extra") or {}).get("image_id", "")
        sp = where.get(img)
        if sp is None:                          # 规则侧没见过这张图, 按比例落位
            sp = SPLITS[0] if (i % 10) < r[0] * 10 else (
                SPLITS[1] if (i % 10) < (r[0] + r[1]) * 10 else SPLITS[2])
            where[img] = sp
        else:
            n_assigned += 1
        buckets[sp].append(row)

    info: dict[str, dict] = {}
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user",
            "assistant_tag": "assistant", "system_tag": "system"}
    for sp in SPLITS:
        for suffix, field, sel in (("", "images", lambda x: "videos" not in x),
                                   ("_video", "videos", lambda x: "videos" in x)):
            part = [x for x in buckets[sp] if sel(x)]
            if not part:
                continue
            fn = f"{sp}{suffix}.json"
            (out / fn).write_text(json.dumps(part, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
            key = f"{args.name}{'_video' if suffix else ''}" + ("" if sp == "train" else f"_{sp}")
            info[key] = {"file_name": fn, "formatting": "sharegpt",
                         "columns": {"messages": "messages", field: field},
                         "tags": tags}
            print(f"  {fn:20s} {len(part):7d} 条   -> dataset_info 条目 {key}")

    (out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for v in buckets.values())
    print(f"\n合计 {total} 条 -> {out}")
    if loose:
        print(f"  LLM 侧 {len(loose)} 条, 其中 {n_assigned} 条按同图归到了规则侧所在的切分")
    print(f"  dataset_info.json 已写出, LLaMA-Factory 里把 dataset_dir 指到 {out} 即可")


if __name__ == "__main__":
    main()

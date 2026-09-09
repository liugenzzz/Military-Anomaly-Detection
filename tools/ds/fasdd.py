"""FASDD_UAV —— 火焰与烟雾检测（无人机视角子集, 36308 火焰 + 17222 烟雾实例）。

<https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda>

目录结构(解压 FASDD_UAV.zip 后):
  FASDD_UAV/
    images/
    annotations/{YOLO,VOC,COCO,TDML}/

处理要点:
  - **直接取 COCO 子目录**, 四种格式里它最省事
  - fire -> explosion 事件(火光), smoke -> smoke 事件; 同时保留 bbox 作为 objects,
    所以这个数据集同时供 R1(判定) 与 R2(区域定位)
  - 只有烟没有火的图归 smoke, 两者都有的同时打两个事件标签
  - FASDD 含大量「既无火也无烟」的负样本图, 这些直接进正常样本池, 很有价值
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ds.common import image_size, iter_images
from scene import Event, Obj, Scene

DATASET = "FASDD_UAV"
LICENSE = "CC-BY-4.0(请以数据集页面声明为准)"

CLASS_MAP = {"fire": "fire", "flame": "fire", "smoke": "smoke"}
CLS_TO_EVENT = {"fire": "explosion", "smoke": "smoke"}


def _coco_files(root: Path) -> list[Path]:
    cands = [p for p in root.rglob("*.json")
             if "coco" in str(p).lower() or "instances" in p.name.lower()]
    if cands:
        return sorted(cands)
    return [p for p in root.rglob("*.json")
            if '"annotations"' in p.read_text(encoding="utf-8", errors="ignore")[:4000]]


def build(root: str, view: str = "uav", keep_negatives: bool = True) -> list[Scene]:
    r = Path(root)
    files = _coco_files(r)
    if not files:
        raise RuntimeError(f"{DATASET}: 未找到 COCO 标注 json。请确认解压后存在 "
                           f"annotations/COCO 目录, 或用 adapters.py coco 手动指定")
    lookup = {p.name: p for p in iter_images(r)}
    scenes: list[Scene] = []
    seen: set[str] = set()

    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        by_img = defaultdict(list)
        for a in d.get("annotations", []):
            by_img[a["image_id"]].append(a)

        for im in d.get("images", []):
            fn = Path(im["file_name"]).name
            if fn in seen:
                continue                              # train/val json 可能重复列同一图
            seen.add(fn)
            path = lookup.get(fn)
            if path is None:
                continue
            w, h = im.get("width") or 0, im.get("height") or 0
            if not w or not h:
                w, h = image_size(path)

            objs, present = [], set()
            for i, a in enumerate(by_img.get(im["id"], [])):
                raw = cats.get(a["category_id"], "object").strip().lower()
                cls = CLASS_MAP.get(raw, raw)
                x, y, bw, bh = a["bbox"]
                objs.append(Obj(id=i, cls=cls, bbox=[x, y, x + bw, y + bh]))
                if cls in CLS_TO_EVENT:
                    present.add(cls)

            if not present and not keep_negatives:
                continue
            events = [Event(type=CLS_TO_EVENT[c], conf=1.0,
                            evidence={"rule": None, "src_label": c,
                                      "boxes": [o.bbox for o in objs if o.cls == c][:20]})
                      for c in sorted(present)]
            scenes.append(Scene(image_id=f"{DATASET}_{Path(fn).stem}", image_path=str(path),
                                width=w, height=h, source_dataset=DATASET, license=LICENSE,
                                view=view, objects=objs, events=events))

    n_neg = sum(1 for s in scenes if not s.events)
    print(f"[{DATASET}] {len(scenes)} 图, 其中无火无烟的负样本 {n_neg}")
    return scenes

"""Mendeley UAV 多类军事目标（7985 图 / 14018 实例, tank / drone / people / soldier）。

<https://data.mendeley.com/datasets/9z7yrcrpjk/1>

处理要点:
  - **Mendeley 社区数据集不声明标注格式**, 所以这里自动探测 coco / voc / yolo,
    不让使用者去猜
  - tank -> tank(命中 require_any), soldier -> soldier, people -> person, drone -> drone
  - 数据集含合成增强图。文件名带 syn/aug/synthetic 的标为 meta.synthetic=true,
    以便后续按需剔除或控制配比(合成占比建议 <= 30%)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ds.common import (detect_ann_format, image_size, iter_images, parse_voc_xml,
                       parse_yolo_txt)
from scene import Obj, Scene

DATASET = "Mendeley-UAV-Military"
LICENSE = "CC-BY-4.0(请以数据集页面声明为准)"

CLASS_MAP = {
    "tank": "tank", "tanks": "tank", "militarytank": "tank",
    "soldier": "soldier", "soldiers": "soldier", "militarypersonnel": "soldier",
    "people": "person", "person": "person", "human": "person", "civilian": "person",
    "drone": "drone", "uav": "drone", "quadcopter": "drone",
    "militarycar": "military-vehicle", "militaryvehicle": "military-vehicle",
    "militarytruck": "military-truck",
}
SYNTHETIC_HINTS = ("syn", "aug", "synthetic", "generated", "render")


def _norm(name: str) -> str:
    k = name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return CLASS_MAP.get(k, name.strip().lower())


def _is_synthetic(path: Path) -> bool:
    low = f"{path.parent.name}/{path.stem}".lower()
    return any(h in low for h in SYNTHETIC_HINTS)


def _from_coco(ann: Path, root: Path) -> list[Scene]:
    d = json.loads(ann.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    lookup = {p.name: p for p in iter_images(root)}

    scenes = []
    for im in d["images"]:
        fn = Path(im["file_name"]).name
        path = lookup.get(fn, root / im["file_name"])
        objs = []
        for i, a in enumerate(by_img.get(im["id"], [])):
            x, y, w, h = a["bbox"]
            objs.append(Obj(id=i, cls=_norm(cats.get(a["category_id"], "object")),
                            bbox=[x, y, x + w, y + h]))
        scenes.append(Scene(image_id=f"{DATASET}_{Path(fn).stem}", image_path=str(path),
                            width=im.get("width") or 0, height=im.get("height") or 0,
                            source_dataset=DATASET, license=LICENSE, view="uav",
                            objects=objs, meta={"synthetic": _is_synthetic(path)}))
    return scenes


def build(root: str, fmt: str | None = None, classes_file: str | None = None,
          view: str = "uav") -> list[Scene]:
    r = Path(root)
    fmt = fmt or detect_ann_format(r)
    print(f"[{DATASET}] 标注格式: {fmt}")

    if fmt == "coco":
        anns = [j for j in r.rglob("*.json")
                if '"annotations"' in j.read_text(encoding="utf-8", errors="ignore")[:4000]]
        if not anns:
            raise RuntimeError(f"{DATASET}: 探测为 coco 但找不到含 annotations 的 json")
        scenes: list[Scene] = []
        for a in anns:
            scenes.extend(_from_coco(a, r))
        for s in scenes:                              # COCO 里 size 可能缺失
            if not s.width or not s.height:
                s.width, s.height = image_size(s.image_path)
        return scenes

    names: list[str] = []
    if fmt == "yolo":
        cf = Path(classes_file) if classes_file else next(
            (p for p in (*r.rglob("classes.txt"), *r.rglob("*.names"), *r.rglob("data.yaml"))), None)
        if cf and cf.suffix == ".yaml":
            import yaml
            y = yaml.safe_load(cf.read_text(encoding="utf-8"))
            names = list(y.get("names", {}).values()) if isinstance(y.get("names"), dict) else list(y.get("names", []))
        elif cf:
            names = [x.strip() for x in cf.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not names:
            print(f"[warn] {DATASET}: 未找到类别文件, 类名将退化为 class_0/class_1..., "
                  f"请用 --classes 指定")

    scenes = []
    for img in iter_images(r):
        w, h = image_size(img)
        raw: list[tuple[str, list[float]]] = []
        if fmt == "voc":
            xml = next((p for p in r.rglob(f"{img.stem}.xml")), None)
            if xml:
                xw, xh, raw = parse_voc_xml(xml)
                w, h = (xw or w), (xh or h)
        elif fmt == "yolo":
            lab = next((p for p in r.rglob(f"{img.stem}.txt") if p.name != "classes.txt"), None)
            if lab and w and h:
                raw = parse_yolo_txt(lab, w, h, names)
        scenes.append(Scene(image_id=f"{DATASET}_{img.stem}", image_path=str(img),
                            width=w, height=h, source_dataset=DATASET, license=LICENSE,
                            view=view,
                            objects=[Obj(id=i, cls=_norm(n), bbox=b) for i, (n, b) in enumerate(raw)],
                            meta={"synthetic": _is_synthetic(img)}))
    n_syn = sum(1 for s in scenes if s.meta.get("synthetic"))
    if n_syn:
        print(f"[{DATASET}] 疑似合成增强图 {n_syn}/{len(scenes)}, 已标 meta.synthetic")
    return scenes

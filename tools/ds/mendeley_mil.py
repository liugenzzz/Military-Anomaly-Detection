"""Mendeley UAV 多类军事目标（7985 图 / 14018 实例, tank / drone / people / soldier）。

<https://data.mendeley.com/datasets/9z7yrcrpjk/1>

类别构成(官方数字): tank 3000 图/4990 实例, people 2644/4492, drone 1359/1296,
soldier 982/3240。

**质量注意事项 —— 这个数据集是二次汇编, 不是原始采集:**
  - 官方说明图像"主要采集自 Roboflow 和 Kaggle", 标注质量参差, 必须过筛
  - soldier 类因真实航拍素材不足, 用 **GTA5 游戏引擎生成的合成图**做了增强
  - 只是"侧重"航拍视角, 实际混有地面视角照片, 需靠 screen.py 的闸4/闸5 剔除

因此本适配器**默认只保留 tank 与 soldier 两类**:
  - drone: 反无人机检测用的类别, 与本项目 4 类异常无关, 默认丢弃
  - people: 质量不及 DroneCrowd, 默认丢弃, 人员聚集交给 DroneCrowd
  - soldier: 默认保留但标 meta.synthetic_risk, 建议控制配比或用 --keep-classes 排除
装备集结的主力应当是 MAR20(3842 图, 学术发布, 质量档次更高), 本数据集只作补充。

处理要点:
  - **Mendeley 社区数据集不声明标注格式**, 所以这里自动探测 coco / voc / yolo
  - tank -> tank(命中 ontology 的 require_any), soldier -> soldier
  - 文件名带 syn/aug/synthetic/gta 的标 meta.synthetic=true
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ds.common import (detect_ann_format, image_size, iter_images, looks_like_coco,
                       parse_voc_xml, parse_yolo_txt)
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
SYNTHETIC_HINTS = ("syn", "aug", "synthetic", "generated", "render", "gta")
# 默认保留的类别: drone 与任务无关, people 让位给 DroneCrowd
DEFAULT_KEEP = ("tank", "soldier")
# soldier 类掺有 GTA5 渲染图, 单独标记以便控制配比
SYNTHETIC_RISK_CLASSES = {"soldier"}


def _norm(name: str) -> str:
    k = name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return CLASS_MAP.get(k, name.strip().lower())


def _is_synthetic(path: Path) -> bool:
    low = f"{path.parent.name}/{path.stem}".lower()
    return any(h in low for h in SYNTHETIC_HINTS)


def _from_coco(ann: Path, root: Path, keep: set[str]) -> list[Scene]:
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
            cls = _norm(cats.get(a["category_id"], "object"))
            if keep and cls not in keep:
                continue
            x, y, w, h = a["bbox"]
            objs.append(Obj(id=i, cls=cls, bbox=[x, y, x + w, y + h]))
        scenes.append(Scene(image_id=f"{DATASET}_{Path(fn).stem}", image_path=str(path),
                            width=im.get("width") or 0, height=im.get("height") or 0,
                            source_dataset=DATASET, license=LICENSE, view="uav",
                            objects=objs, meta={"synthetic": _is_synthetic(path)}))
    return scenes


def build(root: str, fmt: str | None = None, classes_file: str | None = None,
          view: str = "uav", keep_classes: tuple[str, ...] | None = None) -> list[Scene]:
    r = Path(root)
    keep = {c.lower() for c in (keep_classes or DEFAULT_KEEP)}
    fmt = fmt or detect_ann_format(r)
    print(f"[{DATASET}] 标注格式: {fmt}  保留类别: {sorted(keep)}")

    if fmt == "coco":
        anns = [j for j in r.rglob("*.json") if looks_like_coco(j)]
        if not anns:
            raise RuntimeError(f"{DATASET}: 探测为 coco 但找不到含 annotations 的 json")
        scenes: list[Scene] = []
        for a in anns:
            scenes.extend(_from_coco(a, r, keep))
        for s in scenes:                              # COCO 里 size 可能缺失
            if not s.width or not s.height:
                s.width, s.height = image_size(s.image_path)
        return _finalize(scenes, keep)

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
        objs = [Obj(id=i, cls=c, bbox=b) for i, (c, b) in
                enumerate((_norm(n), b) for n, b in raw) if not keep or c in keep]
        scenes.append(Scene(image_id=f"{DATASET}_{img.stem}", image_path=str(img),
                            width=w, height=h, source_dataset=DATASET, license=LICENSE,
                            view=view, objects=objs,
                            meta={"synthetic": _is_synthetic(img)}))
    return _finalize(scenes, keep)


def _finalize(scenes: list[Scene], keep: set[str]) -> list[Scene]:
    for s in scenes:
        if any(o.cls in SYNTHETIC_RISK_CLASSES for o in s.objects):
            s.meta["synthetic_risk"] = True           # soldier 类掺有 GTA5 渲染图
    before = len(scenes)
    scenes = [s for s in scenes if s.objects]         # 过滤后无目标的图不留
    n_syn = sum(1 for s in scenes if s.meta.get("synthetic"))
    n_risk = sum(1 for s in scenes if s.meta.get("synthetic_risk"))
    print(f"[{DATASET}] 保留 {len(scenes)} 图(过滤掉 {before - len(scenes)} 张不含目标类别的)")
    if n_syn:
        print(f"  文件名疑似合成: {n_syn}, 已标 meta.synthetic")
    if n_risk:
        print(f"  含 soldier 类(官方声明掺有 GTA5 渲染图): {n_risk}, 已标 meta.synthetic_risk")
    print("  提醒: 本数据集为 Roboflow/Kaggle 二次汇编, 务必接着跑 screen.py 看保留率; "
          "装备集结主力建议用 MAR20")
    return scenes

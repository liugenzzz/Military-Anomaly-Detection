"""MAR20 —— 军机遥感识别（20 类军机, 3842 图, 22341 实例, HBB + OBB）。

目录结构(官方版):
  MAR20/
    JPEGImages/                     *.jpg
    Annotations/
      Horizontal Bounding Boxes/    *.xml   (VOC 格式)
      Oriented Bounding Boxes/      *.xml
    ImageSets/Main/{train,test}.txt

处理要点:
  - 20 个类别名是 A1..A20(机型代号), **全部是军用飞机**, 统一映射为 military-plane,
    机型代号保留在 attrs.model 里, 供属性题使用
  - military-plane 命中 ontology 的 require_any, 因此机群密集停放会被判为装备集结
  - Roboflow 导出版会重命名文件并重划分 train/valid/test, 适配器会递归查找, 无需改目录
  - Roboflow 的 YOLO 版类名在 data.yaml 里, 会自动读取, 不必手动给 --classes
  - **注意 Roboflow 版本的预处理**: 若该版本做了 Resize(常见默认 640x640),
    高分辨率遥感图里的小飞机会被压糊, 不适合用于 grounding 训练, 建议改用官方原版
"""
from __future__ import annotations

from pathlib import Path

from ds.common import image_size, iter_images, parse_voc_xml, parse_yolo_txt
from scene import Obj, Scene

DATASET = "MAR20"
LICENSE = "academic-research-only"

# A1..A20 全为军机机型代号 -> 统一类名, 机型进 attrs
MODEL_PREFIXES = ("a",)


def _norm(name: str) -> tuple[str, str | None]:
    n = name.strip()
    low = n.lower()
    if low.startswith(MODEL_PREFIXES) and low[1:].isdigit():
        return "military-plane", n.upper()          # A1 -> military-plane, model=A1
    if "helicopter" in low:
        return "military-helicopter", n
    return "military-plane", n                       # MAR20 里没有非军机类别


def _load_class_names(root: Path, classes_file: str | None) -> list[str]:
    """类名来源: 显式 --classes > Roboflow 的 data.yaml > classes.txt / *.names。"""
    cands = [Path(classes_file)] if classes_file else []
    cands += [*root.rglob("data.yaml"), *root.rglob("classes.txt"), *root.rglob("*.names")]
    for cf in cands:
        if not cf.exists():
            continue
        if cf.suffix in (".yaml", ".yml"):
            import yaml
            y = yaml.safe_load(cf.read_text(encoding="utf-8")) or {}
            n = y.get("names")
            if isinstance(n, dict):
                return [n[k] for k in sorted(n, key=lambda x: int(x))]
            if isinstance(n, list) and n:
                return [str(x) for x in n]
        else:
            lines = [x.strip() for x in cf.read_text(encoding="utf-8").splitlines() if x.strip()]
            if lines:
                return lines
    return []


def _find_xml(ann_root: Path, stem: str) -> Path | None:
    for cand in (ann_root / f"{stem}.xml",
                 ann_root / "Horizontal Bounding Boxes" / f"{stem}.xml",
                 ann_root / "HBB" / f"{stem}.xml"):
        if cand.exists():
            return cand
    hits = list(ann_root.rglob(f"{stem}.xml"))
    return hits[0] if hits else None


def build(root: str, img_dir: str | None = None, ann_dir: str | None = None,
          fmt: str = "voc", classes_file: str | None = None,
          view: str = "satellite") -> list[Scene]:
    r = Path(root)
    imgs_root = Path(img_dir) if img_dir else (r / "JPEGImages" if (r / "JPEGImages").is_dir() else r)
    ann_root = Path(ann_dir) if ann_dir else (r / "Annotations" if (r / "Annotations").is_dir() else r)

    names: list[str] = []
    if fmt == "yolo":
        names = _load_class_names(r, classes_file)
        if not names:
            print(f"[warn] {DATASET}: 未找到类别文件(classes.txt / data.yaml), "
                  f"类名将退化为 class_0/class_1..., 请用 --classes 指定")

    scenes, missing = [], 0
    for img in iter_images(imgs_root):
        w, h = image_size(img)
        raw: list[tuple[str, list[float]]] = []
        if fmt == "voc":
            xml = _find_xml(ann_root, img.stem)
            if xml is None:
                missing += 1
            else:
                xw, xh, raw = parse_voc_xml(xml)
                w, h = (xw or w), (xh or h)
        else:
            lab = next((p for p in (ann_root / f"{img.stem}.txt",
                                    *ann_root.rglob(f"{img.stem}.txt")) if p.exists()), None)
            if lab is None:
                missing += 1
            elif w and h:
                raw = parse_yolo_txt(lab, w, h, names)

        objs = []
        for i, (name, bbox) in enumerate(raw):
            cls, model = _norm(name)
            objs.append(Obj(id=i, cls=cls, bbox=bbox,
                            attrs={"model": model} if model else {}))
        scenes.append(Scene(image_id=f"{DATASET}_{img.stem}", image_path=str(img),
                            width=w or 1024, height=h or 1024,
                            source_dataset=DATASET, license=LICENSE, view=view,
                            objects=objs, meta={"airport_scene": True}))
    if missing:
        print(f"[warn] {DATASET}: {missing} 张图找不到对应标注, 已按无标注处理")
    return scenes

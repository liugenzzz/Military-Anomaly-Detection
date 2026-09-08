"""各公开数据集 -> 统一 Scene 的适配器。

已实现:
  coco          COCO instances json           (DOTA/FAIR1M/MAR20 的 COCO 转换版)
  yolo          YOLO txt + classes.txt        (Roboflow / Kaggle 导出的常见格式)
  visdrone-mot  VisDrone MOT annotations      (带 track_id, 用于越界派生)
  folder        按类别分目录的图像            (ERA / FASDD / UCF-Crime 抽帧)
  demo          生成一份合成样例, 用于跑通全链路

用法:
  python tools/adapters.py coco --ann xx.json --img-root images/ --dataset DOTA-v2.0 --out s.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from scene import Event, Obj, Region, Scene, dump_scenes

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# 目录名/原始标签 -> 本体事件 id。新增数据源时在这里加映射即可。
LABEL_MAP = {
    "fire": "explosion", "explosion": "explosion", "arson": "explosion",
    "riot": "crowd_gathering", "conflict": "crowd_gathering",
    "parade": "crowd_gathering", "protest": "crowd_gathering", "party": "crowd_gathering",
    "concert": "crowd_gathering", "religious_activity": "crowd_gathering",
    "smoke": "smoke", "flame_and_smoke": "smoke",
    "constructing": "fortification",
    "non-event": "normal", "non_event": "normal", "normal": "normal", "neither": "normal",
}


def _size(path: Path) -> tuple[int, int]:
    """尽量读真实尺寸; 没装 Pillow 时退化为默认值(不影响 QA 生成逻辑, 只影响归一化精度)。"""
    try:
        from PIL import Image  # noqa: PLC0415
        with Image.open(path) as im:
            return im.size
    except Exception:
        return 1920, 1080


# ---------------------------------------------------------------- COCO
def from_coco(ann_path: str, img_root: str, dataset: str, license_: str, view: str) -> list[Scene]:
    data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    by_img = defaultdict(list)
    for a in data["annotations"]:
        by_img[a["image_id"]].append(a)

    scenes = []
    for img in data["images"]:
        objs = []
        for i, a in enumerate(by_img.get(img["id"], [])):
            x, y, w, h = a["bbox"]
            objs.append(Obj(id=i, cls=cats.get(a["category_id"], "object"),
                            bbox=[x, y, x + w, y + h]))
        scenes.append(Scene(
            image_id=f"{dataset}_{Path(img['file_name']).stem}",
            image_path=str(Path(img_root) / img["file_name"]),
            width=img.get("width") or 1024, height=img.get("height") or 1024,
            source_dataset=dataset, license=license_, view=view, objects=objs))
    return scenes


# ---------------------------------------------------------------- YOLO
def from_yolo(img_dir: str, label_dir: str, classes_file: str,
              dataset: str, license_: str, view: str) -> list[Scene]:
    names = [ln.strip() for ln in Path(classes_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    scenes = []
    for img in sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT):
        w, h = _size(img)
        objs = []
        lab = Path(label_dir) / f"{img.stem}.txt"
        if lab.exists():
            for i, ln in enumerate(lab.read_text(encoding="utf-8").splitlines()):
                parts = ln.split()
                if len(parts) < 5:
                    continue
                ci, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                objs.append(Obj(id=i, cls=names[ci] if ci < len(names) else f"class_{ci}",
                                bbox=[(cx - bw / 2) * w, (cy - bh / 2) * h,
                                      (cx + bw / 2) * w, (cy + bh / 2) * h]))
        scenes.append(Scene(image_id=f"{dataset}_{img.stem}", image_path=str(img),
                            width=w, height=h, source_dataset=dataset,
                            license=license_, view=view, objects=objs))
    return scenes


# ---------------------------------------------------------------- VisDrone MOT
VISDRONE_CLS = {0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
                5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
                9: "bus", 10: "motor", 11: "others"}
VEHICLE = {"car", "van", "truck", "bus"}


def from_visdrone_mot(seq_dir: str, ann_file: str, dataset: str, license_: str,
                      key_frame_stride: int = 30, boundary: str | None = None) -> list[Scene]:
    """VisDrone MOT: <frame,target_id,x,y,w,h,score,category,truncation,occlusion>

    每 stride 帧取一个关键帧, 并把该帧前后各 stride/2 帧的轨迹点挂上去,
    供 derive_events.boundary_cross 判定越界。
    """
    rows = []
    for ln in Path(ann_file).read_text(encoding="utf-8").splitlines():
        p = ln.strip().split(",")
        if len(p) < 8:
            continue
        rows.append(dict(frame=int(p[0]), tid=int(p[1]), x=float(p[2]), y=float(p[3]),
                         w=float(p[4]), h=float(p[5]), cat=int(p[7])))
    if not rows:
        return []

    seq = Path(seq_dir)
    imgs = sorted(p for p in seq.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        return []
    W, H = _size(imgs[0])

    regions = []
    if boundary:                       # "x1,y1;x2,y2;..." 像素坐标折线
        pts = [[float(v) for v in seg.split(",")] for seg in boundary.split(";")]
        regions = [Region(name="restricted_boundary", type="polyline", points=pts)]

    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)
    frames = sorted(by_frame)
    half = max(1, key_frame_stride // 2)

    scenes = []
    for k in frames[::key_frame_stride]:
        cur = by_frame[k]
        objs, tracks = [], defaultdict(list)
        for i, r in enumerate(cur):
            cls = VISDRONE_CLS.get(r["cat"], "object")
            cls = "vehicle" if cls in VEHICLE else ("person" if cls in ("pedestrian", "people") else cls)
            objs.append(Obj(id=i, cls=cls, bbox=[r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]],
                            track_id=r["tid"]))
        ids = {r["tid"] for r in cur}
        for f in range(max(frames[0], k - half), min(frames[-1], k + half) + 1):
            for r in by_frame.get(f, []):
                if r["tid"] in ids:
                    tracks[str(r["tid"])].append([f, r["x"] + r["w"] / 2, r["y"] + r["h"] / 2])
        for tid, pts in tracks.items():
            if len(pts) >= 2:
                disp = ((pts[0][1] - pts[-1][1]) ** 2 + (pts[0][2] - pts[-1][2]) ** 2) ** 0.5
                for o in objs:
                    if str(o.track_id) == tid:
                        o.attrs["moving"] = disp > 0.01 * (W ** 2 + H ** 2) ** 0.5

        img = seq / f"{k:07d}.jpg"
        scenes.append(Scene(
            image_id=f"{dataset}_{seq.name}_frame{k:07d}",
            image_path=str(img if img.exists() else imgs[min(len(imgs) - 1, frames.index(k))]),
            width=W, height=H, source_dataset=dataset, license=license_, view="uav",
            objects=objs, regions=regions, tracks=dict(tracks), meta={"frame_idx": k}))
    return scenes


# ---------------------------------------------------------------- 分类目录
def from_folder(root: str, dataset: str, license_: str, view: str) -> list[Scene]:
    scenes = []
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        mapped = LABEL_MAP.get(cls_dir.name.lower().replace(" ", "_"))
        if mapped is None:
            print(f"[warn] 未映射的类别目录 '{cls_dir.name}'，已跳过。请在 LABEL_MAP 中补充。")
            continue
        for img in sorted(p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXT):
            w, h = _size(img)
            ev = [] if mapped == "normal" else [Event(type=mapped, conf=1.0,
                                                      evidence={"rule": None, "src_label": cls_dir.name})]
            scenes.append(Scene(image_id=f"{dataset}_{cls_dir.name}_{img.stem}",
                                image_path=str(img), width=w, height=h,
                                source_dataset=dataset, license=license_, view=view, events=ev))
    return scenes


# ---------------------------------------------------------------- demo
def make_demo(n: int = 24, seed: int = 7) -> list[Scene]:
    """合成一批 scene 用于跑通全链路(不含真实图像)。"""
    rng = random.Random(seed)
    W = H = 1280
    boundary = Region(name="restricted_zone_A", type="polyline", points=[[0, 700], [1280, 620]])
    scenes = []
    for i in range(n):
        kind = ["massing", "crowd", "crossing", "normal"][i % 4]
        objs, tracks = [], {}
        if kind == "massing":
            cx, cy = rng.uniform(300, 900), rng.uniform(200, 500)
            for j in range(rng.randint(10, 16)):
                x, y = cx + rng.uniform(-90, 90), cy + rng.uniform(-70, 70)
                objs.append(Obj(id=j, cls="vehicle", bbox=[x, y, x + 26, y + 40],
                                attrs={"moving": False}))
        elif kind == "crowd":
            cx, cy = rng.uniform(400, 800), rng.uniform(300, 600)
            for j in range(rng.randint(24, 40)):
                x, y = cx + rng.uniform(-45, 45), cy + rng.uniform(-45, 45)
                objs.append(Obj(id=j, cls="person", bbox=[x, y, x + 10, y + 18],
                                attrs={"moving": True}))
        elif kind == "crossing":
            for j in range(rng.randint(2, 4)):
                x = rng.uniform(200, 1000)
                objs.append(Obj(id=j, cls="vehicle", bbox=[x, 520, x + 28, 560],
                                track_id=j, attrs={"moving": True}))
                tracks[str(j)] = [[0, x + 14, 900], [5, x + 14, 780], [10, x + 14, 540]]
        else:
            for j in range(rng.randint(2, 5)):
                x, y = rng.uniform(0, 1200), rng.uniform(0, 1200)
                objs.append(Obj(id=j, cls="vehicle", bbox=[x, y, x + 25, y + 40],
                                attrs={"moving": bool(rng.getrandbits(1))}))
        scenes.append(Scene(image_id=f"demo_{kind}_{i:03d}",
                            image_path=f"data/raw/demo/{kind}_{i:03d}.jpg",
                            width=W, height=H, source_dataset="DEMO-synthetic",
                            license="cc0", view="uav", objects=objs,
                            regions=[boundary] if kind == "crossing" else [],
                            tracks=tracks))
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description="公开数据集 -> Scene 适配器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dataset", required=True, help="数据集名, 会写进 source_dataset")
        p.add_argument("--license", default="unknown")
        p.add_argument("--view", default="uav", choices=["uav", "satellite", "cctv", "ground"])
        p.add_argument("--out", required=True)

    p = sub.add_parser("coco"); p.add_argument("--ann", required=True); p.add_argument("--img-root", required=True); common(p)
    p = sub.add_parser("yolo"); p.add_argument("--img-dir", required=True); p.add_argument("--label-dir", required=True); p.add_argument("--classes", required=True); common(p)
    p = sub.add_parser("visdrone-mot"); p.add_argument("--seq-dir", required=True); p.add_argument("--ann", required=True); p.add_argument("--stride", type=int, default=30); p.add_argument("--boundary", default=None, help='像素折线, 如 "0,700;1280,620"'); common(p)
    p = sub.add_parser("folder"); p.add_argument("--root", required=True); common(p)
    p = sub.add_parser("demo"); p.add_argument("--n", type=int, default=24); p.add_argument("--out", required=True)

    a = ap.parse_args()
    if a.cmd == "coco":
        scenes = from_coco(a.ann, a.img_root, a.dataset, a.license, a.view)
    elif a.cmd == "yolo":
        scenes = from_yolo(a.img_dir, a.label_dir, a.classes, a.dataset, a.license, a.view)
    elif a.cmd == "visdrone-mot":
        scenes = from_visdrone_mot(a.seq_dir, a.ann, a.dataset, a.license, a.stride, a.boundary)
    elif a.cmd == "folder":
        scenes = from_folder(a.root, a.dataset, a.license, a.view)
    else:
        scenes = make_demo(a.n)
    dump_scenes(scenes, a.out)
    print(f"写出 {len(scenes)} 个 scene -> {a.out}")


if __name__ == "__main__":
    main()

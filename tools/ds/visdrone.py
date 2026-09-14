"""VisDrone DET + MOT —— 航拍底座与越界移动派生。

<https://github.com/VisDrone/VisDrone-Dataset>

两个子集用途不同:
  DET  每图一个 txt 标注, 提供航拍 bbox 底座与大量正常场景负样本
  MOT  带 track_id, **越界移动只能从这里派生**

关键功能 auto_boundary:
  越界判定需要一条禁区边界线。手工给 56 个序列各画一条折线约要 1 小时,
  这里改为自动放置: 取所有轨迹的平均运动方向, 在轨迹中心处作一条与之垂直的直线。
  这样必然有相当比例的轨迹穿过它, 无需人工。仍建议抽查若干序列确认位置合理。
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

from ds.common import image_size, iter_images
from scene import Obj, Region, Scene

DATASET_DET = "VisDrone2019-DET"
DATASET_MOT = "VisDrone2019-MOT"
LICENSE = "academic-research-only"

# VisDrone 类别 -> 本体类名。全部是民用场景, 所以车辆一律 vehicle 而非 military-*,
# 这正是它成为困难负样本来源的原因
RAW = {0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
       6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others"}
CLASS_MAP = {"pedestrian": "person", "people": "person",
             "car": "vehicle", "van": "vehicle", "bus": "vehicle", "truck": "truck",
             "bicycle": "bicycle", "motor": "motorcycle",
             "tricycle": "vehicle", "awning-tricycle": "vehicle"}
SKIP = {"ignored", "others"}


def _norm(cat: int) -> str | None:
    raw = RAW.get(cat, "others")
    if raw in SKIP:
        return None
    return CLASS_MAP.get(raw, raw)


# ---------------------------------------------------------------- DET
def build_det(root: str, view: str = "uav") -> list[Scene]:
    r = Path(root)
    img_dir = next((d for d in (r / "images", r) if d.is_dir()), r)
    ann_dir = next((d for d in (r / "annotations", r) if d.is_dir()), r)
    scenes = []
    for img in iter_images(img_dir, recursive=False) or iter_images(img_dir):
        w, h = image_size(img)
        objs = []
        lab = ann_dir / f"{img.stem}.txt"
        if lab.exists():
            for i, ln in enumerate(lab.read_text(encoding="utf-8", errors="ignore").splitlines()):
                p = ln.strip().rstrip(",").split(",")
                if len(p) < 6:
                    continue
                try:
                    x, y, bw, bh = (float(v) for v in p[:4])
                    cat = int(float(p[5]))
                except ValueError:
                    continue
                cls = _norm(cat)
                if cls is None or bw <= 0 or bh <= 0:
                    continue
                objs.append(Obj(id=i, cls=cls, bbox=[x, y, x + bw, y + bh]))
        scenes.append(Scene(image_id=f"{DATASET_DET}_{img.stem}", image_path=str(img),
                            width=w, height=h, source_dataset=DATASET_DET,
                            license=LICENSE, view=view, objects=objs))
    print(f"[{DATASET_DET}] {len(scenes)} 图")
    return scenes


# ---------------------------------------------------------------- 自动边界
def auto_boundary(tracks: dict[str, list[list[float]]], W: int, H: int) -> list[list[float]]:
    """在轨迹中心处作一条垂直于平均运动方向的直线, 延伸到图像边界。"""
    dxs, dys, cxs, cys = [], [], [], []
    for pts in tracks.values():
        if len(pts) < 2:
            continue
        s, e = pts[0], pts[-1]
        dxs.append(e[1] - s[1])
        dys.append(e[2] - s[2])
        cxs.append((s[1] + e[1]) / 2)
        cys.append((s[2] + e[2]) / 2)
    if not dxs:
        return [[0, H / 2], [W, H / 2]]               # 无轨迹信息, 退化为水平中线

    dx, dy = sum(dxs) / len(dxs), sum(dys) / len(dys)
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return [[0, H / 2], [W, H / 2]]
    dx, dy = dx / n, dy / n
    px, py = -dy, dx                                  # 垂直方向
    cx, cy = sum(cxs) / len(cxs), sum(cys) / len(cys)
    L = float(W + H)                                  # 足够长, 保证跨出画面
    return [[cx - px * L, cy - py * L], [cx + px * L, cy + py * L]]


def auto_boundaries(tracks: dict[str, list[list[float]]], W: int, H: int,
                    n: int) -> list[list[list[float]]]:
    """沿运动方向铺 n 条平行边界线。

    只画一条线(过轨迹中心)时, 一个序列里只有中间那几帧算得上越界, 其余全是负样本
    —— 这正是 border_crossing 产量上不去的根本原因: 24201 帧只换来两千多条 QA。
    沿运动方向把线铺开, 不同的帧会被不同的线截住, 同一段素材因此能产出**不同的**
    问答(越界目标不同、时机不同、方向不同), 而不是同一条答案复制 n 遍。

    这不是凭空造数据: 每条线都是一条同样合理的"禁区边界", 答案仍由真实轨迹算出。
    """
    if n <= 1:
        return [auto_boundary(tracks, W, H)]
    segs = [(p[0], p[-1]) for p in tracks.values() if len(p) >= 2]
    if not segs:
        return [auto_boundary(tracks, W, H)]

    dx = sum(e[1] - s[1] for s, e in segs) / len(segs)
    dy = sum(e[2] - s[2] for s, e in segs) / len(segs)
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return [auto_boundary(tracks, W, H)]
    ux, uy = dx / norm, dy / norm                     # 平均运动方向
    px, py = -uy, ux                                  # 与之垂直, 即边界线方向

    # 把所有轨迹点投影到运动轴上, 取投影范围, 在范围内等距铺线
    proj = [(p[1] * ux + p[2] * uy) for pts in tracks.values() for p in pts]
    lo, hi = min(proj), max(proj)
    if hi - lo < 1e-6:
        return [auto_boundary(tracks, W, H)]
    ox = sum(p[1] for pts in tracks.values() for p in pts) / max(1, len(proj))
    oy = sum(p[2] for pts in tracks.values() for p in pts) / max(1, len(proj))
    o_proj = ox * ux + oy * uy
    L = float(W + H)

    lines = []
    for i in range(n):
        t = lo + (hi - lo) * ((i + 1) / (n + 1))
        cx, cy = ox + ux * (t - o_proj), oy + uy * (t - o_proj)
        lines.append([[cx - px * L, cy - py * L], [cx + px * L, cy + py * L]])
    return lines


def _seg_hit(p1, p2, p3, p4) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def cross_frames(tracks: dict[str, list[list[float]]],
                 line: list[list[float]]) -> set[int]:
    """这条线在哪些帧被穿过。用来把边界配给真的有越界发生的帧。"""
    out: set[int] = set()
    a, b = line[0], line[-1]
    for pts in tracks.values():
        for i in range(len(pts) - 1):
            if _seg_hit((pts[i][1], pts[i][2]), (pts[i + 1][1], pts[i + 1][2]), a, b):
                out.add(int(pts[i + 1][0]))
    return out


def _parse_boundaries(path: str | None) -> dict[str, list[list[float]]]:
    if not path:
        return {}
    import json
    return {k: [[float(a), float(b)] for a, b in v]
            for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


# ---------------------------------------------------------------- MOT
def build_mot(root: str, stride: int = 30, boundaries: str | None = None,
              auto: bool = True, view: str = "uav", n_boundaries: int = 3) -> list[Scene]:
    """MOT 标注: <frame,target_id,x,y,w,h,score,category,truncation,occlusion>"""
    r = Path(root)
    seq_root = next((d for d in (r / "sequences", r) if d.is_dir()), r)
    ann_root = next((d for d in (r / "annotations", r) if d.is_dir()), r)
    manual = _parse_boundaries(boundaries)
    # 把 split 写进 image_id。VisDrone 官方的 train/val/test-dev 序列名本来就不重样,
    # 但三个 split 一起收的时候, 万一撞名就是"重复 image_id"被静默剔除一半,
    # 加个前缀是很便宜的保险。
    split = re.sub(r"^VisDrone\d*-?MOT-?", "", r.name, flags=re.I).strip("-_") or "main"
    split = re.sub(r"[^A-Za-z0-9]", "", split).lower()

    seq_dirs = sorted(d for d in seq_root.iterdir() if d.is_dir()) or [seq_root]
    scenes, n_auto = [], 0

    for seq in seq_dirs:
        imgs = iter_images(seq, recursive=False)
        if not imgs:
            continue
        ann = ann_root / f"{seq.name}.txt"
        if not ann.exists():
            hits = sorted(ann_root.rglob(f"{seq.name}.txt"))
            if not hits:
                print(f"[warn] {DATASET_MOT}: 序列 {seq.name} 无标注, 跳过")
                continue
            ann = hits[0]

        rows = []
        for ln in ann.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.strip().rstrip(",").split(",")
            if len(p) < 8:
                continue
            try:
                rows.append({"f": int(float(p[0])), "tid": int(float(p[1])),
                             "x": float(p[2]), "y": float(p[3]),
                             "w": float(p[4]), "h": float(p[5]), "cat": int(float(p[7]))})
            except ValueError:
                continue
        if not rows:
            continue

        W, H = image_size(imgs[0])
        by_frame = defaultdict(list)
        for row in rows:
            by_frame[row["f"]].append(row)
        frames = sorted(by_frame)
        all_tracks: dict[str, list[list[float]]] = defaultdict(list)
        for row in rows:
            all_tracks[str(row["tid"])].append([row["f"], row["x"] + row["w"] / 2,
                                                row["y"] + row["h"] / 2])
        for v in all_tracks.values():
            v.sort(key=lambda p: p[0])

        man = manual.get(seq.name)
        lines: list[list[list[float]]] = []
        if man is not None:
            lines = [man]
        elif auto:
            lines = auto_boundaries(all_tracks, W, H, n_boundaries)
            n_auto += len(lines)
        # 每条线各自在哪些帧被穿过 —— 配边界时优先给真有越界发生的帧
        hits = [cross_frames(all_tracks, ln) for ln in lines]

        half = max(1, stride // 2)
        diag = math.hypot(W, H)
        for k in frames[::stride]:
            cur = by_frame[k]
            objs, tracks = [], {}
            ids = {row["tid"] for row in cur}
            for tid in ids:
                seg = [p for p in all_tracks[str(tid)] if k - half <= p[0] <= k + half]
                if len(seg) >= 2:
                    tracks[str(tid)] = seg
            for i, row in enumerate(cur):
                cls = _norm(row["cat"])
                if cls is None or row["w"] <= 0 or row["h"] <= 0:
                    continue
                seg = tracks.get(str(row["tid"]), [])
                moving = None
                if len(seg) >= 2:
                    moving = math.dist(seg[0][1:], seg[-1][1:]) > 0.01 * diag
                objs.append(Obj(id=i, cls=cls, track_id=row["tid"],
                                bbox=[row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]],
                                attrs={} if moving is None else {"moving": moving}))
            # 这一帧附近哪条线真被穿过就用哪条; 都没有就轮着来(那就是个负样本)
            region = None
            if lines:
                bi = next((i for i, hs in enumerate(hits)
                           if any(k - half <= f <= k + half for f in hs)),
                          (k // max(1, stride)) % len(lines))
                region = Region(name=f"restricted_{seq.name}_{bi}", type="polyline",
                                points=lines[bi])
            img = seq / f"{k:07d}.jpg"
            if not img.exists():
                idx = frames.index(k)
                img = imgs[min(idx, len(imgs) - 1)]
            scenes.append(Scene(
                image_id=f"{DATASET_MOT}_{split}_{seq.name}_frame{k:07d}", image_path=str(img),
                width=W, height=H, source_dataset=DATASET_MOT, license=LICENSE, view=view,
                objects=objs, regions=[region] if region else [], tracks=tracks,
                meta={"sequence": seq.name, "split": split, "frame_idx": k,
                      "boundary_source": ("manual" if seq.name in manual
                                          else ("auto" if lines else "none"))}))

    print(f"[{DATASET_MOT}] {len(scenes)} 帧(stride={stride}), 自动放置边界 {n_auto} 条"
          f"(每序列 {n_boundaries} 条), 人工边界 {len(manual)} 条")
    if n_auto:
        print("  自动边界 = 垂直于平均运动方向、过轨迹中心的直线。"
              "建议抽查几个序列的越界结果是否合理")
    return scenes

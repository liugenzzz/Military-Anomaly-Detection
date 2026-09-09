"""DroneCrowd —— 航拍密集人群（112 段 / 33600 帧 1080p / 480 万人头点 / 20800 条轨迹）。

<https://github.com/VisDrone/DroneCrowd>

处理要点:
  - 标注是**人头点**不是 bbox。点转成 ±half 的小方框, 统一走 bbox 逻辑;
    密度聚类只用中心点, 所以框的大小不影响聚集判定
  - 官方发布形态有 .mat 与 txt 两种, 这里都支持并自动探测。
    .mat 需要 scipy(pip install scipy); txt 会按列数自动识别
  - 33600 帧几乎全是相邻近重复, **必须按 stride 稀疏抽帧**, 默认每 30 帧取 1
  - 人头点数少于阈值的帧不会被 density_cluster 判为聚集, 自动成为正常样本,
    这批"稀疏人群"是人员聚集类很好的困难负样本
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ds.common import image_size, iter_images, points_to_boxes
from scene import Obj, Scene

DATASET = "DroneCrowd"
LICENSE = "academic-research-only"


def _load_mat(path: Path) -> dict[int, list[tuple[float, float]]]:
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise RuntimeError(
            "读取 DroneCrowd 的 .mat 标注需要 scipy: pip install scipy\n"
            "或改用官方的 txt 标注(本适配器同样支持)") from e
    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    per_frame: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for key, val in m.items():
        if key.startswith("__"):
            continue
        try:                                          # 常见形态: N x 3/4 (frame, id, x, y)
            for row in val:
                seq = list(row)
                if len(seq) >= 4:
                    per_frame[int(seq[0])].append((float(seq[-2]), float(seq[-1])))
                elif len(seq) == 3:
                    per_frame[int(seq[0])].append((float(seq[1]), float(seq[2])))
        except TypeError:
            continue
    return per_frame


def _load_txt(path: Path) -> dict[int, list[tuple[float, float]]]:
    """按列数自动识别: 4 列(frame,id,x,y) / 3 列(frame,x,y) / 2 列(x,y, 单帧)。"""
    per_frame: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = [x for x in ln.replace(",", " ").split() if x]
        if not p:
            continue
        try:
            nums = [float(x) for x in p]
        except ValueError:
            continue
        if len(nums) >= 4:
            per_frame[int(nums[0])].append((nums[-2], nums[-1]))
        elif len(nums) == 3:
            per_frame[int(nums[0])].append((nums[1], nums[2]))
        elif len(nums) == 2:
            per_frame[0].append((nums[0], nums[1]))
    return per_frame


def _find_ann(ann_root: Path, seq: str) -> Path | None:
    for pat in (f"{seq}.mat", f"{seq}.txt", f"*{seq}*.mat", f"*{seq}*.txt"):
        hits = sorted(ann_root.rglob(pat))
        if hits:
            return hits[0]
    return None


def build(root: str, ann_dir: str | None = None, stride: int = 30,
          head_half: float = 8.0, view: str = "uav") -> list[Scene]:
    r = Path(root)
    ann_root = Path(ann_dir) if ann_dir else r
    # 序列目录: 含图像的子目录
    seq_dirs = sorted({p.parent for p in iter_images(r)})
    if not seq_dirs:
        raise RuntimeError(f"{DATASET}: 在 {r} 下找不到图像")

    scenes, no_ann = [], []
    for seq in seq_dirs:
        imgs = iter_images(seq, recursive=False)
        if not imgs:
            continue
        ann = _find_ann(ann_root, seq.name)
        per_frame = {}
        if ann is not None:
            per_frame = _load_mat(ann) if ann.suffix == ".mat" else _load_txt(ann)
        else:
            no_ann.append(seq.name)

        W, H = image_size(imgs[0])
        for k, img in enumerate(imgs):
            if k % stride:
                continue
            # 帧号: 优先用文件名尾部数字, 回落到序号
            digits = "".join(ch for ch in img.stem if ch.isdigit())
            fno = int(digits[-6:]) if digits else k
            pts = per_frame.get(fno) or per_frame.get(k) or per_frame.get(k + 1) or []
            objs = [Obj(id=i, cls="person", bbox=b)
                    for i, b in enumerate(points_to_boxes(pts, head_half, W, H))]
            scenes.append(Scene(
                image_id=f"{DATASET}_{seq.name}_frame{fno:06d}",
                image_path=str(img), width=W, height=H,
                source_dataset=DATASET, license=LICENSE, view=view, objects=objs,
                meta={"sequence": seq.name, "frame_idx": fno, "head_points": len(pts)}))

    if no_ann:
        print(f"[warn] {DATASET}: {len(no_ann)} 个序列找不到标注文件 "
              f"(如 {no_ann[:3]}), 这些帧的人头数为 0, 会被当作正常样本")
    dense = sum(1 for s in scenes if len(s.objects) >= 20)
    print(f"[{DATASET}] {len(scenes)} 帧(stride={stride}), 其中人头 >=20 的密集帧 {dense}")
    return scenes

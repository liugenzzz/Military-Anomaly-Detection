"""Drone-Anomaly —— 航拍异常检测（7 场景 / 51635 训练帧 + 35853 测试帧）。

<https://github.com/Jin-Pu/Drone-Anomaly>

**重要: 这个数据集只用来产出正常样本, 不产出异常样本。**

理由: 它的 10 类异常是面板缺陷、铁轨障碍物、不明物体一类的场景异常, 与我们定稿的
4 类(聚集/爆炸/烟雾/越界)完全不同。把它的异常帧标成我们的异常是错标; 标成正常
也是错标。所以:
  - training 帧(全为正常)          -> normal, 且标 meta.paired_scene
  - testing 中标签为 0 的帧         -> normal
  - testing 中标签为 1 的帧         -> **排除**, 不进任何池子

它的价值在于提供大量真实航拍正常帧, 且与异常帧同场景同机位 —— 场景多样性远好于
随便找的正常图。
"""
from __future__ import annotations

from pathlib import Path

from ds.common import image_size, iter_images
from scene import Scene

DATASET = "Drone-Anomaly"
LICENSE = "academic-research-only"


def _load_labels(path: Path) -> list[int] | None:
    """帧级标签: 支持 txt(每行一个 0/1 或空格分隔) 与 npy(需要 numpy)。"""
    if path.suffix == ".npy":
        try:
            import numpy as np
            return [int(v) for v in np.load(path).ravel()]
        except Exception:
            return None
    try:
        toks = path.read_text(encoding="utf-8", errors="ignore").replace(",", " ").split()
        return [int(float(t)) for t in toks]
    except (ValueError, OSError):
        return None


def _find_labels(root: Path, seq: str) -> Path | None:
    for pat in (f"{seq}.txt", f"{seq}.npy", f"*{seq}*.txt", f"*{seq}*.npy"):
        hits = [p for p in sorted(root.rglob(pat)) if "image" not in p.parts]
        if hits:
            return hits[0]
    return None


def build(root: str, stride: int = 10, view: str = "uav") -> list[Scene]:
    r = Path(root)
    seq_dirs = sorted({p.parent for p in iter_images(r)})
    if not seq_dirs:
        raise RuntimeError(f"{DATASET}: 在 {r} 下找不到图像")

    scenes, n_excluded, n_unlabeled = [], 0, 0
    for seq in seq_dirs:
        imgs = iter_images(seq, recursive=False)
        if not imgs:
            continue
        parts = {p.lower() for p in seq.parts}
        is_train = bool({"training", "train"} & parts)
        scene_name = next((p for p in seq.parts[::-1]
                           if p.lower() not in {"frames", "images", "training", "train",
                                                "testing", "test"}), seq.name)
        labels = None if is_train else _load_labels(_find_labels(r, seq.name) or Path("/nonexistent"))
        if not is_train and labels is None:
            n_unlabeled += len(imgs)
            continue                                  # 测试段没有标签, 无法判断正常与否, 整段跳过

        for k, img in enumerate(imgs):
            if k % stride:
                continue
            if not is_train:
                if k >= len(labels):
                    continue
                if labels[k]:
                    n_excluded += 1
                    continue                          # 异常帧: 类型与本项目不匹配, 排除
            w, h = image_size(img)
            split = "train" if is_train else "test"
            scenes.append(Scene(
                # split 必须进 id: training/ 与 testing/ 下的子目录常同名(如都叫 frames),
                # 不带 split 会产生 image_id 冲突
                image_id=f"{DATASET}_{scene_name}_{split}_{seq.name}_frame{k:06d}",
                image_path=str(img), width=w, height=h,
                source_dataset=DATASET, license=LICENSE, view=view,
                meta={"paired_scene": scene_name, "split": split, "frame_idx": k}))

    print(f"[{DATASET}] 保留正常帧 {len(scenes)}(stride={stride}), "
          f"排除异常帧 {n_excluded}, 因缺标签跳过 {n_unlabeled}")
    print("  说明: 本数据集的异常类型(面板缺陷/铁轨障碍等)与本项目 4 类异常不匹配, "
          "按设计只取正常帧")
    return scenes

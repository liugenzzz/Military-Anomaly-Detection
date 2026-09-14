"""DOTA v2.0 —— 遥感俯视大图，本项目里主要用作**困难负样本来源**。

<https://captain-whu.github.io/DOTA/dataset.html>

目录结构:
  DOTA/
    images/          *.png   原图可达 20000x20000
    labelTxt/        *.txt   x1 y1 x2 y2 x3 y3 x4 y4 category difficult

处理要点:
  - **必须切片**。原图不切无法送模型, 且小目标缩放后直接消失。默认 1024/overlap 200
  - 切片边缘截断的框按面积占比 < 0.6 丢弃 —— 留半个框会教出错误的尺寸先验
  - **类名映射是这个数据集最关键的一步**: DOTA 的 plane 大量是民航客机、ship 大量是
    民用港口船舶。映射为 civil-plane / ship(而非 military-plane / warship),
    于是"民用机场停满飞机""港口密集停泊"会被判为 hard_negative 而不是装备集结。
    这正是我们要的 —— 这批样本教模型看目标类型, 而不是看有没有一堆东西挤在一起
  - 体育场馆、泳池等无关类别直接丢弃, 不占标注预算
"""
from __future__ import annotations

import random
from pathlib import Path

from derive_events import union_find_cluster
from ds.common import (clip_boxes_to_tile, image_size, open_large,
                       parse_dota_txt, slice_image)
from scene import Obj, Scene

DATASET = "DOTA-v2.0"
LICENSE = "academic-research-only"

# 军用/民用必须分开命名, 详见模块 docstring
CLASS_MAP = {
    "plane": "civil-plane",
    "helicopter": "civil-helicopter",
    "ship": "ship",
    "large-vehicle": "large-vehicle",
    "small-vehicle": "vehicle",
    "storage-tank": "storage-tank",
    "harbor": "harbor",
    "bridge": "bridge",
    "airport": "airport",
    "helipad": "helipad",
    "container-crane": "crane",
}
# 与任务无关, 直接丢弃
DROP = {"baseball-diamond", "tennis-court", "basketball-court", "ground-track-field",
        "roundabout", "soccer-ball-field", "swimming-pool"}
# 参与聚集判定的类别, 用于决定切片是否值得保留
RELEVANT = {"civil-plane", "civil-helicopter", "ship", "large-vehicle", "vehicle"}


def _cluster_tiles(raw: list[tuple[str, list[float]]], W: int, H: int, tile: int,
                   eps_ratio: float = 0.06, min_size: int = 8) -> list[tuple[int, int]]:
    """在**原图坐标**上先聚类, 为每个簇算出一个能完整包住它的切片左上角。

    必须这么做: 规则网格切片会把跨切片边界的聚集切散(实测 9 架机群被切成 5+6,
    每片都不足阈值, 聚集判定直接失效)。以簇为中心额外切一块才能保住簇的完整性。
    """
    rel = [(c, b) for c, b in raw if c in RELEVANT]
    pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in rel]
    if len(pts) < min_size:
        return []
    sizes = sorted(max(b[2] - b[0], b[3] - b[1]) for _, b in rel)
    eps = max(eps_ratio * ((W ** 2 + H ** 2) ** 0.5), 3.0 * sizes[len(sizes) // 2])
    origins = []
    for g in union_find_cluster(pts, eps):
        if len(g) < min_size:
            continue
        xs = [pts[i][0] for i in g]
        ys = [pts[i][1] for i in g]
        if (max(xs) - min(xs)) > tile or (max(ys) - min(ys)) > tile:
            continue                                  # 簇本身比切片还大, 放弃
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        x0 = int(max(0, min(W - tile, cx - tile / 2)))
        y0 = int(max(0, min(H - tile, cy - tile / 2)))
        origins.append((x0, y0))
    return origins


def build(root: str, tiles_dir: str, tile: int = 1024, overlap: int = 200,
          min_objects: int = 5, empty_ratio: float = 0.1,
          seed: int = 0, view: str = "satellite") -> list[Scene]:
    r = Path(root)
    img_dir = next((d for d in (r / "images", r) if d.is_dir()), r)
    lab_dir = next((d for d in (r / "labelTxt", r / "labels", r) if d.is_dir()), r)
    rng = random.Random(seed)

    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".tif", ".tiff"))
    if not imgs:
        raise RuntimeError(f"{DATASET}: 在 {img_dir} 下找不到图像")

    scenes, n_tiles, n_kept_empty, n_cluster_tiles = [], 0, 0, 0
    failed: list[tuple[str, str]] = []
    for img in imgs:
        lab = lab_dir / f"{img.stem}.txt"
        raw: list[tuple[str, list[float]]] = []
        if lab.exists():
            for cls, bbox, diff in parse_dota_txt(lab):
                key = cls.strip().lower()
                if key in DROP:
                    continue
                if diff:                              # difficult 标记的目标质量差, 丢弃
                    continue
                raw.append((CLASS_MAP.get(key, key), bbox))

        # 单张图切片失败不能连累整批: DOTA 里混着 8 亿像素的巨图和破损文件,
        # 让异常冒到顶上会把前面已经切好的几千张片子一起丢掉。
        try:
            tiles = slice_image(img, tiles_dir, tile=tile, overlap=overlap)

            # 补上以簇为中心的切片, 防止规则网格把聚集切散
            W, H = _orig_size(img)
            grid_origins = {(t["x"], t["y"]) for t in tiles}
            extra = [o for o in _cluster_tiles(raw, W, H, tile) if o not in grid_origins]
            if extra:
                # 一次打开切完所有簇中心块。原来是每块重新 open 一次 ——
                # 对 8 亿像素的原图, 那等于每块都重解一遍 2.4 GB。
                tiles.extend(_crop_many(img, tiles_dir, extra, tile, W, H))
                n_cluster_tiles += len(extra)
        except Exception as e:                        # noqa: BLE001
            failed.append((img.name, f"{type(e).__name__}: {e}"))
            continue

        for t in tiles:
            n_tiles += 1
            local = clip_boxes_to_tile(raw, t)
            n_rel = sum(1 for c, _ in local if c in RELEVANT)
            if n_rel < min_objects:
                if rng.random() > empty_ratio:        # 空切片只按比例保留一部分
                    continue
                n_kept_empty += 1
            scenes.append(Scene(
                image_id=f"{DATASET}_{Path(t['path']).stem}",
                image_path=t["path"], width=t["w"], height=t["h"],
                source_dataset=DATASET, license=LICENSE, view=view,
                objects=[Obj(id=i, cls=c, bbox=b) for i, (c, b) in enumerate(local)],
                meta={"src_image": img.name, "tile_x": t["x"], "tile_y": t["y"],
                      "cluster_centered": t.get("cluster_centered", False)}))

    print(f"[{DATASET}] {len(imgs)} 张原图 -> {n_tiles} 个切片 -> 保留 {len(scenes)} "
          f"(其中稀疏/空切片 {n_kept_empty}, 以簇为中心补切 {n_cluster_tiles})")
    print("  提示: DOTA 的 plane/ship 映射为 civil-plane/ship, 因此密集民用机群与"
          "港口会成为困难负样本, 这是预期行为")
    if failed:
        print(f"[warn] {DATASET}: {len(failed)} 张原图切片失败, 已跳过(其余照常产出):")
        for name, why in failed[:5]:
            print(f"         {name}  {why}")
        if len(failed) > 5:
            print(f"         ... 另有 {len(failed) - 5} 张")
    return scenes


def _orig_size(img: Path) -> tuple[int, int]:
    return image_size(img)          # 读文件头, 超大图也不解码


def _crop_many(img: Path, out_dir: str, origins: list[tuple[int, int]], tile: int,
               W: int, H: int) -> list[dict]:
    """以给定原点批量切块。**只打开一次原图** —— DOTA-v2.0 的巨图解一次要几秒
    和几 GB 内存, 每块重开一次会把这一张图的耗时乘上块数。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    todo = []
    for x0, y0 in origins:
        x2, y2 = min(x0 + tile, W), min(y0 + tile, H)
        dst = out / f"{img.stem}__c{x0}_{y0}{img.suffix}"
        made.append({"path": str(dst), "x": x0, "y": y0, "w": x2 - x0, "h": y2 - y0,
                     "cluster_centered": True})
        if not dst.exists():
            todo.append((dst, (x0, y0, x2, y2)))
    if todo:
        with open_large(img) as im:
            for dst, box in todo:
                im.crop(box).save(dst)
    return made

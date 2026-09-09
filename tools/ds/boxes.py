"""坐标统一层：各数据集的原始标注 -> 像素 xyxy -> Qwen-VL 的 0~1000 bbox_2d。

**为什么需要这一层**：九个数据集的坐标约定各不相同 ——
  YOLO(Mendeley / MAR20-Roboflow)   归一化 [cx,cy,w,h]，0~1
  COCO(FASDD)                        像素 [x,y,w,h]
  VOC(MAR20 官方)                    像素 [xmin,ymin,xmax,ymax]
  DOTA                               像素四点多边形
  VisDrone                           像素 [x,y,w,h]
  DroneCrowd                         像素点坐标
把归一化当像素（或反过来）会让坐标整体错乱，而且**不会报错**，只会静默产出
一堆错框。所以这里除了做换算，还提供 infer_coord_mode 与 audit_scene_boxes
两个自检函数，在 prepare 阶段就把这类问题喊出来。

换算约定对齐 liugenzzz/target_detection_vl_dataset 的 core/coords.py：
  - 中间表示一律是**像素 xyxy**，越界裁剪在像素域完成（与标注软件行为一致）
  - 输出为 0~1000 整数，两点式 [x1,y1,x2,y2]
  - 答案里写成 {"bbox_2d":[...],"label":"..."}，这是 Qwen-VL 的原生格式
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

BBOX_SCALE = 1000
COORD_MODE = "qwen_relative_1000"          # 写进 extra，下游不必猜


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- 换算
def yolo_to_pixel_xyxy(cx: float, cy: float, w: float, h: float,
                       img_w: int, img_h: int) -> list[float]:
    """YOLO 归一化中心点式 [cx,cy,w,h] -> 像素 [x1,y1,x2,y2]。"""
    bw, bh = w * img_w, h * img_h
    x1 = clamp(cx * img_w - bw / 2, 0.0, float(img_w))
    y1 = clamp(cy * img_h - bh / 2, 0.0, float(img_h))
    x2 = clamp(x1 + bw, 0.0, float(img_w))
    y2 = clamp(y1 + bh, 0.0, float(img_h))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def norm_xyxy_to_pixel(box: Sequence[float], img_w: int, img_h: int) -> list[float]:
    """归一化两点式 [x1,y1,x2,y2](0~1) -> 像素。"""
    x1, y1, x2, y2 = box
    return [clamp(x1 * img_w, 0, img_w), clamp(y1 * img_h, 0, img_h),
            clamp(x2 * img_w, 0, img_w), clamp(y2 * img_h, 0, img_h)]


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def to_bbox2d(pixel_xyxy: Sequence[float], img_w: int, img_h: int,
              scale: int = BBOX_SCALE) -> list[int]:
    """像素 [x1,y1,x2,y2] -> 0~scale 整数。"""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"图片尺寸必须为正，得到 {img_w}x{img_h}")
    x1, y1, x2, y2 = pixel_xyxy
    return [int(round(clamp(x1, 0, img_w) / img_w * scale)),
            int(round(clamp(y1, 0, img_h) / img_h * scale)),
            int(round(clamp(x2, 0, img_w) / img_w * scale)),
            int(round(clamp(y2, 0, img_h) / img_h * scale))]


# ---------------------------------------------------------------- 输出格式
def box_json(bbox2d: Sequence[int], label: str) -> str:
    """单框答案。格式对齐 Qwen-VL 原生约定，模型预训练时见过这个形状。"""
    return json.dumps({"bbox_2d": list(bbox2d), "label": label},
                      ensure_ascii=False, separators=(",", ":"))


def boxes_json(items: Iterable[tuple[Sequence[int], str]]) -> str:
    """多框答案。"""
    return json.dumps([{"bbox_2d": list(b), "label": l} for b, l in items],
                      ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- 自检
def infer_coord_mode(boxes: Iterable[Sequence[float]]) -> str:
    """按整批框判断是归一化还是像素，返回 'normalized' | 'pixel' | 'unknown'。

    **按整批判、不按单框判**：单个框 [0,0,1,1] 在两种约定下都合法，
    但一整批框全都 <= 1.0 就几乎不可能是像素坐标。
    """
    vals = [v for b in boxes for v in b]
    if not vals:
        return "unknown"
    hi = max(abs(v) for v in vals)
    if hi <= 1.001:
        return "normalized"
    if hi > 2.0:
        return "pixel"
    return "unknown"                       # 1.0~2.0 之间，说不准


def audit_scene_boxes(scenes: list[Any], sample: int = 500) -> list[str]:
    """检查一批 Scene 的框是否可疑，返回告警列表（空表示没问题）。

    这些问题都不会让程序崩溃，只会静默产出错框，所以必须主动查。
    """
    warns: list[str] = []
    subset = scenes[:sample]
    boxes = [o.bbox for s in subset for o in s.objects]
    if not boxes:
        return warns

    mode = infer_coord_mode(boxes)
    if mode == "normalized":
        warns.append("所有框的坐标值都 ≤ 1.0 —— 极可能是**归一化坐标被当成像素**存进了 "
                     "Scene.bbox。Scene 约定存像素坐标，请检查该数据集的适配器")
    elif mode == "unknown":
        warns.append("框的坐标值全部落在 1.0~2.0 之间，无法判断是归一化还是像素，请人工确认")

    oob = sum(1 for s in subset for o in s.objects
              if o.bbox[2] > s.width * 1.01 or o.bbox[3] > s.height * 1.01)
    if oob:
        warns.append(f"{oob} 个框超出图像边界 —— 可能是尺寸读错，或标注与图像不配对")

    inverted = sum(1 for s in subset for o in s.objects
                   if o.bbox[0] >= o.bbox[2] or o.bbox[1] >= o.bbox[3])
    if inverted:
        warns.append(f"{inverted} 个框的 x1>=x2 或 y1>=y2 —— 可能把 [x,y,w,h] 当成了 [x1,y1,x2,y2]")

    no_size = sum(1 for s in subset if not s.width or not s.height)
    if no_size:
        warns.append(f"{no_size} 个 scene 没有图像尺寸 —— 坐标无法归一化，这些图会被筛掉")

    areas = [((o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1])) / max(1, s.width * s.height)
             for s in subset for o in s.objects]
    if areas and sum(a > 0.9 for a in areas) / len(areas) > 0.5:
        warns.append("超过一半的框占了整幅画面 —— 可能坐标尺度整体放大了")
    return warns


# ---------------------------------------------------------------- 质量门槛
def box_quality_ok(bbox: Sequence[float], img_w: int, img_h: int, *,
                   min_area_ratio: float = 0.0005, max_area_ratio: float = 0.6,
                   min_short_side_px: int = 12) -> bool:
    """框级质量门槛。参数比目标检测项目略松：航拍小目标本来就小。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return False
    if min(w, h) < min_short_side_px:
        return False
    ratio = (w * h) / (img_w * img_h)
    return min_area_ratio <= ratio <= max_area_ratio


# ---------------------------------------------------------------- 尺寸(读文件头)
def image_size_fast(path: str | Path) -> tuple[int, int]:
    """只读文件头拿宽高，不解码像素。遍历几万张图时比 PIL 快一个量级。

    识别失败时回落到 PIL（DOTA 的大 tif 等冷门格式）。
    """
    p = Path(path)
    try:
        with p.open("rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"BM" and len(head) >= 26:
                f.seek(18)
                w, h = struct.unpack("<ii", f.read(8))
                return abs(int(w)), abs(int(h))
            if head[:2] == b"\xff\xd8":
                return _jpeg_size(f)
    except OSError:
        return 0, 0
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:
        return 0, 0


def _jpeg_size(f) -> tuple[int, int]:
    f.seek(2)
    while True:
        b = f.read(1)
        if not b:
            return 0, 0
        if b != b"\xff":
            continue
        marker = f.read(1)
        while marker == b"\xff":
            marker = f.read(1)
        if not marker:
            return 0, 0
        m = marker[0]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            continue
        seg = f.read(2)
        if len(seg) < 2:
            return 0, 0
        length = struct.unpack(">H", seg)[0]
        if m in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            data = f.read(5)
            if len(data) < 5:
                return 0, 0
            h, w = struct.unpack(">HH", data[1:5])
            return int(w), int(h)
        f.seek(length - 2, 1)

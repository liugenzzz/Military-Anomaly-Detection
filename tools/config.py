"""生成端配置。散在代码里的常量全部收到 configs/generate.yaml。

    from config import CFG
    CFG.task("correct")          # 题型配比
    CFG.count("exact_max")       # 计数阈值
    CFG.area_text(0.22)          # 面积占比的统一说法

改配置不用动代码; 代码里不再留第二份默认值 —— 两处默认值早晚会对不上。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = "configs/generate.yaml"


class Config:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("CONFIG") or DEFAULT_PATH)
        self.d: dict[str, Any] = {}
        if self.path.exists():
            self.d = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}

    def get(self, *keys, default=None):
        cur: Any = self.d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # -------------------------------------------------- 各节的便捷读取
    def task(self, name: str, default: float = 0.0) -> float:
        return float(self.get("tasks", name, default=default))

    def count(self, name: str, default: float = 0.0) -> float:
        return float(self.get("counting", name, default=default))

    def quota(self, name: str, default=0):
        return self.get("quota", name, default=default)

    def facet_weight(self, kind: str, default: int = 10) -> int:
        return int(self.get("facets", "weights", kind, default=default))

    @property
    def skip_facets(self) -> set[str]:
        return set(self.get("facets", "skip", default=["position"]))

    @property
    def max_facets(self) -> int:
        return int(self.get("facets", "max_per_image", default=8))

    def area_text(self, ratio: float) -> str:
        """面积占比 -> 固定档位的说法。

        原先百分比("百分之五")、成("一成")、分数("六分之一")、模糊量混着用,
        模型学不到稳定映射。档位在配置里改, 不在代码里改。
        """
        for b in self.get("area_buckets", default=[]) or []:
            if ratio < float(b["max"]):
                return str(b["text"])
        return "覆盖画面大部分区域"

    # -------------------------------------------------- 坐标制式
    @property
    def coord_mode(self) -> str:
        return str(self.get("coordinate", "mode", default="relative_1000"))

    @property
    def coord_scale(self) -> int:
        return int(self.get("coordinate", "scale", default=1000))

    def smart_resize(self, w: int, h: int) -> tuple[int, int]:
        """Qwen2.5-VL 的预处理尺寸。绝对像素模式下, 坐标要按这个尺寸给,
        而不是原图尺寸 —— 模型看到的就是 resize 之后的图。"""
        import math
        sr = self.get("coordinate", "smart_resize", default={}) or {}
        factor = int(sr.get("factor", 28))
        min_px = int(sr.get("min_pixels", 256 * 28 * 28))
        max_px = int(sr.get("max_pixels", 1280 * 28 * 28))
        hh = max(factor, round(h / factor) * factor)
        ww = max(factor, round(w / factor) * factor)
        if hh * ww > max_px:
            beta = math.sqrt(h * w / max_px)
            hh = max(factor, math.floor(h / beta / factor) * factor)
            ww = max(factor, math.floor(w / beta / factor) * factor)
        elif hh * ww < min_px:
            beta = math.sqrt(min_px / (h * w))
            hh = math.ceil(h * beta / factor) * factor
            ww = math.ceil(w * beta / factor) * factor
        return ww, hh


CFG = Config()

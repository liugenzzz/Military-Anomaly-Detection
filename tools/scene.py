"""统一中间表示 Scene 的定义与读写。

所有异构数据源(VisDrone / DOTA / ERA / FASDD / MAR20 ...)先由 adapters.py
归一到 Scene, 之后的事件派生与 QA 生成只认这一种结构。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Obj:
    id: int
    cls: str
    bbox: list[float]                      # [x1, y1, x2, y2] 像素坐标
    track_id: int | None = None
    score: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Region:
    """禁区/边界。type 为 polygon(闭合区域) 或 polyline(边界线)。"""
    name: str
    type: str
    points: list[list[float]]


@dataclass
class Event:
    type: str                              # 对应 ontology 中的 class id
    conf: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    image_id: str
    image_path: str
    width: int
    height: int
    source_dataset: str
    license: str = "unknown"
    view: str = "uav"                      # uav | satellite | cctv | ground
    objects: list[Obj] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    tracks: dict[str, list[list[float]]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    caption: str | None = None
    # 三种输入形态: image(单图) / multi_image(多帧序列) / video(整段视频)。
    # 静态数据集保持单图, ERA 这类短视频保持视频, 越界这类需要逐帧对位的用多帧。
    modality: str = "image"
    frames: list[str] = field(default_factory=list)   # multi_image 时按时序排列
    video_path: str | None = None                     # video 时的视频文件
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def media(self) -> tuple[str, list[str]]:
        """返回 (字段名, 路径列表), 供生成器直接写进 ShareGPT。"""
        if self.modality == "video" and self.video_path:
            return "videos", [self.video_path]
        if self.modality == "multi_image" and self.frames:
            return "images", list(self.frames)
        return "images", [self.image_path]

    @property
    def n_media(self) -> int:
        return len(self.media[1])

    @property
    def diag(self) -> float:
        return math.hypot(self.width, self.height)

    def count(self, classes: set[str]) -> int:
        return sum(1 for o in self.objects if o.cls.lower() in classes)

    def of_classes(self, classes: set[str]) -> list[Obj]:
        return [o for o in self.objects if o.cls.lower() in classes]

    @property
    def anomaly_types(self) -> list[str]:
        seen, out = set(), []
        for e in self.events:
            if e.type not in seen:
                seen.add(e.type)
                out.append(e.type)
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scene":
        return cls(
            image_id=d["image_id"],
            image_path=d["image_path"],
            width=int(d["width"]),
            height=int(d["height"]),
            source_dataset=d.get("source_dataset", "unknown"),
            license=d.get("license", "unknown"),
            view=d.get("view", "uav"),
            objects=[Obj(**o) for o in d.get("objects", [])],
            regions=[Region(**r) for r in d.get("regions", [])],
            tracks={k: v for k, v in d.get("tracks", {}).items()},
            events=[Event(**e) for e in d.get("events", [])],
            caption=d.get("caption"),
            modality=d.get("modality", "image"),
            frames=list(d.get("frames", [])),
            video_path=d.get("video_path"),
            meta=d.get("meta", {}),
        )


def load_scenes(path: str | Path) -> list[Scene]:
    """读取 JSONL(每行一个 scene) 或 JSON 数组。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix == ".jsonl" or text[0] != "[":
        return [Scene.from_dict(json.loads(ln)) for ln in text.splitlines() if ln.strip()]
    return [Scene.from_dict(d) for d in json.loads(text)]


def dump_scenes(scenes: list[Scene], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in scenes:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)

"""ERA + CapERA —— 航拍事件视频（2864 段 × 5 秒, 25 类事件）+ 人工 caption。

ERA:     <https://lcmou.github.io/ERA_Dataset/>   按类别分目录存放视频
CapERA:  <https://github.com/yakoubbazi/CapEra>   每段视频 5 条人工 caption

处理要点:
  - 每段视频抽 3 帧(首/中/尾)。5 秒视频抽更多帧只会产生近重复
  - 25 个类别分三档处理, 这个划分是关键:
      ANOMALY  映射为我们的异常事件
      NORMAL   明确无烟火的场景 -> 安全负样本(其中运动车辆场景是很好的困难负样本)
      EXCLUDE  **歧义类别一律排除**, 不进负样本池
    排除的理由: 地震后/滑坡/泥石流/洪水/交通事故/赛车 这些画面里常带烟尘或火光,
    当成"正常"喂进去, 等于教模型"有烟也算正常", 会直接破坏烟雾类的判别能力
  - ERA 没有 bbox, 所以这批数据供判定/分类/描述题, 不供 grounding 题
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ds.common import extract_frames, image_size, iter_videos
from scene import Event, Scene

DATASET = "ERA"
LICENSE = "research-only(视频源自 YouTube, 按原协议使用)"

# 映射为异常事件。conflict/parade 等归入 massing 的 personnel 子类
ANOMALY = {
    "fire": ("explosion", None),
    "conflict": ("massing", "personnel"),
    "parade_protest": ("massing", "personnel"),
    "parade": ("massing", "personnel"),
    "protest": ("massing", "personnel"),
    "party": ("massing", "personnel"),
    "concert": ("massing", "personnel"),
    "religious_activity": ("massing", "personnel"),
}
# 明确无烟火, 可安全作为负样本
NORMAL = {
    "non_event", "non-event", "nonevent",
    "traffic_congestion", "police_chase",
    "harvesting", "ploughing",
    "baseball", "basketball", "boating", "cycling", "running", "soccer", "swimming",
}
# 歧义类别: 画面常含烟尘/火光/土方, 既不能当异常也不能当正常, 直接排除
EXCLUDE = {
    "post_earthquake", "flood", "landslide", "mudslide",
    "traffic_collision", "car_racing", "constructing",
}


def _norm_label(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _load_capera(path: str | Path) -> dict[str, list[str]]:
    """CapERA caption 加载。官方发布形态可能是 dict 或 records, 两种都兼容。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}

    def put(key: str, caps) -> None:
        k = Path(str(key)).stem
        if isinstance(caps, str):
            caps = [caps]
        out.setdefault(k, []).extend([str(c) for c in caps if c])

    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                put(v.get("video") or v.get("video_id") or k,
                    v.get("captions") or v.get("caption") or [])
            else:
                put(k, v)
    elif isinstance(raw, list):
        for rec in raw:
            if isinstance(rec, dict):
                put(rec.get("video") or rec.get("video_id") or rec.get("id") or "",
                    rec.get("captions") or rec.get("caption") or [])
    if not out:
        print("[warn] CapERA: 未解析出任何 caption, 请检查文件结构后调整 _load_capera")
    return out


def build(root: str, frames_dir: str, n_frames: int = 3,
          capera_json: str | None = None, view: str = "uav",
          include_normal: bool = True) -> list[Scene]:
    r = Path(root)
    caps = _load_capera(capera_json) if capera_json else {}
    scenes: list[Scene] = []
    stat = {"anomaly": 0, "normal": 0, "excluded": 0, "unknown": 0}

    for cls_dir in sorted(p for p in r.iterdir() if p.is_dir()):
        label = _norm_label(cls_dir.name)
        if label in EXCLUDE:
            n = len(iter_videos(cls_dir))
            stat["excluded"] += n
            print(f"[{DATASET}] 排除歧义类别 {cls_dir.name} ({n} 段) —— 画面常含烟尘或火光")
            continue
        if label in ANOMALY:
            kind, subtype = ANOMALY[label]
        elif label in NORMAL:
            kind, subtype = None, None
            if not include_normal:
                continue
        else:
            stat["unknown"] += len(iter_videos(cls_dir))
            print(f"[warn] {DATASET}: 未归档的类别 '{cls_dir.name}', 已跳过。"
                  f"请在 era.py 的 ANOMALY/NORMAL/EXCLUDE 中补充")
            continue

        for vid in iter_videos(cls_dir):
            out_sub = Path(frames_dir) / label
            frames = extract_frames(vid, out_sub, n_frames=n_frames,
                                    prefix=f"{label}__{vid.stem}")
            if not frames:
                continue
            caption_list = caps.get(vid.stem, [])
            for fi, fp in enumerate(frames):
                w, h = image_size(fp)
                events = []
                if kind:
                    ev = {"rule": None, "src_label": cls_dir.name}
                    if subtype:
                        ev["subtype"] = subtype
                    events = [Event(type=kind, conf=1.0, evidence=ev)]
                scenes.append(Scene(
                    image_id=f"{DATASET}_{label}_{vid.stem}_frame{fi}",
                    image_path=str(fp), width=w or 640, height=h or 640,
                    source_dataset=DATASET, license=LICENSE, view=view,
                    events=events,
                    caption=caption_list[fi % len(caption_list)] if caption_list else None,
                    meta={"src_video": vid.name, "src_label": cls_dir.name,
                          "frame_idx": fi, "all_captions": caption_list}))
            stat["anomaly" if kind else "normal"] += 1

    print(f"[{DATASET}] 视频: 异常 {stat['anomaly']} / 正常 {stat['normal']} / "
          f"排除 {stat['excluded']} / 未归档 {stat['unknown']}  ->  {len(scenes)} 帧")
    if caps:
        n_cap = sum(1 for s in scenes if s.caption)
        print(f"[{DATASET}] 已挂上 CapERA caption 的帧: {n_cap}/{len(scenes)}")
    return scenes

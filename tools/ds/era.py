"""ERA + CapERA —— 航拍事件视频（2864 段 × 5 秒, 25 类事件）+ 人工 caption。

ERA:     <https://lcmou.github.io/ERA_Dataset/>
         实际发布结构是三层: Videos/{Training,Test}/<类别>/<名称>.mp4
         —— 类别在第三层不是第二层, 而且文件名里带空格("Concert_016 .mp4")。
         另有官方 SingleFrames/ 单帧分类数据, 可作为图像模态一并采用。
CapERA:  <https://github.com/yakoubbazi/CapEra>   每段视频 5 条人工 caption

处理要点:
  - **默认保留整段视频**(modality=video)。5 秒片段天然适合视频输入, 而且人群运动、
    火焰跳动这些线索只有视频看得到, 抽成静帧就丢了。需要静帧时用 --modality frame
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

from ds.common import VID_EXT, extract_frames, image_size, iter_images, iter_videos
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


# 这些目录名是划分层(train/test), 不是类别
SPLIT_DIRS = {"train", "training", "trainval", "test", "testing", "val", "validation", "all"}


def _norm_label(name: str) -> str:
    """把类别名归一到"去掉所有分隔符的小写"。

    ERA 实际发布的目录名是驼峰式(CarRacing / ParadeProtest / TrafficCollision),
    而论文与多数镜像写作下划线或空格形式。只替换空格与连字符的话, 驼峰名会原样
    保留成 carracing, 与映射表里的 car_racing 对不上 —— 实测因此漏掉 682 段视频,
    其中 ParadeProtest 与 ReligiousActivity 本该计入人员聚集,
    TrafficCollision / PostEarthquake / CarRacing 本该被排除。
    剥掉所有非字母数字字符, 三种写法就都能对上。
    """
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def _canon(d: dict | set) -> dict | set:
    """把映射表的键也归一到同一形式。"""
    if isinstance(d, dict):
        return {_norm_label(k): v for k, v in d.items()}
    return {_norm_label(k) for k in d}


# 三张映射表统一归一, 这样源文件里仍可写成可读的 parade_protest,
# 而与磁盘上的 ParadeProtest 能对上
ANOMALY = _canon(ANOMALY)
NORMAL = _canon(NORMAL)
EXCLUDE = _canon(EXCLUDE)
SPLIT_DIRS = _canon(SPLIT_DIRS)


def _class_dirs(root: Path) -> list[Path]:
    """找出真正的类别目录。

    ERA 的类别在 Videos/{Training,Test}/<类别>/ 第三层, 但别的镜像可能是两层,
    所以这里不写死层数: 递归找含媒体文件的目录, 目录名不是 train/test 这类划分层
    的就当类别。
    """
    hits: dict[str, Path] = {}
    for d in sorted(root.rglob("*")):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if _norm_label(d.name) in SPLIT_DIRS:
            continue
        if any(x.suffix.lower() in VID_EXT for x in d.iterdir() if x.is_file()):
            hits.setdefault(_norm_label(d.name), d)      # 同名类别只取一次
    return list(hits.values())


def _class_dirs_all(root: Path) -> list[Path]:
    """同上, 但保留同名类别的所有目录(Training 与 Test 各一份)。"""
    out = []
    for d in sorted(root.rglob("*")):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if _norm_label(d.name) in SPLIT_DIRS:
            continue
        if any(x.suffix.lower() in VID_EXT for x in d.iterdir() if x.is_file()):
            out.append(d)
    return out


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


def build(root: str, frames_dir: str | None = None, n_frames: int = 3,
          capera_json: str | None = None, view: str = "uav",
          include_normal: bool = True, modality: str = "video") -> list[Scene]:
    """modality: video 保留整段视频(推荐, 5 秒片段天然适合视频输入);
    frame 抽帧成单图; both 两种都产(会让同一段视频出现在两种样本里, 注意去重)。"""
    r = Path(root)
    caps = _load_capera(capera_json) if capera_json else {}
    scenes: list[Scene] = []
    stat = {"anomaly": 0, "normal": 0, "excluded": 0, "unknown": 0}
    warned: set[str] = set()

    for cls_dir in _class_dirs_all(r):
        label = _norm_label(cls_dir.name)
        if label in EXCLUDE:
            n = sum(1 for x in cls_dir.iterdir()
                    if x.is_file() and x.suffix.lower() in VID_EXT)
            stat["excluded"] += n
            if label not in warned:
                warned.add(label)
                print(f"[{DATASET}] 排除歧义类别 {cls_dir.name} —— 画面常含烟尘或火光")
            continue
        if label in ANOMALY:
            kind, subtype = ANOMALY[label]
        elif label in NORMAL:
            kind, subtype = None, None
            if not include_normal:
                continue
        else:
            stat["unknown"] += sum(1 for x in cls_dir.iterdir()
                                   if x.is_file() and x.suffix.lower() in VID_EXT)
            if label not in warned:
                warned.add(label)
                print(f"[warn] {DATASET}: 未归档的类别 '{cls_dir.name}', 已跳过。"
                      f"请在 era.py 的 ANOMALY/NORMAL/EXCLUDE 中补充")
            continue

        for vid in sorted(x for x in cls_dir.iterdir()
                          if x.is_file() and x.suffix.lower() in VID_EXT):
            stem = vid.stem.strip()          # 官方文件名带尾空格: "Concert_016 .mp4"
            caption_list = caps.get(stem, []) or caps.get(vid.stem, [])

            def _events():
                if not kind:
                    return []
                ev = {"rule": None, "src_label": cls_dir.name}
                if subtype:
                    ev["subtype"] = subtype
                return [Event(type=kind, conf=1.0, evidence=ev)]

            if modality in ("video", "both"):
                scenes.append(Scene(
                    image_id=f"{DATASET}_{label}_{stem}",
                    image_path=str(vid), modality="video", video_path=str(vid),
                    width=640, height=640,
                    source_dataset=DATASET, license=LICENSE, view=view,
                    events=_events(),
                    caption=caption_list[0] if caption_list else None,
                    meta={"src_video": vid.name, "src_label": cls_dir.name,
                          "duration_s": 5, "all_captions": caption_list}))

            if modality in ("frame", "both"):
                if not frames_dir:
                    raise ValueError("modality 含 frame 时必须给 --frames-dir")
                frames = extract_frames(vid, Path(frames_dir) / label, n_frames=n_frames,
                                        prefix=f"{label}__{stem}")
                for fi, fp in enumerate(frames):
                    w, h = image_size(fp)
                    scenes.append(Scene(
                        image_id=f"{DATASET}_{label}_{stem}_frame{fi}",
                        image_path=str(fp), width=w or 640, height=h or 640,
                        source_dataset=DATASET, license=LICENSE, view=view,
                        events=_events(),
                        caption=caption_list[fi % len(caption_list)] if caption_list else None,
                        meta={"src_video": vid.name, "src_label": cls_dir.name,
                              "frame_idx": fi, "all_captions": caption_list}))
            stat["anomaly" if kind else "normal"] += 1

    n_v = sum(1 for s in scenes if s.modality == "video")
    print(f"[{DATASET}] 视频: 异常 {stat['anomaly']} 段 / 正常 {stat['normal']} 段 / "
          f"排除 {stat['excluded']} / 未归档 {stat['unknown']}  ->  "
          f"{n_v} 个视频样本" + (f" + {len(scenes) - n_v} 个抽帧样本" if len(scenes) > n_v else ""))
    if caps:
        n_cap = sum(1 for s in scenes if s.caption)
        print(f"[{DATASET}] 已挂上 CapERA caption: {n_cap}/{len(scenes)}")
    return scenes


def build_single_frames(root: str, view: str = "uav",
                        include_normal: bool = True) -> list[Scene]:
    """ERA 官方的 SingleFrames/ 单帧分类数据。

    与视频样本互补: 视频教模型看运动, 单帧教它从静态画面判断。两者都要,
    但 image_id 前缀不同, 不会被当成重复。
    """
    r = Path(root)
    sf = next((d for d in (r / "SingleFrames", r) if d.is_dir()), None)
    if sf is None:
        return []
    scenes: list[Scene] = []
    stat = {"anomaly": 0, "normal": 0, "excluded": 0, "unknown": 0}

    for d in sorted(x for x in sf.rglob("*") if x.is_dir() and not x.name.startswith(".")):
        label = _norm_label(d.name)
        if label in SPLIT_DIRS:
            continue
        imgs = [x for x in d.iterdir() if x.is_file() and x.suffix.lower() in
                (".jpg", ".jpeg", ".png", ".bmp")]
        if not imgs:
            continue
        if label in EXCLUDE:
            stat["excluded"] += len(imgs)
            continue
        if label in ANOMALY:
            kind, subtype = ANOMALY[label]
        elif label in NORMAL:
            if not include_normal:
                continue
            kind, subtype = None, None
        else:
            stat["unknown"] += len(imgs)
            continue

        for img in sorted(imgs):
            w, h = image_size(img)
            events = []
            if kind:
                ev = {"rule": None, "src_label": d.name}
                if subtype:
                    ev["subtype"] = subtype
                events = [Event(type=kind, conf=1.0, evidence=ev)]
            scenes.append(Scene(
                image_id=f"{DATASET}-SF_{label}_{img.stem.strip()}",
                image_path=str(img), width=w or 640, height=h or 640,
                source_dataset=DATASET + "-SingleFrames", license=LICENSE, view=view,
                events=events, meta={"src_label": d.name}))
        stat["anomaly" if kind else "normal"] += len(imgs)

    print(f"[{DATASET}-SingleFrames] {len(scenes)} 张单帧 "
          f"(异常 {stat['anomaly']} / 正常 {stat['normal']} / "
          f"排除 {stat['excluded']} / 未归档 {stat['unknown']})")
    return scenes

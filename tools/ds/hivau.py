"""HIVAU-70k + UCF-Crime + XD-Violence —— 爆炸类数据与困难负样本的主要来源。

HIVAU-70k: <https://github.com/pipixin321/HolmesVAU>（7 万+ 层级化异常理解指令）
UCF-Crime: <https://www.crcv.ucf.edu/projects/real-world/>
XD-Violence: <https://roc-ng.github.io/XD-Violence/>

**这个数据集的处理是全流程里最需要小心的一处**, 三个要点:

1. 13 类/6 类异常中只有少数映射到我们的 4 类。Explosion/Arson/G -> explosion,
   Riot/B4 -> massing/personnel。Fighting、Robbery、Shooting、Stealing、Vandalism
   等**一律排除** —— 它们是治安事件, 不是我们定义的军事异常, 混进来会把类别搞脏。

2. **异常视频的正常段是最优质的负样本**。一段 4 分钟的 Explosion 视频里可能只有 8 秒
   在爆炸, 其余 3 分 52 秒是同机位同光照的正常画面。用事件时间边界把它们切出来,
   模型就无法靠"这个场景看着就危险"蒙对, 只能真的去看有没有火光。
   这些帧标 meta.hard_negative_source = 原异常类型。

3. 没有时间边界的异常视频**不能用**: 不知道哪几秒是异常, 抽出来的帧会大面积错标。
   正常视频不需要边界, 整段都是正常。

时间边界来源(任一即可):
  --ann            HIVAU-70k 的标注 json(自带 clip/event 时间范围)
  --temporal-ann   UCF-Crime 官方 Temporal_Anomaly_Annotation.txt
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ds.common import extract_frames, image_size, iter_videos, video_duration
from scene import Event, Scene

DATASET = "HIVAU-70k"
LICENSE = "academic-research-only(含真实暴力内容, 注意隐私与脱敏)"

# 只有这些映射到本项目的异常类; 其余异常类型一律排除
ANOMALY_MAP: dict[str, tuple[str, str | None]] = {
    "explosion": ("explosion", None),
    "arson": ("explosion", None),
    "riot": ("massing", "personnel"),
}
# XD-Violence 文件名后缀编码: A=正常, G=爆炸, B1=打斗, B2=枪击, B4=骚乱, B5=虐待, B6=车祸
XD_CODE = {"a": "normal", "g": "explosion", "b1": "fighting", "b2": "shooting",
           "b4": "riot", "b5": "abuse", "b6": "car_accident"}
NORMAL_HINTS = ("normal", "normal_videos", "training_normal")


def _label_of(video: Path) -> str:
    """从文件名/父目录推断原始类别标签。"""
    stem, parent = video.stem.lower(), video.parent.name.lower()
    m = re.search(r"label_([abg]\d?)", stem)          # XD-Violence
    if m:
        codes = [c for c in re.findall(r"[abg]\d?", m.group(1))]
        for c in codes:
            lab = XD_CODE.get(c)
            if lab and lab != "normal":
                return lab
        return "normal"
    for h in NORMAL_HINTS:                            # UCF-Crime 正常视频
        if h in stem or h in parent:
            return "normal"
    m = re.match(r"([a-z]+)\d", stem)                 # UCF-Crime: Explosion001_x264
    if m:
        return m.group(1)
    return parent or "unknown"


def _load_spans_from_hivau(path: Path) -> dict[str, list[tuple[float, float]]]:
    """从 HIVAU 标注中抽取异常时间区间。官方结构可能演进, 这里按常见键名容错解析。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("data", raw.get("annotations", []))
    if isinstance(records, dict):
        records = [dict(v, video=k) for k, v in records.items()]

    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        if not isinstance(rec, dict):
            continue
        vid = rec.get("video") or rec.get("video_id") or rec.get("video_name") or rec.get("id")
        if not vid:
            continue
        key = Path(str(vid)).stem
        for kk in ("timestamp", "timestamps", "time", "span", "segment", "clip"):
            v = rec.get(kk)
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(
                    isinstance(x, (int, float)) for x in v):
                spans[key].append((float(v[0]), float(v[1])))
                break
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                spans[key].extend((float(a), float(b)) for a, b in v if b > a)
                break
    if not spans:
        print("[warn] HIVAU: 未从标注中解析出时间区间。请用 --inspect 查看结构, "
              "或改用 --temporal-ann 提供 UCF-Crime 官方时间标注")
    return spans


def _load_spans_from_ucf(path: Path, fps: float = 30.0) -> dict[str, list[tuple[float, float]]]:
    """UCF-Crime Temporal_Anomaly_Annotation.txt:
    <video> <class> <start1> <end1> <start2> <end2>   (帧号, -1 表示无)"""
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = ln.split()
        if len(p) < 4:
            continue
        key = Path(p[0]).stem
        nums = [int(float(x)) for x in p[2:] if x.lstrip("-").isdigit()]
        for i in range(0, len(nums) - 1, 2):
            s, e = nums[i], nums[i + 1]
            if s >= 0 and e > s:
                spans[key].append((s / fps, e / fps))
    return spans


def _complement(spans: list[tuple[float, float]], dur: float,
                margin: float = 2.0) -> list[tuple[float, float]]:
    """异常区间的补集 = 正常段。两侧留 margin 秒安全边距, 避免采到过渡帧。"""
    out, cur = [], 0.0
    for s, e in sorted(spans):
        if s - margin > cur:
            out.append((cur, s - margin))
        cur = max(cur, e + margin)
    if dur - cur > 1.0:
        out.append((cur, dur))
    return [(a, b) for a, b in out if b - a >= 1.0]


def inspect(ann: str, n: int = 3) -> None:
    """打印标注结构, 用于确认字段名。格式与预期不符时先跑这个。"""
    raw = json.loads(Path(ann).read_text(encoding="utf-8"))
    print(f"顶层类型: {type(raw).__name__}")
    records = raw if isinstance(raw, list) else raw.get("data", raw.get("annotations", raw))
    if isinstance(records, dict):
        items = list(records.items())[:n]
        print(f"字典, {len(records)} 条。前 {n} 条:")
        for k, v in items:
            print(f"  key={k!r}  value_keys={list(v)[:12] if isinstance(v, dict) else type(v).__name__}")
    elif isinstance(records, list):
        print(f"列表, {len(records)} 条。前 {n} 条的键:")
        for rec in records[:n]:
            print(f"  {list(rec)[:12] if isinstance(rec, dict) else type(rec).__name__}")
            if isinstance(rec, dict):
                print(f"    样例: {json.dumps(rec, ensure_ascii=False)[:400]}")


def build(video_root: str, frames_dir: str, ann: str | None = None,
          temporal_ann: str | None = None, fps_anomaly: float = 2.0,
          n_normal_frames: int = 4, ucf_fps: float = 30.0,
          view: str = "cctv") -> list[Scene]:
    spans: dict[str, list[tuple[float, float]]] = {}
    if ann:
        spans.update(_load_spans_from_hivau(Path(ann)))
    if temporal_ann:
        for k, v in _load_spans_from_ucf(Path(temporal_ann), ucf_fps).items():
            spans.setdefault(k, []).extend(v)

    scenes: list[Scene] = []
    stat = {"anomaly_frames": 0, "hard_negative_frames": 0, "normal_frames": 0,
            "skipped_no_span": 0, "excluded_type": 0}

    for vid in iter_videos(video_root):
        label = _label_of(vid)
        key = vid.stem
        is_normal = label == "normal"
        mapped = ANOMALY_MAP.get(label)

        if is_normal:
            frames = extract_frames(vid, Path(frames_dir) / "normal",
                                    n_frames=n_normal_frames, prefix=f"normal__{key}")
            for fi, fp in enumerate(frames):
                w, h = image_size(fp)
                scenes.append(Scene(
                    image_id=f"{DATASET}_normal_{key}_f{fi}", image_path=str(fp),
                    width=w, height=h, source_dataset=DATASET, license=LICENSE,
                    view=view, meta={"src_video": vid.name, "src_label": label}))
            stat["normal_frames"] += len(frames)
            continue

        vspans = spans.get(key) or spans.get(vid.name) or []
        if not vspans:
            stat["skipped_no_span"] += 1
            continue                                   # 没有时间边界的异常视频不能用

        dur = video_duration(vid) or (max(e for _, e in vspans) + 10.0)

        # 异常段: 仅当类型映射到我们的 4 类时才抽
        if mapped:
            kind, subtype = mapped
            for si, (s, e) in enumerate(vspans):
                frames = extract_frames(vid, Path(frames_dir) / kind, fps=fps_anomaly,
                                        span=(s, e), prefix=f"{label}__{key}__s{si}")
                for fp in frames:
                    w, h = image_size(fp)
                    ev = {"rule": None, "src_label": label, "span": [s, e]}
                    if subtype:
                        ev["subtype"] = subtype
                    scenes.append(Scene(
                        image_id=f"{DATASET}_{kind}_{key}_s{si}_{fp.stem}",
                        image_path=str(fp), width=w, height=h,
                        source_dataset=DATASET, license=LICENSE, view=view,
                        events=[Event(type=kind, conf=1.0, evidence=ev)],
                        meta={"src_video": vid.name, "src_label": label}))
                stat["anomaly_frames"] += len(frames)
        else:
            stat["excluded_type"] += 1

        # 正常段: 无论异常类型是否被采用, 都是同场景困难负样本
        for ni, (s, e) in enumerate(_complement(vspans, dur)):
            frames = extract_frames(vid, Path(frames_dir) / "hard_negative",
                                    n_frames=max(1, n_normal_frames // 2), span=(s, e),
                                    prefix=f"hn_{label}__{key}__n{ni}")
            for fp in frames:
                w, h = image_size(fp)
                scenes.append(Scene(
                    image_id=f"{DATASET}_hn_{key}_n{ni}_{fp.stem}", image_path=str(fp),
                    width=w, height=h, source_dataset=DATASET, license=LICENSE, view=view,
                    meta={"src_video": vid.name, "src_label": label,
                          "hard_negative_source": label, "span": [s, e]}))
            stat["hard_negative_frames"] += len(frames)

    print(f"[{DATASET}] 异常帧 {stat['anomaly_frames']} / "
          f"同场景困难负样本 {stat['hard_negative_frames']} / 正常视频帧 {stat['normal_frames']}")
    if stat["excluded_type"]:
        print(f"  排除 {stat['excluded_type']} 个视频: 异常类型(打斗/抢劫/盗窃等)"
              f"不属于本项目定义的 4 类, 但其正常段仍已采集为负样本")
    if stat["skipped_no_span"]:
        print(f"  [warn] 跳过 {stat['skipped_no_span']} 个异常视频: 缺少时间边界。"
              f"请提供 --ann 或 --temporal-ann, 否则抽出的帧会大面积错标")
    return scenes


def export_instructions(ann: str, out: str, video_key: str = "video",
                        prompt_keys: tuple[str, ...] = ("instruction", "question", "prompt"),
                        answer_keys: tuple[str, ...] = ("output", "answer", "response", "caption"),
                        frames_index: str | None = None) -> int:
    """把 HIVAU 自带的指令标注直接导出为 ShareGPT，省掉这部分的生成成本。

    frames_index: 由 build() 产出的 scenes jsonl，用于把 video 关联到已抽好的帧。
    """
    raw = json.loads(Path(ann).read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("data", raw.get("annotations", []))
    if isinstance(records, dict):
        records = [dict(v, **{video_key: k}) for k, v in records.items()]

    frame_of: dict[str, str] = {}
    if frames_index:
        for ln in Path(frames_index).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                frame_of.setdefault(Path(d["meta"].get("src_video", "")).stem, d["image_path"])

    out_items: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        vid = Path(str(rec.get(video_key, ""))).stem
        q = next((rec[k] for k in prompt_keys if rec.get(k)), None)
        a = next((rec[k] for k in answer_keys if rec.get(k)), None)
        if not q or not a:
            continue
        img = frame_of.get(vid)
        if not img:
            continue
        out_items.append({
            "messages": [{"role": "user", "content": "<image>" + str(q)},
                         {"role": "assistant", "content": str(a)}],
            "images": [img],
            "extra": {"image_id": f"{DATASET}_{vid}", "qa_type": "hivau_native",
                      "instruction_style": "direct", "anomaly": ["unknown"],
                      "source_dataset": DATASET, "license": LICENSE},
        })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{DATASET}] 导出原生指令 {len(out_items)} 条 -> {out}")
    if not out_items:
        print("  未导出任何条目: 请先用 inspect 确认字段名, 再用 --video-key/--prompt-keys 指定")
    return len(out_items)

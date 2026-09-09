"""从 Scene 生成多模态 VQA 样本(LLaMA-Factory ShareGPT 格式)。

9 类题型: 判定 / 分类 / 计数 / 定位 / 属性 / 空间关系 / 描述 / 推理 / 否定拒答。
其中前 6 类与第 9 类完全由标注规则生成, 事实一致性有保证;
描述与推理题这里给出"带标注约束的骨架答案", 建议再送 VLM 润色(见 --emit-vlm-prompts)。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds.boxes import COORD_MODE, box_json  # noqa: E402
from scene import Obj, Scene, load_scenes  # noqa: E402

BOX_SCALE = 1000       # grounding 坐标归一化上限; 换基座时只改这里


# ---------------------------------------------------------------- 模板
JUDGE_Q = [
    "画面中是否存在异常军事活动？",
    "这张航拍图里有没有值得警戒的异常情况？",
    "请判断当前监控画面是否出现异常，并说明依据。",
    "从态势感知角度看，该画面是否需要上报？",
]
CLASSIFY_Q = [
    "画面中出现的异常属于哪一类？",
    "请识别图中异常活动的类型。",
    "这属于什么性质的异常事件？",
]
DESCRIBE_Q = [
    "请描述这张画面中的情况。",
    "用一段话概括画面的态势要点。",
    "作为侦察情报，请对该画面进行简要描述。",
]
REASON_Q = {
    "massing": "为什么判断画面中存在兵力集结？请给出判定依据。",
    "crowd_gathering": "画面中的人员聚集有哪些可观察到的特征？",
    "convoy": "为什么认为这是车队机动而不是普通停车？",
    "border_crossing": "请说明越界行为的发生方向与判定依据。",
    "explosion": "请说明判定为爆炸的视觉依据。",
    "smoke": "画面中的烟雾有什么特征？可能意味着什么？",
    "fortification": "画面中的施工痕迹说明了什么？",
    "air_activity": "机场区域的活动是否异常？依据是什么？",
    "naval_massing": "为什么判断存在舰船集结？",
    "normal": "画面中为什么判定为无异常？",
}
DIRECTION_ZH = {"enter": "由外部进入禁区", "exit": "由禁区向外撤出",
                "-1->1": "自边界线一侧越至另一侧", "1->-1": "自边界线一侧越至另一侧"}


def _norm_box(bbox: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [max(0, min(BOX_SCALE, round(x1 / w * BOX_SCALE))),
            max(0, min(BOX_SCALE, round(y1 / h * BOX_SCALE))),
            max(0, min(BOX_SCALE, round(x2 / w * BOX_SCALE))),
            max(0, min(BOX_SCALE, round(y2 / h * BOX_SCALE)))]


def _quadrant(obj: Obj, w: int, h: int) -> str:
    cx, cy = obj.center
    v = "上" if cy < h / 3 else ("下" if cy > h * 2 / 3 else "中")
    hz = "左" if cx < w / 3 else ("右" if cx > w * 2 / 3 else "中")
    if v == "中" and hz == "中":
        return "画面中央"
    return f"画面{v}{hz}方"


class QABuilder:
    def __init__(self, ontology: dict[str, Any], seed: int = 0):
        self.onto = ontology
        self.zh = {c["id"]: c["zh"] for c in ontology["classes"]}
        self.cues = {c["id"]: c.get("cues", []) for c in ontology["classes"]}
        self.all_ids = [c["id"] for c in ontology["classes"]
                        if c["id"] != "normal" and c.get("enabled", True)]
        self.rng = random.Random(seed)

    # -------------------------------------------------- 单题型
    def q_judgement(self, s: Scene) -> list[dict]:
        types = s.anomaly_types
        if types:
            names = "、".join(self.zh[t] for t in types if t in self.zh)
            ev = s.events[0].evidence
            detail = f"，共涉及约 {ev['count']} 个目标" if "count" in ev else ""
            a = f"是。画面中存在异常：{names}{detail}。"
        else:
            a = "否。画面中未发现集结、烟火、越界等异常迹象，属于正常态势。"
        return [self._mk(s, self.rng.choice(JUDGE_Q), a, "judgement")]

    def q_classify(self, s: Scene) -> list[dict]:
        types = [t for t in s.anomaly_types if t in self.zh]
        if not types:
            return []
        opts = list({*types, *self.rng.sample(self.all_ids, k=min(3, len(self.all_ids)))})
        self.rng.shuffle(opts)
        q = (self.rng.choice(CLASSIFY_Q) + "\n候选："
             + "、".join(self.zh[o] for o in opts))
        return [self._mk(s, q, "、".join(self.zh[t] for t in types) + "。", "classification")]

    def q_count(self, s: Scene) -> list[dict]:
        by_cls: dict[str, int] = {}
        for o in s.objects:
            by_cls[o.cls] = by_cls.get(o.cls, 0) + 1
        out = []
        for cls, n in sorted(by_cls.items(), key=lambda kv: -kv[1])[:2]:
            out.append(self._mk(s, f"画面中可见多少个 {cls}？", f"{n} 个。", "counting"))
        return out

    def q_grounding(self, s: Scene) -> list[dict]:
        cluster_ev = next((e for e in s.events if "cluster_bbox" in e.evidence), None)
        out = []
        if cluster_ev:
            box = _norm_box(cluster_ev.evidence["cluster_bbox"], s.width, s.height)
            out.append(self._mk(
                s, f"请框出画面中的{self.zh[cluster_ev.type]}区域，给出坐标。",
                box_json(box, self.zh[cluster_ev.type]), "grounding"))
        if s.objects:
            o = max(s.objects, key=lambda x: x.area)
            out.append(self._mk(
                s, f"请给出画面中最显著的 {o.cls} 的边界框坐标。",
                box_json(_norm_box(o.bbox, s.width, s.height), o.cls), "grounding"))
        return out

    def q_attribute(self, s: Scene) -> list[dict]:
        if not s.objects:
            return []
        o = self.rng.choice(s.objects)
        moving = o.attrs.get("moving")
        if moving is None:
            return []
        state = "处于移动状态" if moving else "处于静止状态"
        return [self._mk(s, f"位于{_quadrant(o, s.width, s.height)}的 {o.cls} 是静止还是移动的？",
                         f"该 {o.cls} {state}。", "attribute")]

    def q_spatial(self, s: Scene) -> list[dict]:
        out = []
        for e in s.events:
            if e.evidence.get("rule") == "boundary_cross":
                d = DIRECTION_ZH.get(str(e.evidence.get("direction")), "跨越了边界")
                out.append(self._mk(
                    s, f"目标相对于「{e.evidence['region']}」的位置关系发生了什么变化？",
                    f"该 {e.evidence.get('cls', '目标')} {d}，构成越界移动。", "spatial"))
                break
        if not out and s.objects:
            o = self.rng.choice(s.objects)
            out.append(self._mk(s, f"画面中的 {o.cls} 主要分布在什么位置？",
                                f"主要位于{_quadrant(o, s.width, s.height)}。", "spatial"))
        return out

    def q_describe(self, s: Scene) -> list[dict]:
        if s.caption:
            a = s.caption
        else:
            by_cls: dict[str, int] = {}
            for o in s.objects:
                by_cls[o.cls] = by_cls.get(o.cls, 0) + 1
            objs = "、".join(f"{n} 个 {c}" for c, n in sorted(by_cls.items(), key=lambda kv: -kv[1])[:3])
            view = {"uav": "无人机航拍", "satellite": "卫星遥感", "cctv": "地面监控", "ground": "地面"}.get(s.view, "航拍")
            if s.anomaly_types:
                names = "、".join(self.zh[t] for t in s.anomaly_types if t in self.zh)
                a = f"这是一张{view}画面，可见{objs or '若干目标'}。画面呈现{names}特征，建议持续观察并上报。"
            else:
                a = f"这是一张{view}画面，可见{objs or '少量常规目标'}，目标分布稀疏，未见异常活动迹象。"
        return [self._mk(s, self.rng.choice(DESCRIBE_Q), a, "description")]

    def q_reason(self, s: Scene) -> list[dict]:
        t = s.anomaly_types[0] if s.anomaly_types else "normal"
        q = REASON_Q.get(t)
        if not q:
            return []
        cues = "；".join(self.cues.get(t, [])) or "无明显异常线索"
        if t == "normal":
            a = f"依据：{cues}。目标数量与分布均处于常态范围，因此判定为无异常。"
        else:
            ev = s.events[0].evidence
            quant = f"规则量化结果为 {ev.get('count', ev.get('r2', 'N/A'))}。" if ev else ""
            a = f"判定依据：{cues}。{quant}综合以上特征，判定为{self.zh.get(t, t)}。"
        return [self._mk(s, q, a, "reasoning")]

    def q_negative(self, s: Scene) -> list[dict]:
        """问画面里没有的异常, 逼模型学会说'没有' —— 抗幻觉的关键题型。"""
        present = set(s.anomaly_types)
        absent = [c for c in self.all_ids if c not in present]
        if not absent:
            return []
        t = self.rng.choice(absent)
        return [self._mk(s, f"画面中是否出现了{self.zh[t]}？",
                         f"否。画面中未观察到{self.zh[t]}的迹象。", "negation")]

    # -------------------------------------------------- 组装
    def _mk(self, s: Scene, q: str, a: str, qa_type: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "<image>" + q},
                {"role": "assistant", "content": a},
            ],
            "images": [s.image_path],
            "extra": {
                "image_id": s.image_id,
                "qa_type": qa_type,
                "anomaly": s.anomaly_types or ["normal"],
                "source_dataset": s.source_dataset,
                "license": s.license,
                "view": s.view,
                "image_width": s.width,
                "image_height": s.height,
                "coordinate_mode": COORD_MODE,
                "bbox_scale": BOX_SCALE,
            },
        }

    def build(self, s: Scene) -> list[dict]:
        out: list[dict] = []
        for fn in (self.q_judgement, self.q_classify, self.q_count, self.q_grounding,
                   self.q_attribute, self.q_spatial, self.q_describe, self.q_reason,
                   self.q_negative):
            out.extend(fn(s))
        return out


# ---------------------------------------------------------------- 切分
def split_by_group(samples: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 0):
    """按来源图/视频分组切分, 避免同源相邻帧跨集造成指标虚高。"""
    groups: dict[str, list[dict]] = {}
    for s in samples:
        gid = s["extra"]["image_id"].rsplit("_frame", 1)[0]
        groups.setdefault(gid, []).append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    parts = {"train": keys[:a], "val": keys[a:b], "test": keys[b:]}
    return {k: [s for g in v for s in groups[g]] for k, v in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Scene -> VQA(ShareGPT) 数据生成")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out-dir", default="data/vqa")
    ap.add_argument("--ontology", default="configs/ontology.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-split", action="store_true", help="不切分, 只输出单个 all.json")
    ap.add_argument("--exclude-ids", default=None,
                    help="golden set 的 image_id 清单, 必须排除否则泄漏(make_golden.py 产出)")
    args = ap.parse_args()

    onto = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    global BOX_SCALE
    BOX_SCALE = int(onto.get("box_scale", BOX_SCALE))

    builder = QABuilder(onto, seed=args.seed)
    scenes = load_scenes(args.scenes)
    if args.exclude_ids:
        excl = {ln.strip() for ln in Path(args.exclude_ids).read_text(encoding="utf-8").splitlines() if ln.strip()}
        before = len(scenes)
        scenes = [s for s in scenes if s.image_id not in excl]
        print(f"排除 golden set: {before} -> {len(scenes)} 个 scene")
    samples = [qa for s in scenes for qa in builder.build(s)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, data: list[dict]) -> None:
        (out_dir / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {name}.json  {len(data)} 条")

    if args.no_split:
        _write("all", samples)
    else:
        for name, part in split_by_group(samples, seed=args.seed).items():
            _write(name, part)

    hist: dict[str, int] = {}
    for s in samples:
        hist[s["extra"]["qa_type"]] = hist.get(s["extra"]["qa_type"], 0) + 1
    n_norm = sum(1 for s in samples if s["extra"]["anomaly"] == ["normal"])
    print(f"\n共 {len(samples)} 条 QA / {len(scenes)} 个 scene")
    print(f"正常样本占比 {n_norm / max(1, len(samples)):.1%}"
          f"（目标 ≥{onto.get('negative_ratio_target', 0.3):.0%}）")
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:6d}  {v / len(samples):.1%}")


if __name__ == "__main__":
    main()

"""规则生成：判定 / 方位指代 / 坐标框 / 计数（含否定变体）。

这四类任务的答案**由标注唯一决定**，所以走规则不走 LLM —— 零成本、零幻觉。
描述与推理需要自然语言，走 llm_qa.py。

反同质化按 docs/14 的四维矩阵采样：轮数、图数、问法风格、答案形态。
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
from ds.boxes import BBOX_SCALE, COORD_MODE, box_json, boxes_json, to_bbox2d  # noqa: E402
from facets import load_all, load_tool  # noqa: E402
from scene import Obj, Scene, load_scenes  # noqa: E402

# 任务配额(占该异常类总量)。描述与推理由 llm_qa 产出, 这里只列规则侧。
RULE_QUOTA = {"judge": 0.15, "locate": 0.12, "count": 0.05}
LOCATE_VERBAL_RATIO = 0.70      # 定位任务里方位指代占七成, 坐标占三成
NEGATION_RATIO = 0.20           # 每类任务的否定变体占比
TWO_TURN_RATIO = 0.30           # 规则侧的双轮占比

MEASURE = {"military-plane": "架", "civil-plane": "架", "plane": "架",
           "military-helicopter": "架", "civil-helicopter": "架", "helicopter": "架",
           "tank": "辆", "vehicle": "辆", "large-vehicle": "辆", "truck": "辆",
           "military-vehicle": "辆", "military-truck": "辆", "armored": "辆",
           "person": "名", "soldier": "名", "drone": "架",
           "ship": "艘", "warship": "艘"}
CLS_ZH = {"military-plane": "军用飞机", "civil-plane": "民航飞机", "tank": "坦克",
          "vehicle": "车辆", "large-vehicle": "大型车辆", "truck": "卡车",
          "person": "人员", "soldier": "士兵", "ship": "船只", "warship": "军舰",
          "drone": "无人机", "military-vehicle": "军用车辆", "fire": "火焰", "smoke": "烟雾"}


# ---------------------------------------------------------------- 方位表述
_TENTH = ["", "一成", "两成", "三成", "四成", "五成", "六成", "七成", "八成", "九成"]


def _frac(v: float) -> str:
    """0~1 -> 自然的中文比例说法。"""
    if v <= 0.06:
        return "最左端" if v == 0 else "近左缘"
    if v >= 0.94:
        return "近右缘"
    if abs(v - 0.5) < 0.04:
        return "正中"
    if abs(v - 1 / 3) < 0.04:
        return "三分之一处"
    if abs(v - 2 / 3) < 0.04:
        return "三分之二处"
    if abs(v - 0.25) < 0.04:
        return "四分之一处"
    if abs(v - 0.75) < 0.04:
        return "四分之三处"
    return _TENTH[max(1, min(9, round(v * 10)))]


def _zone(cx: float, cy: float) -> str:
    v = "上" if cy < 1 / 3 else ("下" if cy > 2 / 3 else "中")
    h = "左" if cx < 1 / 3 else ("右" if cx > 2 / 3 else "中")
    if v == "中" and h == "中":
        return "画面中央"
    if v == "中":
        return f"画面{h}侧居中"
    if h == "中":
        return f"画面{v}部偏中"
    return f"画面{v}{h}方" if (v, h) not in (("上", "左"), ("上", "右"), ("下", "左"), ("下", "右")) \
        else f"画面{v}{h}角"


def _span(a: float, b: float, axis: str) -> str:
    """把一段区间说成人话。区间很窄时说成一个点, 不要写"从正中到正中"。"""
    lead = "从左往右" if axis == "x" else "从上往下"
    if b - a < 0.10:
        return f"{lead}约{_frac((a + b) / 2)}"
    return f"{lead}{_frac(a)}到{_frac(b)}之间"


def position_text(bbox: list[float], W: int, H: int, granularity: str, rng: random.Random) -> str:
    """把 bbox 转成自然语言方位。三档粒度, 由采样决定用哪档。**不输出坐标。**"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    area = ((x2 - x1) * (y2 - y1)) / (W * H)
    occupy = ("占画幅很小一块" if area < 0.03 else
              "约占画幅的十分之一" if area < 0.12 else
              "约占画幅的五分之一" if area < 0.25 else
              "约占画幅的三分之一" if area < 0.42 else "占据画幅一半以上")

    if granularity == "zone":
        return f"在{_zone(cx, cy)}，{occupy}。"
    if granularity == "ratio":
        return (f"横向{_span(x1 / W, x2 / W, 'x')}，"
                f"纵向{_span(y1 / H, y2 / H, 'y')}，{occupy}。")
    # landmark: 只有确实靠边时才用边缘参照, 否则退回 zone+占比, 免得写出"距上边缘"这种残句
    edges = sorted([("左边缘", x1 / W), ("右边缘", 1 - x2 / W),
                    ("上边缘", y1 / H), ("下边缘", 1 - y2 / H)], key=lambda kv: kv[1])
    near, d = edges[0]
    if d >= 0.18:
        return f"位于{_zone(cx, cy)}，四周均留有空白，{occupy}。"
    far = edges[-1][0]
    prox = "紧贴" if d < 0.05 else "靠近"
    tail = rng.choice([f"，与画面{far}之间留有较大空白", "", f"，距画面{far}较远"])
    return f"{prox}画面{near}，位于{_zone(cx, cy)}{tail}，{occupy}。"


# ---------------------------------------------------------------- 生成器
class RuleBuilder:
    def __init__(self, onto: dict[str, Any], prompt_dir: str, seed: int = 0):
        self.onto = onto
        self.zh = {c["id"]: c["zh"] for c in onto["classes"]}
        self.enabled = [c["id"] for c in onto["classes"]
                        if c["id"] != "normal" and c.get("enabled", True)]
        _, self.asks = load_all(prompt_dir)
        self.system = (Path(prompt_dir) / "system.txt").read_text(encoding="utf-8").strip()
        self.rng = random.Random(seed)

    # -------------------------------------------------- 小工具
    def _ask(self, task: str, **kw) -> str:
        q = self.rng.choice(self.asks[task].lines)
        for k, v in kw.items():
            q = q.replace("{" + k + "}", str(v))
        return q

    def _anomaly_zh(self, s: Scene) -> str:
        return "、".join(self.zh[t] for t in s.anomaly_types if t in self.zh) or "异常"

    def _cluster(self, s: Scene):
        return next((e for e in s.events if "cluster_bbox" in e.evidence), None)

    def _region_box(self, s: Scene) -> tuple[list[float], str] | None:
        """异常区域的像素 bbox 与标签。没有区域信息就返回 None。"""
        ev = self._cluster(s)
        if ev:
            return ev.evidence["cluster_bbox"], f"{self.zh.get(ev.type, ev.type)}区域"
        for e in s.events:
            if e.evidence.get("boxes"):
                bs = e.evidence["boxes"]
                xs = [b[0] for b in bs] + [b[2] for b in bs]
                ys = [b[1] for b in bs] + [b[3] for b in bs]
                return [min(xs), min(ys), max(xs), max(ys)], self.zh.get(e.type, e.type)
        cands = [o for o in s.objects if o.cls in ("fire", "smoke")]
        if cands:
            o = max(cands, key=lambda x: x.area)
            return o.bbox, CLS_ZH.get(o.cls, o.cls)
        # 越界场景没有 cluster_bbox, 用越界目标的外接框
        objs = self._crossed_objs(s)
        if objs:
            xs = [v for o in objs for v in (o.bbox[0], o.bbox[2])]
            ys = [v for o in objs for v in (o.bbox[1], o.bbox[3])]
            return [min(xs), min(ys), max(xs), max(ys)], "越界目标所在区域"
        return None

    def _crossed_objs(self, s: Scene) -> list[Obj]:
        tids = {str(e.evidence.get("track_id")) for e in s.events
                if e.evidence.get("rule") == "boundary_cross"}
        return [o for o in s.objects if o.track_id is not None and str(o.track_id) in tids]

    def _mk(self, s: Scene, turns: list[tuple[str, str]], task: str, **extra) -> dict:
        msgs: list[dict] = [{"role": "system", "content": self.system}]
        n_img = len(s.meta.get("frames", [])) or 1
        for i, (q, a) in enumerate(turns):
            msgs.append({"role": "user",
                         "content": ("<image>" * n_img + q) if i == 0 else q})
            msgs.append({"role": "assistant", "content": a})
        return {
            "messages": msgs,
            "images": s.meta.get("frames") or [s.image_path],
            "extra": {"image_id": s.image_id, "task": task, "gen": "rule",
                      "n_turns": len(turns), "n_images": n_img,
                      "anomaly": s.anomaly_types or ["normal"],
                      "hard_negative": bool(s.meta.get("hard_negative")),
                      "source_dataset": s.source_dataset, "license": s.license,
                      "view": s.view, "image_width": s.width, "image_height": s.height,
                      "coordinate_mode": COORD_MODE, "bbox_scale": BBOX_SCALE, **extra},
        }

    # -------------------------------------------------- 四类任务
    def judge(self, s: Scene) -> tuple[str, str]:
        q = self._ask("judge")
        if s.anomaly_types:
            ev = s.events[0].evidence
            n = ev.get("count")
            cls_hint = ""
            if n:
                main = max(((o.cls, sum(1 for x in s.objects if x.cls == o.cls))
                            for o in s.objects), key=lambda kv: kv[1], default=None)
                if main:
                    cls_hint = f"{n}{MEASURE.get(main[0], '个')}{CLS_ZH.get(main[0], main[0])}"
            lead = self.rng.choice(["存在异常，为", "画面中出现异常：", "判定为",
                                    "有异常。类型为"])
            tail = ""
            if cls_hint:
                tail = self.rng.choice([
                    f"画面中可见 {cls_hint} 密集分布。",
                    f"共观察到 {cls_hint}，成簇分布。",
                    f"涉及 {cls_hint}。",
                ])
            need = self.rng.choice(["", "建议上报并持续观察。", "建议持续观察。", ""])
            a = f"{lead}{self._anomaly_zh(s)}。{tail}{need}"
        elif s.meta.get("hard_negative"):
            by: dict[str, int] = {}
            for o in s.objects:
                by[o.cls] = by.get(o.cls, 0) + 1
            cls, n = max(by.items(), key=lambda kv: kv[1]) if by else ("目标", 0)
            what = f"{n}{MEASURE.get(cls, '个')}{CLS_ZH.get(cls, cls)}" if n else "若干目标"
            a = (f"未见异常。画面中虽有 {what} 密集成簇、达到了集结的规模条件，"
                 f"但均为民用目标，未见坦克、装甲车或军机等军事装备，属于正常场景。")
        else:
            a = self.rng.choice([
                "未见异常。目标分布稀疏，无烟火迹象，也没有跨越边界的移动目标。",
                "未见异常，态势正常，不需要上报。画面中没有成规模的目标聚集，"
                "也没有烟雾或火光。",
                "属于正常态势。逐项核查：无军事装备集结、无人员异常聚集、"
                "无烟火、无越界移动。",
                "没有发现需要关注的情况。画面中的目标数量与分布都在常态范围内。",
                "未见异常。该画面属于常规场景，无需进一步处置。",
            ])
        return q, a

    def locate_verbal(self, s: Scene) -> tuple[str, str] | None:
        rb = self._region_box(s)
        if rb is None:
            return None
        box, label = rb
        gran = self.rng.choices(["zone", "ratio", "landmark"], weights=[40, 35, 25])[0]
        q = self._ask("locate_verbal", zh=self._anomaly_zh(s))
        return q, position_text(box, s.width, s.height, gran, self.rng)

    def locate_box(self, s: Scene) -> tuple[str, str] | None:
        rb = self._region_box(s)
        if rb is None:
            return None
        box, label = rb
        q = self._ask("locate_box", zh=self._anomaly_zh(s))
        return q, box_json(to_bbox2d(box, s.width, s.height), label)

    def locate_box_multi(self, s: Scene) -> tuple[str, str] | None:
        """越界类: 多个目标一起框出。"""
        objs = self._crossed_objs(s)
        if not objs:
            return None
        q = self._ask("locate_box", zh="越界目标")
        return q, boxes_json([(to_bbox2d(o.bbox, s.width, s.height), "越界目标") for o in objs])

    def count(self, s: Scene) -> tuple[str, str] | None:
        by: dict[str, int] = {}
        for o in s.objects:
            by[o.cls] = by.get(o.cls, 0) + 1
        if not by:
            return None
        cls, n = max(by.items(), key=lambda kv: kv[1])
        mw = MEASURE.get(cls, "个")
        q = self._ask("count", mw=mw, label=CLS_ZH.get(cls, cls))
        return q, f"{n}{mw}。"

    def negation(self, s: Scene) -> tuple[str, str] | None:
        """否定变体。**只问本数据集里真实存在、这张图上恰好没有的异常类型** ——
        问航拍图有没有潜艇，不看图也知道没有，那种题 needs_image 维会被判低分。"""
        absent = [c for c in self.enabled if c not in s.anomaly_types]
        if not absent:
            return None
        t = self.rng.choice(absent)
        style = self.rng.random()
        if style < 0.45:
            return self._ask("negation", zh=self.zh[t]), f"否，未观察到{self.zh[t]}的迹象。"
        if style < 0.75:
            return (self._ask("locate_verbal", zh=self.zh[t]),
                    f"画面中未发现{self.zh[t]}，无从指出其方位。")
        cls = self.rng.choice(["tank", "warship", "military-plane"])
        if any(o.cls == cls for o in s.objects):
            return self._ask("negation", zh=self.zh[t]), f"否，未观察到{self.zh[t]}的迹象。"
        return (self._ask("count", mw=MEASURE.get(cls, "个"), label=CLS_ZH[cls]),
                f"0{MEASURE.get(cls, '个')}。")

    # -------------------------------------------------- 组装
    def build(self, s: Scene) -> list[dict]:
        out: list[dict] = []
        r = self.rng

        # 判定 —— 每张图必出
        jq, ja = self.judge(s)
        if r.random() < TWO_TURN_RATIO:
            second = self.locate_verbal(s) if r.random() < LOCATE_VERBAL_RATIO else self.locate_box(s)
            if second:
                out.append(self._mk(s, [(jq, ja), second], "judge+locate",
                                    form="multi_turn"))
            else:
                out.append(self._mk(s, [(jq, ja)], "judge"))
        else:
            out.append(self._mk(s, [(jq, ja)], "judge"))

        # 定位 —— 方位七成 / 坐标三成。取不到区域框的图(如无事件的正常图)自然跳过
        if r.random() < LOCATE_VERBAL_RATIO:
            if (t := self.locate_verbal(s)):
                out.append(self._mk(s, [t], "locate_verbal"))
        elif (t := self.locate_box_multi(s) or self.locate_box(s)):
            out.append(self._mk(s, [t], "locate_box"))

        # 计数
        if (t := self.count(s)):
            out.append(self._mk(s, [t], "count"))

        # 否定变体
        if r.random() < NEGATION_RATIO and (t := self.negation(s)):
            out.append(self._mk(s, [t], "negation"))
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
    ap = argparse.ArgumentParser(description="规则生成: 判定/方位/坐标/计数")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out-dir", default="data/vqa_rule")
    ap.add_argument("--ontology", default="configs/ontology.yaml")
    ap.add_argument("--prompt-dir", default="configs/prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--exclude-ids", default=None,
                    help="golden set 的 image_id 清单, 必须排除否则泄漏")
    args = ap.parse_args()

    onto = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    builder = RuleBuilder(onto, args.prompt_dir, seed=args.seed)
    scenes = load_scenes(args.scenes)
    if args.exclude_ids:
        excl = {ln.strip() for ln in Path(args.exclude_ids).read_text(encoding="utf-8").splitlines() if ln.strip()}
        before = len(scenes)
        scenes = [s for s in scenes if s.image_id not in excl]
        print(f"排除 golden set: {before} -> {len(scenes)} 个 scene")

    samples = [qa for s in scenes for qa in builder.build(s)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _write(name: str, data: list[dict]) -> None:
        (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        print(f"  {name}.json  {len(data)} 条")

    if args.no_split:
        _write("all", samples)
    else:
        for name, part in split_by_group(samples, seed=args.seed).items():
            _write(name, part)

    from collections import Counter
    tasks = Counter(s["extra"]["task"] for s in samples)
    turns = Counter(s["extra"]["n_turns"] for s in samples)
    n_norm = sum(1 for s in samples if s["extra"]["anomaly"] == ["normal"])
    print(f"\n共 {len(samples)} 条 / {len(scenes)} 个 scene")
    print(f"正常样本占比 {n_norm / max(1, len(samples)):.1%}"
          f"（目标 ≥{onto.get('negative_ratio_target', 0.3):.0%}）")
    print("  任务:", dict(tasks.most_common()))
    print("  轮数:", {f"{k}轮": v for k, v in sorted(turns.items())})


if __name__ == "__main__":
    main()

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
from derive_events import _side  # noqa: E402  越界方向判定, 与派生端共用一份实现
from facets import load_all, load_tool  # noqa: E402
from scene import Obj, Scene, load_scenes  # noqa: E402

# 任务配额(占该异常类总量)。描述与推理由 llm_qa 产出, 这里只列规则侧。
RULE_QUOTA = {"judge": 0.15, "locate": 0.12, "count": 0.05}
LOCATE_VERBAL_RATIO = 0.70      # 定位任务里方位指代占七成, 坐标占三成
NEGATION_RATIO = 0.20           # 每类任务的否定变体占比
TWO_TURN_RATIO = 0.30           # 规则侧的多轮占比
THREE_TURN_RATIO = 0.35         # 多轮里再有三成半追到第三轮("依据是什么")
COUNT_BOX_RATIO = 0.25          # 计数题里四分之一要求逐个框出
COMPOSE_RATIO = 0.30            # 目标构成(哪几类各多少)
COMPARE_RATIO = 0.25            # 左右/上下密度对比
CORRECT_RATIO = 0.30            # 纠错题(给错误陈述让模型推翻)
TEMPORAL_RATIO = 0.60           # 时序题。只有带轨迹的视频/多帧能出, 条件本就少, 配比给高

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


def _region_zh(ev: dict[str, Any]) -> str:
    """禁区名往往是 restricted_zone_A 这种英文标识, 直接塞进中文答案会很突兀。
    有中文名就用中文名, 没有就按区域类型给一个通名。"""
    name = str(ev.get("region") or "")
    if name and any("\u4e00" <= ch <= "\u9fff" for ch in name):
        return name
    return "禁区边界" if ev.get("region_type") == "polygon" else "界线"


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
        field, paths = s.media
        tok = "<video>" if field == "videos" else "<image>"
        for i, (q, a) in enumerate(turns):
            msgs.append({"role": "user",
                         "content": (tok * len(paths) + q) if i == 0 else q})
            msgs.append({"role": "assistant", "content": a})
        return {
            "messages": msgs,
            field: paths,
            "extra": {"image_id": s.image_id, "task": task, "gen": "rule",
                      "n_turns": len(turns), "modality": s.modality,
                      "n_media": len(paths),
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
            # 放宽档补出来的事件措辞要留余地: 它本来就是够不上严格阈值才被捡回来的,
            # 用"判定为集结"这种确定语气去训, 等于教模型把小规模聚集当成集结
            if s.events and all(e.evidence.get("relaxed") for e in s.events):
                lead = self.rng.choice(["存在需要留意的迹象，倾向于", "有轻度异常迹象，疑似",
                                        "初步判断为", "存在苗头，可能属于"])
            else:
                lead = self.rng.choice(["存在异常，为", "画面中出现异常：", "判定为",
                                        "有异常。类型为"])
            tail = ""
            if cls_hint:
                tail = self.rng.choice([
                    f"画面中可见 {cls_hint} 密集分布。",
                    f"共观察到 {cls_hint}，成簇分布。",
                    f"涉及 {cls_hint}。",
                ])
            if s.events and all(e.evidence.get("relaxed") for e in s.events):
                need = self.rng.choice(["规模有限，建议继续观察确认。", "尚未达到典型规模，建议复核。",
                                        "证据强度一般，建议结合后续画面判断。"])
            else:
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
            named = [ln for ln in self.asks["locate_verbal"].lines if "{zh}" in ln]
            if named:
                q = self.rng.choice(named).replace("{zh}", self.zh[t])
                return q, f"画面中未发现{self.zh[t]}，无从指出其方位。"
        cls = self.rng.choice(["tank", "warship", "military-plane"])
        if any(o.cls == cls for o in s.objects):
            return self._ask("negation", zh=self.zh[t]), f"否，未观察到{self.zh[t]}的迹象。"
        return (self._ask("count", mw=MEASURE.get(cls, "个"), label=CLS_ZH[cls]),
                f"0{MEASURE.get(cls, '个')}。")

    # -------------------------------------------------- 扩充题型
    def _class_counts(self, s: Scene) -> list[tuple[str, int]]:
        by: dict[str, int] = {}
        for o in s.objects:
            by[o.cls] = by.get(o.cls, 0) + 1
        return sorted(by.items(), key=lambda kv: -kv[1])

    def _phrase(self, cls: str, n: int) -> str:
        return f"{CLS_ZH.get(cls, cls)} {n} {MEASURE.get(cls, '个')}"

    def compose(self, s: Scene) -> tuple[str, str] | None:
        """目标构成: 哪几类、各多少。答案逐类由标注算出, 不做任何推断。"""
        cc = self._class_counts(s)
        if len(cc) < 2:
            return None                      # 只有一类就退化成 count 了
        total = sum(n for _, n in cc)
        body = "；".join(self._phrase(c, n) for c, n in cc[:6])
        tail = "。" if len(cc) <= 6 else f"；其余 {len(cc) - 6} 类数量较少。"
        return self._ask("compose"), f"画面中共 {total} 个目标：{body}{tail}"

    def compare(self, s: Scene) -> tuple[str, str] | None:
        """左右(或上下)对比。答案是数出来的, 差距在一成以内如实说"基本均衡"。"""
        if len(s.objects) < 4:
            return None
        axis = self.rng.choice(["h", "v"])
        if axis == "h":
            a_zh, b_zh, mid = "左半区", "右半区", s.width / 2
            na = sum(1 for o in s.objects if o.center[0] < mid)
        else:
            a_zh, b_zh, mid = "上半区", "下半区", s.height / 2
            na = sum(1 for o in s.objects if o.center[1] < mid)
        nb = len(s.objects) - na
        q = self._ask("compare", a=a_zh, b=b_zh)
        if abs(na - nb) <= max(1, round(len(s.objects) * 0.1)):
            return q, f"两侧基本均衡：{a_zh} {na} 个目标，{b_zh} {nb} 个，数量接近。"
        hi, lo = (a_zh, b_zh) if na > nb else (b_zh, a_zh)
        return q, f"{hi}更密集：{a_zh} {na} 个目标，{b_zh} {nb} 个，主要集中在{hi}。"

    def correct(self, s: Scene) -> tuple[str, str] | None:
        """纠错题: 给一句陈述让模型判断真伪。

        **三成给的是真陈述**。全给假的, 模型会学成"凡是被问就否定",
        换个正确说法它照样推翻, 这比附和还糟。
        """
        cc = self._class_counts(s)
        if not cc:
            return None
        cls, n = cc[0]
        mw, zh = MEASURE.get(cls, "个"), CLS_ZH.get(cls, cls)
        if self.rng.random() < 0.3:                       # 真陈述
            claim = f"画面中有 {n} {mw}{zh}"
            return self._ask("correct", claim=claim), f"说法属实，画面中确为 {n} {mw}{zh}。"
        style = self.rng.random()
        if style < 0.5:                                   # 数量错
            delta = self.rng.choice([-3, -2, -1, 1, 2, 3, 5])
            wrong = max(0, n + delta)
            if wrong == n:
                wrong = n + 1
            claim = f"画面中有 {wrong} {mw}{zh}"
            return (self._ask("correct", claim=claim),
                    f"不对。{zh}的数量是 {n} {mw}，不是 {wrong} {mw}。")
        if style < 0.8 and self.enabled:                  # 异常类型错
            absent = [c for c in self.enabled if c not in s.anomaly_types]
            if absent:
                t = self.rng.choice(absent)
                claim = f"这张画面里正在发生{self.zh[t]}"
                real = (f"实际的情况是{self._anomaly_zh(s)}" if s.anomaly_types
                        else "画面属于正常态势，没有异常")
                return (self._ask("correct", claim=claim),
                        f"不对，未观察到{self.zh[t]}的迹象。{real}。")
        rb = self._region_box(s)                          # 方位错
        if rb is None:
            return None
        box, _ = rb
        cx = (box[0] + box[2]) / 2
        said = "右侧" if cx < s.width / 2 else "左侧"
        real = "左侧" if said == "右侧" else "右侧"
        claim = f"重点区域在画面{said}"
        return (self._ask("correct", claim=claim),
                f"不对，方位说反了。该区域位于画面{real}。")

    def count_box(self, s: Scene) -> tuple[str, str] | None:
        """计数 + 逐个框出。目标太多时不出这题 —— 框二十几个必然有错漏。"""
        cc = self._class_counts(s)
        if not cc:
            return None
        cls, n = cc[0]
        if n > 12:
            return None
        objs = [o for o in s.objects if o.cls == cls]
        mw, zh = MEASURE.get(cls, "个"), CLS_ZH.get(cls, cls)
        q = self._ask("count_box", mw=mw, label=zh)
        body = boxes_json([(to_bbox2d(o.bbox, s.width, s.height), zh) for o in objs])
        return q, f"共 {n} {mw}{zh}。\n{body}"

    def temporal(self, s: Scene) -> tuple[str, str] | None:
        """时序题。**只出在视频/多帧上, 且只答轨迹能证明的事** ——
        静态图问"什么时候开始的"必然只能靠编。"""
        if s.modality not in ("video", "multi_image") or not s.tracks:
            return None
        cross = [e for e in s.events if e.evidence.get("rule") == "boundary_cross"]
        if not cross:
            return None
        fracs = []
        for e in cross:
            pts = s.tracks.get(str(e.evidence.get("track_id")))
            if not pts or len(pts) < 2:
                continue
            path = sorted(pts, key=lambda p: p[0])
            reg = next((r for r in s.regions if r.name == e.evidence.get("region")), None)
            if reg is None or len(reg.points) < 2:
                continue
            a, b = reg.points[0], reg.points[-1]
            s0 = _side((path[0][1], path[0][2]), a, b)
            for i, p in enumerate(path):
                if _side((p[1], p[2]), a, b) != s0:
                    fracs.append(i / max(1, len(path) - 1))
                    break
        if not fracs:
            return None
        f = sum(fracs) / len(fracs)
        stage = "前段" if f < 0.34 else ("中段" if f < 0.67 else "后段")
        who = CLS_ZH.get(cross[0].evidence.get("cls") or "", "目标")
        n = len(fracs)
        q = self._ask("temporal")
        more = f"共 {n} 个目标先后越过。" if n > 1 else ""
        return q, (f"越界发生在序列的{stage}（约第 {round(f * 100)}% 处）。"
                   f"序列开始时{who}还在界线一侧，随后移动并跨过界线。{more}")

    def why(self, s: Scene) -> str | None:
        """多轮里的第三轮"依据是什么"。只复述证据字段, 不做任何延伸判断。"""
        ev = self._cluster(s)
        if ev:
            n = ev.evidence.get("count")
            cls = "、".join(CLS_ZH.get(c, c) for c in ev.evidence.get("classes", [])[:3])
            if ev.evidence.get("relaxed"):
                return (f"依据是目标的空间密度：{n} 个{cls or '目标'}聚成一簇，间距小于周边，"
                        f"但规模不大，只能算{self.zh.get(ev.type, '该类异常')}的迹象，"
                        f"还不足以下确定结论。")
            return (f"依据是目标的空间密度：{n} 个{cls or '目标'}的间距明显小于画面中"
                    f"其他区域，聚成一簇，规模已达到{self.zh.get(ev.type, '该类异常')}的判定条件。")
        cross = next((e for e in s.events if e.evidence.get("rule") == "boundary_cross"), None)
        if cross:
            reg = _region_zh(cross.evidence)
            return (f"依据是目标的运动轨迹：其路径与{reg}相交，"
                    f"起点与终点分处界线两侧，构成跨越。")
        if s.meta.get("hard_negative"):
            rs = s.meta.get("hard_negative_reason") or []
            return ("依据是目标性质：" + (rs[0] if rs else "成簇目标均为民用，不构成军事集结") +
                    "，因此不按异常处置。")
        if not s.anomaly_types:
            return "依据是逐项排查的结果：目标分布稀疏、无烟火迹象、无跨界移动，各项均未触发。"
        return None

    # -------------------------------------------------- 组装
    def build(self, s: Scene) -> list[dict]:
        out: list[dict] = []
        r = self.rng

        # 判定 —— 每张图必出。三成走多轮, 多轮里再分两轮/三轮
        jq, ja = self.judge(s)
        if r.random() < TWO_TURN_RATIO:
            second = self.locate_verbal(s) if r.random() < LOCATE_VERBAL_RATIO else self.locate_box(s)
            if second:
                turns = [(jq, ja), second]
                task = "judge+locate"
                if r.random() < THREE_TURN_RATIO and (w := self.why(s)):
                    turns.append((self._ask("why"), w))
                    task = "judge+locate+why"
                out.append(self._mk(s, turns, task, form="multi_turn"))
            elif (w := self.why(s)):
                out.append(self._mk(s, [(jq, ja), (self._ask("why"), w)],
                                    "judge+why", form="multi_turn"))
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

        # 计数 —— 少数走"计数+框出", 与纯计数错开
        if r.random() < COUNT_BOX_RATIO and (t := self.count_box(s)):
            out.append(self._mk(s, [t], "count_box"))
        elif (t := self.count(s)):
            out.append(self._mk(s, [t], "count"))

        # 构成 / 对比 / 纠错 / 时序 —— 各按配比抽, 抽不到条件就跳过, 不硬凑
        for ratio, fn, name in ((COMPOSE_RATIO, self.compose, "compose"),
                                (COMPARE_RATIO, self.compare, "compare"),
                                (CORRECT_RATIO, self.correct, "correct"),
                                (TEMPORAL_RATIO, self.temporal, "temporal")):
            if r.random() < ratio and (t := fn(s)):
                out.append(self._mk(s, [t], name))

        # 否定变体
        if r.random() < NEGATION_RATIO and (t := self.negation(s)):
            out.append(self._mk(s, [t], "negation"))
        return out


# ---------------------------------------------------------------- 切分
def scene_quality(scenes: list[Scene]) -> dict[str, float]:
    """给每个 scene 打一个 0~1 的质量分, 供"多的筛精"时排序用。

    分数不跨数据源直接比: 卫星图天然锐利、夜间监控天然发糊, blur 的绝对值
    在数据源之间没有可比性。所以清晰度与目标数都先在**本数据源内部**换算成
    分位, 再加权 —— 比的是"在同源图里算不算好", 不是"比别的数据集清楚"。
    """
    by_src: dict[str, list[Scene]] = {}
    for s in scenes:
        by_src.setdefault(s.source_dataset, []).append(s)

    def pct(vals: list[float]) -> dict[int, float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        return {idx: (rank / max(1, len(vals) - 1)) for rank, idx in enumerate(order)}

    out: dict[str, float] = {}
    for group in by_src.values():
        blur_p = pct([float(s.meta.get("q", {}).get("blur", 0.0)) for s in group])
        nobj_p = pct([float(min(len(s.objects), 60)) for s in group])
        for i, s in enumerate(group):
            conf = max((e.conf for e in s.events), default=0.0)
            relaxed = any(e.evidence.get("relaxed") for e in s.events)
            score = (0.35 * blur_p[i] + 0.25 * nobj_p[i] + 0.30 * conf
                     + (0.10 if s.meta.get("hard_negative") else 0.0))
            if relaxed:
                score -= 0.20        # 有严格档样本可选时, 放宽出来的排后面
            if s.meta.get("uncertain"):
                score -= 0.10        # 闸5 判"不确定"的, 富余时优先让位
            out[s.image_id] = round(max(0.0, min(1.0, score)), 4)
    return out


def apply_quota(samples: list[dict], target: int, seed: int = 0,
                quality: dict[str, float] | None = None) -> list[dict]:
    """按异常类配额: 多的筛精, 少的原样保留并在报告里点名缺口。

    富余类不是随机丢, 是**按质量分排序后在各数据源之间轮着取**:
      - 排序保证留下的是同源里最清晰、目标最多、事件置信度最高的那批;
      - 轮取保证不会因为某个数据源又大又清晰, 就把这一类的名额全占了 ——
        全来自一个数据源的 25000 条, 训出来的是那个数据源的模型。
    下采样按 image_id 分组做, 不是逐条随机丢: 同一张图产出的几条 QA
    要么一起留要么一起丢, 否则同图样本被拆散, 后面按组切 train/test 就不准了。
    """
    if target <= 0:
        return samples
    quality = quality or {}
    by_cls: dict[str, list[dict]] = {}
    for s in samples:
        by_cls.setdefault("+".join(s["extra"]["anomaly"]), []).append(s)

    rng = random.Random(seed)
    out: list[dict] = []
    report: list[tuple[str, int, int, float]] = []
    for cls, group in by_cls.items():
        tgt = target if cls != "normal" else int(target * len(by_cls) * 3 / 7)
        if len(group) <= tgt:
            out.extend(group)
            report.append((cls, len(group), len(group), 0.0))
            continue

        by_img: dict[str, list[dict]] = {}
        for s in group:
            by_img.setdefault(s["extra"]["image_id"], []).append(s)
        # 数据源 -> 该源的图, 按质量分从高到低
        by_src: dict[str, list[str]] = {}
        for img, rows in by_img.items():
            by_src.setdefault(rows[0]["extra"].get("source_dataset", "?"), []).append(img)
        for src in by_src:
            by_src[src].sort(key=lambda k: (-quality.get(k, 0.5), k))

        kept: list[dict] = []
        picked_q: list[float] = []
        cursors = {src: 0 for src in by_src}
        srcs = sorted(by_src)
        rng.shuffle(srcs)                       # 轮取的起点随机, 避免总是同一个源占先
        while len(kept) < tgt and any(cursors[s] < len(by_src[s]) for s in srcs):
            for src in srcs:
                if cursors[src] >= len(by_src[src]) or len(kept) >= tgt:
                    continue
                img = by_src[src][cursors[src]]
                cursors[src] += 1
                kept.extend(by_img[img])
                picked_q.append(quality.get(img, 0.5))
        out.extend(kept)
        report.append((cls, len(group), len(kept),
                       sum(picked_q) / max(1, len(picked_q))))

    print("\n按类别配额(规则侧):")
    for cls, before, after, q in sorted(report, key=lambda r: -r[1]):
        tgt = target if cls != "normal" else int(target * len(by_cls) * 3 / 7)
        if after < tgt * 0.8:
            print(f"  {cls:22s} {after:7d} / 目标 {tgt}   ❌ 缺口 {tgt - after}")
        elif before > after:
            print(f"  {cls:22s} {after:7d} / 目标 {tgt}   ✅ 从 {before} 择优保留"
                  f"(留下的平均质量分 {q:.2f})")
        else:
            print(f"  {cls:22s} {after:7d} / 目标 {tgt}   ✅")
    return out


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
    ap.add_argument("--target-per-class", type=int, default=0,
                    help="每个异常类的目标条数(规则侧配额, 0 表示不限)。"
                         "富余的按图分组下采样, 不足的告警")
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
    if args.target_per_class:
        samples = apply_quota(samples, args.target_per_class, args.seed,
                              quality=scene_quality(scenes))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _write(name: str, data: list[dict]) -> None:
        """图像样本与视频样本分开落盘 —— LLaMA-Factory 里它们是两个数据集条目,
        columns 分别映射 images 与 videos, 混在一个文件里会加载失败。"""
        for suffix, sel in (("", lambda s: "videos" not in s),
                            ("_video", lambda s: "videos" in s)):
            part = [s for s in data if sel(s)]
            if not part:
                continue
            (out / f"{name}{suffix}.json").write_text(
                json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {name}{suffix}.json  {len(part)} 条")

    if args.no_split:
        _write("all", samples)
    else:
        for name, part in split_by_group(samples, seed=args.seed).items():
            _write(name, part)

    from collections import Counter
    tasks = Counter(s["extra"]["task"] for s in samples)
    turns = Counter(s["extra"]["n_turns"] for s in samples)
    mods = Counter(s["extra"]["modality"] for s in samples)
    n_norm = sum(1 for s in samples if s["extra"]["anomaly"] == ["normal"])
    print(f"\n共 {len(samples)} 条 / {len(scenes)} 个 scene")
    print(f"正常样本占比 {n_norm / max(1, len(samples)):.1%}"
          f"（目标 ≥{onto.get('negative_ratio_target', 0.3):.0%}）")
    print("  任务:", dict(tasks.most_common()))
    print("  轮数:", {f"{k}轮": v for k, v in sorted(turns.items())})
    print("  形态:", dict(mods))


if __name__ == "__main__":
    main()

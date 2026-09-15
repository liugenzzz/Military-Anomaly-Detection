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
from sharegpt import make_row, meta_of  # noqa: E402
from scene import Obj, Scene, load_scenes  # noqa: E402

# 任务配额(占该异常类总量)。描述与推理由 llm_qa 产出, 这里只列规则侧。
RULE_QUOTA = {"judge": 0.15, "locate": 0.12, "count": 0.05}
LOCATE_VERBAL_RATIO = 0.70      # 定位任务里方位指代占七成, 坐标占三成
NEGATION_RATIO = 0.20           # 每类任务的否定变体占比
TWO_TURN_RATIO = 0.30           # 规则侧的多轮占比
THREE_TURN_RATIO = 0.35         # 多轮里再有三成半追到第三轮("依据是什么")
COUNT_BOX_RATIO = 0.25          # 计数题里四分之一要求逐个框出
COUNT_THEN_JUDGE_RATIO = 0.55   # 计数/覆盖题里过半要追问一句"那这算正常吗"          # 计数题里四分之一要求逐个框出
COMPOSE_RATIO = 0.30            # 目标构成(哪几类各多少)
COMPARE_RATIO = 0.25            # 左右/上下密度对比
CORRECT_RATIO = 0.30            # 纠错题(给错误陈述让模型推翻)
TEMPORAL_RATIO = 0.60           # 时序题。只有带轨迹的视频/多帧能出, 条件本就少, 配比给高
# 越界专属三题。越界是四类里图最少的一类, 而它恰恰是信息最多的一类
# (有轨迹、有界线、有方向), 配比给到最高, 把一张图的信息榨干
CROSS_DIR_RATIO = 0.75
CROSS_COUNT_RATIO = 0.70
DENSE_REGION_RATIO = 0.55       # 困难负样本的"密集区在哪"
CROSS_NEG_RATIO = 0.35          # "有线但没人越线"的负样本。不给它, 模型会学成
                                # "只要问到警戒线就答有越界"

# 不可数的"东西": 烟和火是连续的一团, 没有"几个"可言。
# "清点一下画面中的烟雾"、"共 2 个目标: 火焰 1 个; 烟雾 1 个" —— 这种问法本身就不成立,
# 标注里的一个框只是标出了它的范围, 不代表"一个烟雾"。这类目标改问覆盖范围。
UNCOUNTABLE = {"fire", "smoke", "flame", "dust"}

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


def _area_zh(v: float) -> str:
    """面积占比 -> 中文。**别拿 _frac 代替它** —— 那个是方位格式化器,
    返回的是"近左缘""三分之一处"这类位置说法, 套到面积上会写出
    "约占画幅的近左缘"这种句子。"""
    if v < 0.02:
        return "占画幅不到百分之二"
    if v < 0.06:
        return "约占画幅百分之五"
    if v >= 0.9:
        return "几乎覆盖整个画面"
    return f"约占画幅{_TENTH[max(1, min(9, round(v * 10)))]}"


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


def crossed_objs(s: Scene) -> list[Obj]:
    tids = {str(e.evidence.get("track_id")) for e in s.events
            if e.evidence.get("rule") == "boundary_cross"}
    return [o for o in s.objects if o.track_id is not None and str(o.track_id) in tids]


def region_box_of(s: Scene, zh: dict[str, str]) -> tuple[list[float], str] | None:
    """异常区域的像素 bbox 与标签。没有区域信息就返回 None。

    规则侧和 LLM 侧共用这一份 —— 带框描述那类题里, **文字由模型写, 坐标由这里算**,
    两边必须是同一个框, 否则同一张图的坐标题和描述题会给出不一样的框。
    """
    ev = next((e for e in s.events if "cluster_bbox" in e.evidence), None)
    if ev:
        return ev.evidence["cluster_bbox"], f"{zh.get(ev.type, ev.type)}区域"
    for e in s.events:
        if e.evidence.get("boxes"):
            bs = e.evidence["boxes"]
            xs = [b[0] for b in bs] + [b[2] for b in bs]
            ys = [b[1] for b in bs] + [b[3] for b in bs]
            return [min(xs), min(ys), max(xs), max(ys)], zh.get(e.type, e.type)
    cands = [o for o in s.objects if o.cls in ("fire", "smoke")]
    if cands:
        o = max(cands, key=lambda x: x.area)
        return o.bbox, CLS_ZH.get(o.cls, o.cls)
    # 越界场景没有 cluster_bbox, 用越界目标的外接框
    objs = crossed_objs(s)
    if objs:
        xs = [v for o in objs for v in (o.bbox[0], o.bbox[2])]
        ys = [v for o in objs for v in (o.bbox[1], o.bbox[3])]
        return [min(xs), min(ys), max(xs), max(ys)], "越界目标所在区域"
    return None


# ---------------------------------------------------------------- 生成器
class RuleBuilder:
    def __init__(self, onto: dict[str, Any], prompt_dir: str, seed: int = 0):
        self.onto = onto
        self.zh = {c["id"]: c["zh"] for c in onto["classes"]}
        self.enabled = [c["id"] for c in onto["classes"]
                        if c["id"] != "normal" and c.get("enabled", True)]
        _, self.asks = load_all(prompt_dir)
        # system 变体。全量样本挂同一段长 system, 全参 SFT 下模型会把它当常量背下来,
        # 换个 prompt 就掉性能。给 5 个语义等价的写法 + 一个极简 + 一个空,
        # 让模型学的是"做这件事", 不是"背这段话"。
        sysdir = Path(prompt_dir) / "system"
        if sysdir.is_dir():
            self.systems = [f.read_text(encoding="utf-8").strip()
                            for f in sorted(sysdir.glob("*.txt"))]
        else:
            self.systems = [(Path(prompt_dir) / "system.txt").read_text(encoding="utf-8").strip()]
        self.system = self.systems[0]
        self.rng = random.Random(seed)
        self._n = 0

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
        return region_box_of(s, self.zh)

    def _crossed_objs(self, s: Scene) -> list[Obj]:
        return crossed_objs(s)

    def _mk(self, s: Scene, turns: list[tuple[str, str]], task: str, **extra) -> dict:
        field, paths = s.media
        # 越界类的问题必须自带警戒线前提 —— 线是虚拟的, 不写进问题, 模型无从判断。
        # 但只挂在真的与线有关的题上: 判定题答的是集结, 前面顶一句警戒线纯属噪声,
        # 还会让模型以为"凡是提到线的场合答案就该跟越界有关"。
        about_line = task.startswith("cross_") or (
            task.startswith(("judge", "locate")) and self._has_cross(s))
        if about_line and (pre := self.boundary_premise(s)):
            turns = [(pre + turns[0][0], turns[0][1])] + list(turns[1:])
        self._n += 1
        return make_row(
            sample_id=f"{s.image_id}_{task}_{self._n}",
            media_field=field, media=paths,
            system=self.rng.choice(self.systems), turns=turns,
            metadata={"image_id": s.image_id, "task_type": task, "gen": "rule",
                      "modality": s.modality, "n_media": len(paths),
                      "anomaly": s.anomaly_types or ["normal"],
                      "hard_negative": bool(s.meta.get("hard_negative")),
                      "source_dataset": s.source_dataset, "license": s.license,
                      "view": s.view, "image_width": s.width, "image_height": s.height,
                      "coordinate_mode": COORD_MODE, "bbox_scale": BBOX_SCALE, **extra},
        )

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
                    cls_hint = self.count_phrase(s, main[0], n)
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
            what = self.count_phrase(s, cls, n) if n else "若干目标"
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
        # **没有异常事件就不出定位题**。困难负样本确实有个真实的密集区(民用停车场、
        # 民航机坪), _region_box 也拿得到框, 但画面已经被判为正常 —— 再答一句
        # "异常位于画面右上", 就和上一轮的判定结论直接打架。
        # 自相矛盾的样本比缺样本伤得多, 这一类的区域信息改走 dense_region。
        if not s.anomaly_types:
            return None
        rb = self._region_box(s)
        if rb is None:
            return None
        box, label = rb
        gran = self.rng.choices(["zone", "ratio", "landmark"], weights=[40, 35, 25])[0]
        q = self._ask("locate_verbal", zh=self._anomaly_zh(s))
        return q, position_text(box, s.width, s.height, gran, self.rng)

    def locate_box(self, s: Scene) -> tuple[str, str] | None:
        if not s.anomaly_types:
            return None
        rb = self._region_box(s)
        if rb is None:
            return None
        box, label = rb
        q = self._ask("locate_box", zh=self._anomaly_zh(s))
        return q, box_json(to_bbox2d(box, s.width, s.height), label)

    def locate_box_multi(self, s: Scene) -> tuple[str, str] | None:
        """越界类: 多个目标一起框出。"""
        if not s.anomaly_types:
            return None
        objs = self._crossed_objs(s)
        if not objs:
            return None
        q = self._ask("locate_box", zh="越界目标")
        return q, boxes_json([(to_bbox2d(o.bbox, s.width, s.height), "越界目标") for o in objs])

    # 精确计数的闸。GT 来自高分辨原图和多帧跟踪, 人对着单帧数不出 26 个人头 ——
    # 拿它当单帧答案, 模型学到的是"猜一个像样的数", 这正是遥感 VLM 幻觉的主要来源。
    #
    # **两种数不清要分开**: 太多和太小, 说法完全不同。
    #   太多  -> 给量级("二三十辆"), 说明是密到点不清;
    #   太小  -> 干脆不出计数题。硬答"目测 3 辆上下, 目标很密"既自相矛盾又没意义。
    COUNT_EXACT_MAX = 15        # 超过这个数, 人看图也只能估
    COUNT_FEW = 6               # 这么少, 哪怕目标偏小也数得过来
    COUNT_MIN_AREA = 0.0008     # 想报准数, 中位目标至少占画幅万分之八
    AREA_FLOOR = 0.0002         # 低于这个, 连"看得见"都谈不上, 什么计数题都别出

    def _med_area(self, s: Scene, cls: str) -> float:
        objs = [o for o in s.objects if o.cls == cls]
        if not objs or not s.width or not s.height:
            return 0.0
        return sorted(o.area for o in objs)[len(objs) // 2] / (s.width * s.height)

    def _count_mode(self, s: Scene, cls: str, n: int) -> str:
        """exact = 给准数 / magnitude = 给量级 / none = 这题不出"""
        a = self._med_area(s, cls)
        if a < self.AREA_FLOOR:
            return "none"
        if n <= self.COUNT_FEW:
            return "exact"
        if n <= self.COUNT_EXACT_MAX and a >= self.COUNT_MIN_AREA:
            return "exact"
        if n > self.COUNT_EXACT_MAX:
            return "magnitude"
        return "none"                       # 不多不少但看不清 —— 不出

    def _countable_exact(self, s: Scene, cls: str, n: int) -> bool:
        return self._count_mode(s, cls, n) == "exact"

    def count_phrase(self, s: Scene, cls: str, n: int) -> str:
        """把数量写进句子时**一律走这里**。

        否则会漏: 计数题已经改口说"二三十名"了, 紧接着的判定句却又写
        "涉及 26 名人员" —— 同一条对话里前脚说数不清、后脚报准数。
        """
        mw, zh = MEASURE.get(cls, "个"), CLS_ZH.get(cls, cls)
        mode = self._count_mode(s, cls, n)
        if mode == "exact":
            return f"{n}{mw}{zh}"
        if mode == "magnitude":
            return f"{self._magnitude(n, mw)}{zh}"
        return f"若干{zh}"

    @staticmethod
    def _magnitude(n: int, mw: str) -> str:
        """数不清时的量级说法。**不给区间中值**, 免得又变成一个假精确的数。"""
        if n <= 30:
            return f"二三十{mw}"
        if n <= 60:
            return f"数十{mw}"
        if n <= 150:
            return f"上百{mw}"
        return f"数百{mw}以上"

    def count(self, s: Scene) -> tuple[str, str] | None:
        by: dict[str, int] = {}
        for o in s.objects:
            if o.cls in UNCOUNTABLE:
                continue                      # 烟雾/火焰没有"几个", 见 UNCOUNTABLE
            by[o.cls] = by.get(o.cls, 0) + 1
        if not by:
            return None
        cls, n = max(by.items(), key=lambda kv: kv[1])
        mw, zh = MEASURE.get(cls, "个"), CLS_ZH.get(cls, cls)
        mode = self._count_mode(s, cls, n)
        if mode == "exact":
            return self._ask("count", mw=mw, label=zh), f"{n}{mw}。"
        if mode == "magnitude":
            return (self._ask("scale_ask", label=zh),
                    f"{self._magnitude(n, mw)}，密集成片，无法逐个点清。")
        return None                         # 目标太小, 这题不成立

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
        """只统计**可数**目标。烟火不计数, 它们的信息走 coverage 题。"""
        by: dict[str, int] = {}
        for o in s.objects:
            if o.cls in UNCOUNTABLE:
                continue
            by[o.cls] = by.get(o.cls, 0) + 1
        return sorted(by.items(), key=lambda kv: -kv[1])

    def coverage(self, s: Scene) -> tuple[str, str] | None:
        """烟火这类不可数目标改问**覆盖范围**。面积由标注框算得出来, 仍是唯一答案。"""
        # 和定位题同一条闸: 没有异常事件就别问烟火 —— 问完范围紧接着一句
        # "未见异常", 又是自相矛盾。烟火在本体里本来就是异常类, 有烟火却无事件
        # 说明标注没跟上, 这种图不出题。
        if not s.anomaly_types:
            return None
        cands = [o for o in s.objects if o.cls in UNCOUNTABLE]
        if not cands:
            return None
        o = max(cands, key=lambda x: x.area)
        zh = CLS_ZH.get(o.cls, o.cls)
        ratio = o.area / max(1.0, float(s.width * s.height))
        x1, y1, x2, y2 = o.bbox
        where = _zone((x1 + x2) / 2 / max(1, s.width), (y1 + y2) / 2 / max(1, s.height))
        lead = ("范围很小，" if ratio < 0.06 else
                "范围较大，" if ratio >= 0.12 else "")
        scale = lead + _area_zh(ratio)
        return (self._ask("coverage", zh=zh),
                f"{zh}集中在{where}，{scale}。")

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
        if style < 0.5 and self._countable_exact(s, cls, n):   # 数量错(数得清才出)
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
        if n > 12 or cls in UNCOUNTABLE or not self._countable_exact(s, cls, n):
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

    def dense_region(self, s: Scene) -> tuple[str, str] | None:
        """困难负样本的"密集区在哪"。区域是真的, 只是它不构成异常 ——
        问法与答案都不提"异常"二字, 才不会和判定结论冲突。"""
        if s.anomaly_types or not s.meta.get("hard_negative"):
            return None
        box = s.meta.get("hard_negative_bbox") or (self._region_box(s) or [None])[0]
        if not box:
            return None
        gran = self.rng.choices(["zone", "ratio"], weights=[60, 40])[0]
        where = position_text(box, s.width, s.height, gran, self.rng)
        by: dict[str, int] = {}
        for o in s.objects:
            by[o.cls] = by.get(o.cls, 0) + 1
        cls, n = max(by.items(), key=lambda kv: kv[1]) if by else ("目标", 0)
        what = self.count_phrase(s, cls, n) if n else "若干目标"
        return (self._ask("dense_region"),
                f"{where}该区域聚集了 {what}，密度明显高于画面其他部分，"
                f"但均为民用目标，不属于需要上报的情况。")

    # ---------------------------------------------- 越界专属题
    @staticmethod
    def _has_cross(s: Scene) -> bool:
        return any(e.evidence.get("rule") == "boundary_cross" for e in s.events)

    def cross_negative(self, s: Scene) -> tuple[str, str] | None:
        """有警戒线、有活动目标、但**没人越线**。

        这是越界这一类里最该有的样本: 不给它, 模型会学成"只要问到警戒线就答有越界"。
        它天然是负样本, 而且数量管够 —— 序列里绝大多数帧本来就没有穿越发生。
        """
        if self._has_cross(s) or not s.regions or not s.tracks:
            return None
        n = len(s.tracks)
        if n == 0:
            return None
        q = self._ask("cross_count")
        return q, (f"没有目标越过这条线。画面中 {n} 个活动目标全程停留在线的同一侧，"
                   f"各自的移动都未触及界线。")

    def _cross_tracks(self, s: Scene) -> list[tuple[str, list[float], list[float]]]:
        """越界目标的轨迹首尾点。(track_id, 起点, 终点)"""
        out = []
        for e in s.events:
            if e.evidence.get("rule") != "boundary_cross":
                continue
            st, en = e.evidence.get("start"), e.evidence.get("end")
            if st and en:
                out.append((str(e.evidence.get("track_id")), list(st), list(en)))
        return out

    def cross_direction(self, s: Scene) -> tuple[str, str] | None:
        """越界方向。**只说画面方位, 不说界内界外** ——
        自动放置的是一条线, 线的两侧哪边算"内"根本无从判断, 硬说就是编。"""
        tr = self._cross_tracks(s)
        if not tr:
            return None
        dx = sum(e[0] - b[0] for _, b, e in tr) / len(tr)
        dy = sum(e[1] - b[1] for _, b, e in tr) / len(tr)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        horiz = "自左向右" if dx > 0 else "自右向左"
        vert = "自上而下" if dy > 0 else "自下而上"
        if abs(dx) > 2.5 * abs(dy):
            way = horiz
        elif abs(dy) > 2.5 * abs(dx):
            way = vert
        else:
            way = f"{horiz.replace('自', '自').replace('向', '偏')}{vert[1:]}"
            way = f"{horiz}、同时{vert}"
        who = CLS_ZH.get((s.events[0].evidence.get("cls") or ""), "目标")
        n = len(tr)
        tail = "，方向一致" if n > 1 and self._same_way(tr) else ""
        return (self._ask("cross_direction"),
                f"{n} 个{who}{way}穿过了界线{tail}。起始时位于界线一侧，"
                f"结束时已越到另一侧。")

    @staticmethod
    def _same_way(tr) -> bool:
        vs = [(e[0] - b[0], e[1] - b[1]) for _, b, e in tr]
        ax = sum(v[0] for v in vs) / len(vs)
        ay = sum(v[1] for v in vs) / len(vs)
        return all(v[0] * ax + v[1] * ay > 0 for v in vs)

    def cross_count(self, s: Scene) -> tuple[str, str] | None:
        """越界计数。同时报"没越的那些" —— 只问越界数, 模型会学成
        "画面里有几个动的就答几个"。"""
        tr = self._cross_tracks(s)
        if not tr:
            return None
        n_cross = len({t[0] for t in tr})
        n_total = len(s.tracks) or len({o.track_id for o in s.objects if o.track_id is not None})
        rest = max(0, n_total - n_cross)
        q = self._ask("cross_count")
        if rest:
            return q, (f"{n_cross} 个目标越过了界线；另有 {rest} 个目标"
                       f"全程停留在界线同一侧，未构成越界。")
        return q, f"{n_cross} 个目标越过了界线，画面中的活动目标全部发生了越界。"

    def boundary_premise(self, s: Scene) -> str:
        """把虚拟警戒线**写进问题里**。

        这条线是我们自己画的, 图上根本不存在。原先直接问"有几个目标越过了界线",
        等于要模型对一个它看不见的前提作答 —— 那不是在教它看图, 是在教它猜。
        真实的周界告警系统也是这么工作的: 线由系统给定, 模型判断的是"有没有跨过它"。
        """
        reg = next((r for r in s.regions if len(r.points) >= 2), None)
        if reg is None:
            return ""
        (x1, y1), (x2, y2) = reg.points[0], reg.points[-1]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) > 2.5 * abs(dy):
            shape = "近似水平横贯画面"
        elif abs(dy) > 2.5 * abs(dx):
            shape = "近似垂直纵贯画面"
        else:
            shape = "自左上向右下斜贯画面" if dx * dy > 0 else "自左下向右上斜贯画面"
        mx = (max(0.0, min(s.width, x1)) + max(0.0, min(s.width, x2))) / 2
        my = (max(0.0, min(s.height, y1)) + max(0.0, min(s.height, y2))) / 2
        where = _zone(mx / max(1, s.width), my / max(1, s.height))
        return f"设有一条{shape}、经过{where}的虚拟警戒线。"


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
            # **数完要接着问异常**。"清点一下画面中的车辆" -> "12 辆" 然后就断了,
            # 这种题只教模型数数, 教不会它"数出来之后说明什么"。实际用起来,
            # 清点是判定的铺垫: 先数清楚, 再据此回答这批目标的分布正不正常。
            if r.random() < COUNT_THEN_JUDGE_RATIO:
                out.append(self._mk(s, [t, (self._ask("after_count"), ja)],
                                    "count+judge", form="multi_turn"))
            else:
                out.append(self._mk(s, [t], "count"))

        # 烟火不可数, 改问覆盖范围; 同样接一轮判定
        if (t := self.coverage(s)):
            if r.random() < COUNT_THEN_JUDGE_RATIO:
                out.append(self._mk(s, [t, (self._ask("after_count"), ja)],
                                    "coverage+judge", form="multi_turn"))
            else:
                out.append(self._mk(s, [t], "coverage"))

        # 构成 / 对比 / 纠错 / 时序 —— 各按配比抽, 抽不到条件就跳过, 不硬凑
        for ratio, fn, name in ((COMPOSE_RATIO, self.compose, "compose"),
                                (COMPARE_RATIO, self.compare, "compare"),
                                (CORRECT_RATIO, self.correct, "correct"),
                                (TEMPORAL_RATIO, self.temporal, "temporal")):
            if r.random() < ratio and (t := fn(s)):
                out.append(self._mk(s, [t], name))

        # 越界专属题。这一类的图最少, 但一张图能问的东西不止"有没有异常":
        # 方向、越了几个没越几个、界线怎么走, 答案全由轨迹和界线唯一决定,
        # 零成本零幻觉 —— 比在同一批画面上多生成几段描述划算得多。
        # 困难负样本: 区域信息走"密集区在哪", 不走"异常在哪"
        if r.random() < DENSE_REGION_RATIO and (t := self.dense_region(s)):
            out.append(self._mk(s, [t], "dense_region"))

        for ratio, fn, name in ((CROSS_DIR_RATIO, self.cross_direction, "cross_direction"),
                                (CROSS_COUNT_RATIO, self.cross_count, "cross_count"),
                                (CROSS_NEG_RATIO, self.cross_negative, "cross_negative")):
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

    # 按"每个异常类实际出现多少次"统计, 再把同时属于多类的样本归到其中最稀缺的一类。
    # 不能拿 "+".join 当类名: 上一轮 explosion+smoke 被当成了独立类别, 于是报告里
    # explosion 只有 282 条、缺口 9718, 而实际上另外 8557 条 explosion+smoke 里
    # 每一条都是 explosion —— 按假类名配额, 既把缺口报错, 也会把该留的样本丢掉。
    total: dict[str, int] = {}
    for s in samples:
        for c in meta_of(s)["anomaly"]:
            total[c] = total.get(c, 0) + 1
    by_cls: dict[str, list[dict]] = {}
    for s in samples:
        cls = min(meta_of(s)["anomaly"], key=lambda c: (total.get(c, 0), c))
        by_cls.setdefault(cls, []).append(s)

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
            by_img.setdefault(meta_of(s)["image_id"], []).append(s)
        # 数据源 -> 该源的图, 按质量分从高到低
        by_src: dict[str, list[str]] = {}
        for img, rows in by_img.items():
            by_src.setdefault(meta_of(rows[0]).get("source_dataset", "?"), []).append(img)
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

    # 最终每个异常类实际覆盖多少条(同时属于多类的样本, 每一类都算它一次)
    final: dict[str, int] = {}
    for s in out:
        for c in meta_of(s)["anomaly"]:
            final[c] = final.get(c, 0) + 1

    print("\n按类别配额(规则侧)。归属列 = 归到本类名下的条数, 覆盖列 = 含本类的全部条数:")
    for cls, before, after, q in sorted(report, key=lambda r: -r[1]):
        tgt = target if cls != "normal" else int(target * len(by_cls) * 3 / 7)
        cov = final.get(cls, 0)
        if before > after:
            note = f"✅ 从 {before} 择优保留(平均质量分 {q:.2f})"
        elif cov < tgt * 0.8:
            # 把"配额砍下来的"和"数据本来就这么多"分开说。
            # 后者不叫缺口 —— 该类可用的样本已经一条不剩地留下了, 再报缺口只会
            # 让人去想办法凑数, 而凑数只能靠在同一批画面上反复出题。
            note = f"⚠ 数据见底(该类可用样本已全部保留)"
        else:
            note = "✅"
        print(f"  {cls:20s} 归属 {after:7d}  覆盖 {cov:7d} / 目标 {tgt:6d}  {note}")
    multi = sum(1 for s in out if len(meta_of(s)["anomaly"]) > 1)
    if multi:
        print(f"  其中 {multi} 条同时属于多个异常类(如烟雾与爆炸同框), "
              f"已归到最稀缺的那一类, 覆盖列里两类都计入")

    # 真正该盯的不是绝对条数, 是类间比例。差到 3:1 以上, 少的那类会学不动;
    # 2:1 上下属于正常波动, 不值得为它去复制样本。
    anom = {c: n for c, n in final.items() if c != "normal"}
    if len(anom) >= 2:
        hi, lo = max(anom.items(), key=lambda kv: kv[1]), min(anom.items(), key=lambda kv: kv[1])
        ratio = hi[1] / max(1, lo[1])
        verdict = ("✅ 均衡" if ratio <= 2 else
                   "✅ 轻微不均, 不影响训练" if ratio <= 3 else
                   "⚠ 偏斜明显, 建议训练时对少的那类加采样权重")
        print(f"  类间比例: 最多 {hi[0]} {hi[1]} / 最少 {lo[0]} {lo[1]} = "
              f"{ratio:.1f}:1  {verdict}")
    return out


def split_by_group(samples: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 0):
    """按来源图/视频分组切分, 避免同源相邻帧跨集造成指标虚高。"""
    groups: dict[str, list[dict]] = {}
    for s in samples:
        gid = meta_of(s)["image_id"].rsplit("_frame", 1)[0]
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
    tasks = Counter(meta_of(s)["task_type"] for s in samples)
    turns = Counter(meta_of(s)["n_turns"] for s in samples)
    mods = Counter(meta_of(s)["modality"] for s in samples)
    n_norm = sum(1 for s in samples if meta_of(s)["anomaly"] == ["normal"])
    print(f"\n共 {len(samples)} 条 / {len(scenes)} 个 scene")
    print(f"正常样本占比 {n_norm / max(1, len(samples)):.1%}"
          f"（目标 ≥{onto.get('negative_ratio_target', 0.3):.0%}）")
    print("  任务:", dict(tasks.most_common()))
    print("  轮数:", {f"{k}轮": v for k, v in sorted(turns.items())})
    print("  形态:", dict(mods))


if __name__ == "__main__":
    main()

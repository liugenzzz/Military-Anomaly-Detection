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
from facets import load_all  # noqa: E402
from config import CFG  # noqa: E402
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
# 烟是连续的一团, 数不出"几个"; 但**火是可数的** —— 实测 FASDD 一张图标了
# 5 个、9 个独立火点, 那是真的可以逐个数的。把 fire 一起归进不可数是我想当然了。
UNCOUNTABLE = {"smoke", "dust", "haze"}

MEASURE = {"military-plane": "架", "civil-plane": "架", "plane": "架",
           "military-helicopter": "架", "civil-helicopter": "架", "helicopter": "架",
           "tank": "辆", "vehicle": "辆", "large-vehicle": "辆", "truck": "辆",
           "military-vehicle": "辆", "military-truck": "辆", "armored": "辆",
           "person": "名", "soldier": "名", "drone": "架",
           "ship": "艘", "warship": "艘"}
CLS_ZH = {"military-plane": "军用飞机", "military-truck": "军用卡车",
          "armored": "装甲车", "storage-tank": "储罐", "harbor": "码头",
          "bridge": "桥梁", "helipad": "直升机坪",
          "building": "建筑", "house": "房屋", "roof": "屋顶", "road": "道路",
          "water": "水体", "debris": "碎屑", "civil-plane": "民航飞机", "tank": "坦克",
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


_ZH = r"\u4e00-\u9fff"


def zh_norm(t: str) -> str:
    """中文与数字之间不留半角空格。

    模板里 f"{n} 个目标" 这种写法很自然, 但落到中文句子里就是
    "共观察到 9 架军用飞机" —— tokenizer 会把这个空格当成风格学走。
    与其在几十处模板里逐个抠, 不如在出口统一处理一次。
    bbox JSON 用的是紧凑分隔符, 本来就没有空格, 不受影响。
    """
    import re
    t = re.sub(rf"([{_ZH}])\s+([0-9{_ZH}])", r"\1\2", t)
    t = re.sub(rf"([0-9%])\s+([{_ZH}])", r"\1\2", t)
    return t


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
    # 本体里查不到中文名时**退到通用中文词, 绝不落英文 id**。
    # `"label":"crowd_gathering区域"` 这种一旦进了训练集, 模型就会学着往
    # 中文答案里吐英文标识符, 而且抽检时很容易被当成正常内容滑过去。
    ev = next((e for e in s.events if "cluster_bbox" in e.evidence), None)
    if ev:
        return ev.evidence["cluster_bbox"], f"{zh.get(ev.type) or '异常'}区域"
    for e in s.events:
        if e.evidence.get("boxes"):
            bs = e.evidence["boxes"]
            xs = [b[0] for b in bs] + [b[2] for b in bs]
            ys = [b[1] for b in bs] + [b[3] for b in bs]
            return [min(xs), min(ys), max(xs), max(ys)], zh.get(e.type) or "异常目标"
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
        # 本体里查不到中文名的类别。跑完在报告里点出来 —— 这类问题静默降级的
        # 代价是整批数据里混进「为异常」和英文 id, 肉眼抽检很难发现。
        self.unknown_types: set[str] = set()

    # -------------------------------------------------- 小工具
    def _ask(self, task: str, **kw) -> str:
        q = self.rng.choice(self.asks[task].lines)
        for k, v in kw.items():
            q = q.replace("{" + k + "}", str(v))
        return q

    def _anomaly_zh(self, s: Scene, default: str = "异常") -> str:
        """异常类别的中文名。**本体里没有的类别要记账, 不能悄悄糊过去。**

        原来这里无条件 `or "异常"`, 于是 ontology.yaml 里没登记的类别(比如 demo
        数据的 crowd_gathering)会生成「存在异常，为异常。」—— 一句什么都没说的
        废话, 却照样落盘进训练数据。同一个根因在 region_box_of 那边表现为
        `"label":"crowd_gathering区域"`, 直接把英文本体 id 训给了模型。

        现在: 解析不出来的类别进 unknown_types, 跑完统一告警; 判定句那边传
        default="" 让它换一套不点名的措辞, 而不是硬凑一个"为异常"。
        """
        names = [self.zh[t] for t in s.anomaly_types if t in self.zh]
        miss = [t for t in s.anomaly_types if t not in self.zh]
        if miss:
            self.unknown_types.update(miss)
        return "、".join(names) or default

    def _cluster(self, s: Scene):
        return next((e for e in s.events if "cluster_bbox" in e.evidence), None)

    def _region_box(self, s: Scene) -> tuple[list[float], str] | None:
        return region_box_of(s, self.zh)

    def _crossed_objs(self, s: Scene) -> list[Obj]:
        return crossed_objs(s)

    def _mk(self, s: Scene, turns: list[tuple[str, str]], task: str, **extra) -> dict:
        field, paths = s.media
        turns = [(zh_norm(q), zh_norm(a)) for q, a in turns]
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
                      "coordinate_mode": COORD_MODE, "bbox_scale": BBOX_SCALE,
                      # 两条流水线的字段集必须一致 —— 规则侧缺 facet、LLM 侧缺 view,
                      # 下游按 key 取值就会时有时无
                      "facet": None, "review": {"status": "rule_deterministic"},
                      **extra},
        )

    # -------------------------------------------------- 四类任务
    DISASTER_ZH = {"flood": "大面积水体漫过原有地表", "landslide": "山体垮塌与泥流痕迹",
                   "collapse": "建筑成片倒塌", "collision": "事故点车辆异常聚集"}

    def _judge_detail(self, s: Scene, cls_hint: str) -> str:
        """判定句里的那一句细节。**按事件类型分开写** —— 车队说"成簇分布"是错的,
        那是集结的话; 灾害类没有 count 字段, 套计数模板会变成一句空话。"""
        e = s.events[0] if s.events else None
        rule = e.evidence.get("rule") if e else None
        if rule == "linear_formation":
            n = e.evidence.get("count")
            what = cls_hint or (f"{n} 个目标" if n else "多个目标")
            return self.rng.choice([
                f"{what}沿一条线排开，前后间距大致相等。",
                f"可见 {what} 排成纵队行进，队形整齐。",
                f"涉及 {what}，呈线性排列而非散布。",
            ])
        if e is not None and e.type == "disaster":
            sub = e.evidence.get("subtype") or ""
            what = self.DISASTER_ZH.get(sub, "地表出现大范围异常变化")
            return self.rng.choice([f"画面中可见{what}。", f"主要特征是{what}。",
                                    f"{what}，覆盖范围明显。"])
        if cls_hint:
            return self.rng.choice([
                f"画面中可见{cls_hint}密集分布。",
                f"共观察到{cls_hint}，成簇分布。",
                f"涉及{cls_hint}。",
            ])
        return ""

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
            zh_name = self._anomaly_zh(s, default="")
            if not zh_name:
                # 叫不出名字就别硬点名。「存在异常，为异常。」这种句子训不出东西,
                # 只会教模型用同义反复搪塞。
                lead, zh_name = self.rng.choice(
                    ["存在异常", "画面中出现异常", "判定为异常"]), ""
            tail = self._judge_detail(s, cls_hint)
            if s.events and all(e.evidence.get("relaxed") for e in s.events):
                need = self.rng.choice(["规模有限，建议继续观察确认。", "尚未达到典型规模，建议复核。",
                                        "证据强度一般，建议结合后续画面判断。"])
            else:
                need = self.rng.choice(["", "建议上报并持续观察。", "建议持续观察。", ""])
            a = f"{lead}{zh_name}。{tail}{need}"
        elif s.meta.get("hard_negative"):
            by: dict[str, int] = {}
            for o in s.objects:
                by[o.cls] = by.get(o.cls, 0) + 1
            cls, n = max(by.items(), key=lambda kv: kv[1]) if by else ("目标", 0)
            what = self.count_phrase(s, cls, n) if n else "若干目标"
            # **困难负样本"为什么不算异常"要分情况说。**
            # 写死"均为民用目标, 未见军机"的话, 一片停机坪上的军机会被这么描述 ——
            # 那是睁眼说瞎话。derive_events 已经按簇里的实际类别记了 kind,
            # 这里照着说, 不要自己猜。
            kind = s.meta.get("hard_negative_kind", "civil_cluster")
            if kind == "ordinary_crowd":
                a = self.rng.choice([
                    f"未见异常。画面里确实有{what}，数量不少，但这是广场、道路一类"
                    f"公共场所的日常人流，分布松散，不构成异常聚集。",
                    f"未见异常。{what}分散在开阔场地上活动，属于正常的公共活动，"
                    f"没有向某一点集中的迹象。",
                    f"未见异常。人数虽多，但{what}三三两两地走动停留，"
                    f"并非成规模地聚在一处。",
                ])
            elif kind == "aircraft_parking":
                a = self.rng.choice([
                    f"未见异常。画面中虽有{what}集中停放，但这是机场停机坪的常态，"
                    f"航空器均处于静止停放状态，没有向某处汇聚的迹象，属于正常场景。",
                    f"未见异常。{what}排列整齐地停在停机坪上，属于日常停放，"
                    f"不构成兵力或装备的集结。",
                    f"未见异常。画面里确实有{what}密集分布，但它们停在固定机位上，"
                    f"这是机场的常规状态，不是异常聚集。",
                ])
            else:
                a = self.rng.choice([
                    f"未见异常。画面中虽有{what}密集成簇、达到了集结的规模条件，"
                    f"但均为民用目标，未见坦克、装甲车等军事装备，属于正常场景。",
                    f"未见异常。{what}聚在一起，规模看着够，但都是民用车辆，"
                    f"不构成装备集结。",
                ])
        else:
            a = self.rng.choice([
                "未见异常。目标分布稀疏，无烟火迹象，也没有成队行进的车辆。",
                "未见异常，态势正常，不需要上报。画面中没有成规模的目标聚集，"
                "也没有烟雾或火光。",
                "属于正常态势。逐项核查：无军事装备集结、无人员异常聚集、"
                "无烟火、无车队机动、无灾害迹象。",
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
        # **"太小"只否掉准数, 不该连量级一起否掉。**
        # 原来 a < AREA_FLOOR 直接 return "none", 于是 DroneCrowd 那种一图上百个
        # 人头点(每个只占画幅 0.012%)一道计数题都出不来 —— 可人群计数正是这个
        # 数据集的本行, GT 极可靠, 问「大概多少人」完全成立, 只是不能问准数。
        if a < self.AREA_FLOOR:
            return "magnitude" if n > self.COUNT_EXACT_MAX else "none"
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
            # **数不清的场景不能给准数。** 这里原来直接写 f"{n}{mw}{zh}",
            # 于是同一张 DroneCrowd 图上, 计数题答"数百名以上, 无法逐个点清",
            # 纠错题却答"确为158名人员" —— 前脚说数不清后脚报准数。
            # 数量措辞一律走 count_phrase(), 这是 CLAUDE.md 里的"单一出口"。
            if not self._countable_exact(s, cls, n):
                return None                   # 量级说法撑不起纠错题, 这题不出
            what = self.count_phrase(s, cls, n)
            return (self._ask("correct", claim=f"画面中有{what}"),
                    f"说法属实，画面中确为{what}。")
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
        return q, f"共{n}{mw}{zh}。\n{body}"

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

    # ---------------------------------------------- 能力边界
    UNANSWERABLE = [
        ("型号", "画面分辨率不足以分辨具体型号，只能看出大致的目标类别，无法确认型号。"),
        ("朝向", "目标在画面中只占很小一块，轮廓看不清，无法判断朝向。"),
        ("颜色", "目标尺寸太小，像素有限，颜色难以准确分辨。"),
        ("穿着", "航拍视角下人员只有几个像素，看不清衣着，无法判断。"),
        ("归属", "画面里没有可供识别归属的标识，无法确认属于哪一方。"),
        ("时间", "这是一张静止画面，看不出事件从什么时候开始，无法判断。"),
        ("发展", "单帧画面给不出后续变化，无法判断接下来会怎么发展。"),
        ("运动", "单帧看不出运动状态，无法判断目标是静止还是在移动。"),
        ("起因", "画面只呈现了当前状态，看不出事件的起因。"),
        ("逐个类型", "目标密集且单个尺寸很小，无法逐一分辨各自的具体类型。"),
    ]

    def unanswerable(self, s: Scene) -> tuple[str, str] | None:
        """能力边界声明。**这是全批最该有的一类样本**。

        现有开源 VLM 普遍 overclaim: 目标只有十几个像素也敢报型号, 单帧也敢说
        火势在扩大。没有"看不出就说看不出"的样本, 微调只会把这个毛病放大。
        挑的问题都是**这张图确实答不出来**的: 单帧问时序、小目标问型号颜色、
        航拍问人员衣着。
        """
        tiny = False
        if s.objects and s.width and s.height:
            med = sorted(o.area for o in s.objects)[len(s.objects) // 2]
            tiny = med / (s.width * s.height) < self.COUNT_MIN_AREA
        single = s.modality == "image"
        pool = []
        for key, ans in self.UNANSWERABLE:
            if key in ("型号", "朝向", "颜色", "穿着", "逐个类型") and not tiny:
                continue
            if key in ("时间", "发展", "运动") and not single:
                continue
            pool.append((key, ans))
        if not pool:
            return None
        key, ans = self.rng.choice(pool)
        lines = [q for q in self.asks["unanswerable"].lines]
        idx = {"型号": 0, "朝向": 1, "颜色": 2, "穿着": 3, "归属": 4,
               "时间": 5, "发展": 6, "运动": 7, "起因": 8, "逐个类型": 9}[key]
        return lines[min(idx, len(lines) - 1)], ans

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
        # 和判定句同一个道理: 措辞要按困难负样本的种类来, 不能写死。
        # 对一广场行人说"均为民用目标"是答非所问 —— 人本来就不分军民,
        # 问题在于"这是日常人流不是聚集"; 对一片军机说这句话更是睁眼说瞎话。
        tail = {
            "ordinary_crowd": "，但这是公共场所的日常人流，分布松散，不属于需要上报的情况。",
            "aircraft_parking": "，但它们停在固定机位上，属于机场的常规停放，不需要上报。",
        }.get(s.meta.get("hard_negative_kind", "civil_cluster"),
              "，但均为民用目标，不属于需要上报的情况。")
        return (self._ask("dense_region"),
                f"{where}该区域聚集了{what}，密度明显高于画面其他部分{tail}")

    def why(self, s: Scene) -> str | None:
        """多轮里的第三轮"依据是什么"。只复述证据字段, 不做任何延伸判断。

        **理由必须按异常类型分开**: 车队的理由是"排成一条、间距均匀", 不是集结的
        "空间密度高" —— 套错模板会让第一轮结论和第三轮理由对不上, 那比没有理由更糟。
        """
        ev = next((e for e in s.events if e.evidence.get("rule") == "linear_formation"), None)
        if ev:
            n = ev.evidence.get("count")
            cv = ev.evidence.get("spacing_cv")
            if isinstance(cv, float):
                extra = ("，前后间距几乎没有起伏" if cv < 0.05
                         else f"，前后间距的起伏在 {cv:.0%} 以内")
            else:
                extra = ""
            return (f"依据是排布形态：{n} 个目标沿一条线首尾相接排开{extra}，"
                    f"整体细长而非成片散布，与路边随机停放或拥堵车流的形态不同。")
        ev = self._cluster(s)
        if ev:
            n = ev.evidence.get("count")
            cls = "、".join(CLS_ZH.get(c, c) for c in ev.evidence.get("classes", [])[:3])
            if ev.evidence.get("relaxed"):
                return (f"依据是目标的空间密度：{n}个{cls or '目标'}聚成一簇，间距小于周边，"
                        f"但规模不大，只能算{self.zh.get(ev.type, '该类异常')}的迹象，"
                        f"还不足以下确定结论。")
            return (f"依据是目标的空间密度：{n}个{cls or '目标'}的间距明显小于画面中"
                    f"其他区域，聚成一簇，规模已达到{self.zh.get(ev.type, '该类异常')}的判定条件。")
        ev = next((e for e in s.events if e.type == "disaster"), None)
        if ev:
            sub = ev.evidence.get("subtype") or ""
            return (f"依据是地表状态的改变：{self.DISASTER_ZH.get(sub, '大范围地表异常')}，"
                    f"受影响区域与周边未受影响的部分有明显分界。")
        ev = next((e for e in s.events if e.type in ("smoke", "explosion")), None)
        if ev:
            what = "高亮度火光与其向外的骤降" if ev.type == "explosion" else "边界模糊的半透明烟团"
            return (f"依据是可见的{what}，它遮住了下方的地表纹理，"
                    f"说明位于地表之上而非地物本身的颜色。")
        if s.meta.get("hard_negative"):
            rs = s.meta.get("hard_negative_reason") or []
            return ("依据是目标性质：" + (rs[0] if rs else "成簇目标均为民用，不构成军事集结") +
                    "，因此不按异常处置。")
        if not s.anomaly_types:
            return "依据是逐项排查的结果：目标分布稀疏、无烟火迹象、无成队行进的车辆，各项均未触发。"
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

        # 能力边界声明 —— 专治 overclaim
        if r.random() < CFG.task("unanswerable", 0.10) and (t := self.unanswerable(s)):
            out.append(self._mk(s, [t], "unanswerable"))

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
                quality: dict[str, float] | None = None,
                neg_target: float = 0.3) -> list[dict]:
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

    # **负样本配额必须从"实际留下多少正样本"反推, 不能从 target 推。**
    # 原来写的是 target * 类数 * 3/7, 即拿**愿望值**去算。可各异常类根本达不到
    # target: smoke 12899 / explosion 8259 / massing 1007, 加起来 22165,
    # 而这个公式给负样本算出 25000*4*3/7 = 42857 —— 负样本是正样本的两倍,
    # 占比 66% 而目标是 30%。那样训出来的模型会学成"一律回答没异常"。
    # 先把异常类过一遍, 数清实际留下多少, 再按比例给 normal 定额。
    anomaly_cls = [c for c in by_cls if c != "normal"]
    kept_pos = sum(min(len(by_cls[c]), target) for c in anomaly_cls)
    neg_ratio = float(neg_target or 0.3)
    normal_tgt = int(kept_pos * neg_ratio / max(1e-6, 1.0 - neg_ratio))

    def _tgt(cls: str) -> int:
        return normal_tgt if cls == "normal" else target

    # 先跑异常类, 再跑 normal —— 顺序无关紧要(额度已提前算好), 但让报告里
    # 异常类排在前面更好读。
    for cls in sorted(by_cls, key=lambda c: (c == "normal", c)):
        group = by_cls[cls]
        tgt = _tgt(cls)
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
        tgt = _tgt(cls)
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
    # 负样本占比: 这是训练判定能力的关键配比, 必须核对而不是只报告
    n_neg = final.get("normal", 0)
    n_pos = sum(n for c, n in final.items() if c != "normal")
    if n_neg + n_pos:
        ratio = n_neg / (n_neg + n_pos)
        ok = abs(ratio - neg_target) <= 0.05
        print(f"  负样本占比: {ratio:.0%}  目标 {neg_target:.0%}  "
              f"{'✅' if ok else '⚠ 偏离目标, 检查 negative_ratio_target 与可用负样本数'}")

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
                   "⚠ 偏斜明显, 建议训练时对少的那类加采样权重" if ratio <= 8 else
                   "❌ 严重失衡 —— 最稀的那一类会被淹掉")
        print(f"  类间比例: 最多 {hi[0]} {hi[1]} / 最少 {lo[0]} {lo[1]} = "
              f"{ratio:.1f}:1  {verdict}")
        if ratio > 8:
            print(f"    要么给 {lo[0]} 补数据源, 要么对 {hi[0]} 主动下采样;")
            print(f"    **不要靠在 {lo[0]} 的同一批画面上反复出题来凑** —— "
                  f"那是同质化, 不是数据量。")
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
                              quality=scene_quality(scenes),
                              neg_target=float(onto.get("negative_ratio_target", 0.3)))
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
    if builder.unknown_types:
        print(f"\n⚠ 本体里查不到中文名的类别: {sorted(builder.unknown_types)}")
        print("  这些场景的判定句会退成不点名的说法、框的 label 退成通用词 ——")
        print("  能跑, 但训不出这几类的名字。要么在 configs/ontology.yaml 里补上,")
        print("  要么确认这些 scene 本来就不该进这一批。")


if __name__ == "__main__":
    main()

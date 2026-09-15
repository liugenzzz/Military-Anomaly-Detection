"""从底座检测/跟踪标注自动派生异常事件标签。

三条规则(与 configs/ontology.yaml 中的 rule.kind 对应):
  density_cluster  目标密度聚类 -> 集结 / 人员聚集 / 舰船集结 / 机群
  linear_formation 共线且同向    -> 车队机动
  boundary_cross   轨迹跨越边界  -> 越界移动

这样做的意义: 公开数据里没有"集结""越界"的标签, 但有大量 bbox 与轨迹,
用客观几何规则把它们转成事件标签, 是唯一能规模化的路子。
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from scene import Event, Obj, Scene, dump_scenes, load_scenes


# ---------------------------------------------------------------- 聚类
def _union_find_cluster(points: list[tuple[float, float]], eps: float) -> list[list[int]]:
    """单链接聚类: 距离 < eps 的点连通。点数不大(<数千), O(n^2) 足够。"""
    n = len(points)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    eps2 = eps * eps
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 < eps2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# 公开别名: DOTA 切片需要在原图坐标上先聚类, 避免把一个簇切散
union_find_cluster = _union_find_cluster


def cluster_eps(objs: list[Obj], diag: float, rule: dict[str, Any]) -> float:
    """邻域半径 = max(图像对角线比例, 目标尺寸倍数)。

    只按图像对角线取会导致**尺度依赖**: 同一片机群, 在 2400x1600 原图上间距
    90px 能连通, 裁成 1024x1024 切片后对角线变小、阈值收缩到 87px 就连不上了。
    加一条"目标尺寸的若干倍"作为下限, 判定就与裁剪尺寸无关 —— 相邻的物理含义
    本来就是"间距不超过目标本身的几倍", 而不是"占画幅的百分之几"。
    """
    eps = rule.get("eps_ratio", 0.06) * diag
    mult = rule.get("eps_obj_mult")
    if mult and objs:
        sizes = sorted(max(o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]) for o in objs)
        median = sizes[len(sizes) // 2]
        eps = max(eps, mult * median)
    return eps


def derive_density_cluster(scene: Scene, rule: dict[str, Any], cls_id: str,
                          subtype: str | None = None) -> list[Event]:
    """密度聚类派生聚集事件。

    require_any 是关键的一条: 装备集结必须含军事目标, 否则民用停车场/卡车场
    会被大量错标成兵力集结。规模达标但不含军事目标的簇不产生事件, 而是把该图
    标为 hard_negative —— 这类"看着像集结但不是"的样本是最有价值的困难负样本。
    """
    targets = {c.lower() for c in rule["target_classes"]}
    objs = scene.of_classes(targets)
    if len(objs) < rule["min_cluster_size"]:
        return _veto(cls_id, "数据源无此类目标" if not objs
                     else f"目标数不足({len(objs)}<{rule['min_cluster_size']})")

    # 交通否决: 同框有成规模的车辆/摩托 -> 这是城市街景的人流, 不是聚集
    veto_cls = {c.lower() for c in rule.get("traffic_veto_classes", [])}
    if veto_cls:
        n_traffic = sum(1 for o in scene.objects if o.cls.lower() in veto_cls)
        if n_traffic >= int(rule.get("traffic_veto_count", 5)):
            scene.meta["hard_negative"] = True
            scene.meta.setdefault("hard_negative_reason", []).append(
                f"{cls_id}/{subtype or '-'}: {len(objs)} 人聚集, 但同框有 {n_traffic} 个"
                f"车辆/摩托等交通目标, 属于城市街景的人流")
            cs = [o.center for o in objs]
            if cs:
                scene.meta.setdefault("hard_negative_bbox",
                                      [min(c[0] for c in cs), min(c[1] for c in cs),
                                       max(c[0] for c in cs), max(c[1] for c in cs)])
            return _veto(cls_id, "同框交通目标过多(城市人流)")

    require = {c.lower() for c in rule.get("require_any", [])}
    on_fail = rule.get("on_require_fail")
    eps = cluster_eps(objs, scene.diag, rule)
    centers = [o.center for o in objs]
    events = []
    for group in _union_find_cluster(centers, eps):
        if len(group) < rule["min_cluster_size"]:
            REJECTS[cls_id][f"单簇规模不足(<{rule['min_cluster_size']})"] += 1
            continue
        cls_in_group = {objs[i].cls.lower() for i in group}
        if require and not (cls_in_group & require):
            REJECTS[cls_id]["不含军事目标(require_any)"] += 1
            if on_fail == "hard_negative":
                scene.meta["hard_negative"] = True
                scene.meta.setdefault("hard_negative_reason", []).append(
                    f"{cls_id}/{subtype or '-'}: {len(group)} 个目标密集成簇但不含军事目标")
                # 把"像但不是"的那块区域记下来。困难负样本没有事件, 下游算不出区域框,
                # 而这块区域正是它最有价值的部分 —— 描述题要讲清"密在哪、为什么不算",
                # 定位题要能问"目标最密的一片在哪"(但绝不能问成"异常在哪")。
                xs = [centers[i][0] for i in group]
                ys = [centers[i][1] for i in group]
                box = [min(xs), min(ys), max(xs), max(ys)]
                prev = scene.meta.get("hard_negative_bbox")
                if prev is None or (box[2] - box[0]) * (box[3] - box[1]) > \
                        (prev[2] - prev[0]) * (prev[3] - prev[1]):
                    scene.meta["hard_negative_bbox"] = box   # 取最大的那一簇
            continue
        xs = [centers[i][0] for i in group]
        ys = [centers[i][1] for i in group]
        ev: dict[str, Any] = {
            "rule": "density_cluster",
            "count": len(group),
            "cluster_bbox": [min(xs), min(ys), max(xs), max(ys)],
            "object_ids": [objs[i].id for i in group],
            "classes": sorted({objs[i].cls for i in group}),
        }
        if subtype:
            ev["subtype"] = subtype
        events.append(Event(
            type=cls_id,
            conf=min(1.0, len(group) / (2.0 * rule["min_cluster_size"])),
            evidence=ev,
        ))
    return events


# ---------------------------------------------------------------- 共线编队
def _r_squared(pts: list[tuple[float, float]]) -> float:
    """对主轴做最小二乘, 返回 R^2。对竖直排列做坐标交换以避免斜率发散。"""
    n = len(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if max(xs) - min(xs) < max(ys) - min(ys):
        xs, ys = ys, xs
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return 1.0
    return (sxy * sxy) / (sxx * syy)


def _axis_project(centers: list[tuple[float, float]]) -> list[float]:
    """把点投影到主轴上, 返回排序后的投影坐标。用于算间距。"""
    n = len(centers)
    mx = sum(p[0] for p in centers) / n
    my = sum(p[1] for p in centers) / n
    sxx = sum((p[0] - mx) ** 2 for p in centers)
    syy = sum((p[1] - my) ** 2 for p in centers)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in centers)
    # 主轴方向 = 协方差矩阵最大特征向量
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(theta), math.sin(theta)
    return sorted((p[0] - mx) * ux + (p[1] - my) * uy for p in centers), (ux, uy)


# ---------------------------------------------------------------- 否决记账
# 规则跑完只说"convoy 0"是没法排查的 —— 到底是根本没有候选目标, 还是候选一路
# 过关却卡在最后的 require_any? 前者要补数据, 后者改一行配置就行, 两件事的
# 成本差着量级。所以每一处 return [] 都记下死在哪一关, 跑完按类打出来。
REJECTS: dict[str, "Counter[str]"] = defaultdict(Counter)


def _veto(cls_id: str, reason: str) -> list:
    REJECTS[cls_id][reason] += 1
    return []


def reject_report() -> str:
    if not REJECTS:
        return ""
    out = ["\n候选为什么没成事件(按类):"]
    for cls_id in sorted(REJECTS):
        items = REJECTS[cls_id].most_common()
        out.append(f"  {cls_id}: " + ", ".join(f"{k} {v}" for k, v in items))
    out.append("  ↑ 「数据源无此类目标」= 这批数据压根不含该规则要的标注, "
               "正常, 不用管;")
    out.append("    「不含军事目标」占多数 = 队形/规模都够, 只差 require_any, "
               "改一行配置就能放出来;")
    out.append("    「目标数不足」「单簇规模不足」占多数 = 门槛高了或素材本来就稀, "
               "先看 relax 档能捡回多少。")
    return "\n".join(out)


def derive_linear_formation(scene: Scene, rule: dict[str, Any], cls_id: str,
                           subtype: str | None = None) -> list[Event]:
    """车队 / 列队机动。

    **只看"共线 + 数量"是不够的** —— 正常道路上的车流同样共线, 这正是这条规则
    当初被停用的原因。车队区别于车流的地方有三个, 缺一不可:
      1. 间距均匀    车队保持队形, 间距变异系数小; 车流走走停停, 间距忽大忽小;
      2. 车型一致    车队是同一批装备; 车流是轿车卡车客车混在一起;
      3. 细长        沿主轴拉得很长、垂直方向很窄。停车场也共线, 但它是一片不是一条。
    再叠加 require_any 的军事目标约束, 民用卡车列队就落到困难负样本里去 ——
    和装备集结同一个套路, 那个已经验证有效。
    """
    targets = {c.lower() for c in rule["target_classes"]}
    all_objs = scene.of_classes(targets)
    if len(all_objs) < rule["min_count"]:
        # 一个都没有 ≠ 有但不够。前者说明这个数据源压根不含这类标注(比如
        # DroneCrowd 只有人头点, 没有车), 属于"本来就轮不到这条规则", 不是被否决;
        # 混在一起报会把真正该看的信号淹掉。
        return _veto(cls_id, "数据源无此类目标" if not all_objs
                     else f"目标数不足({len(all_objs)}<{rule['min_count']})")

    # **找共线的子集, 不是要求全图共线。** 这是这条规则最早的致命缺陷:
    # 原来对画面里所有车一起算 R², 于是一支七辆车的车队旁边只要有十辆散车,
    # 整体 R² 就掉下去, 车队被判"不共线"。真实航拍图里车队周围永远有别的车,
    # 所以全库跑下来 convoy 只有 49 条, 而"不共线"的否决多达 13983 —— 那不是
    # 没有车队, 是规则看不见它们。一图两支车队同理会互相拉低 R²。
    out: list[Event] = []
    covered: set[int] = set()
    remaining = list(all_objs)
    for _ in range(int(rule.get("max_lines", 3))):
        if len(remaining) < rule["min_count"]:
            break
        found = None
        for cand in _line_candidates(remaining, rule):
            # 一条线上可能混着车队和散车。**取其中间距均匀的最长一段**, 而不是
            # 整条线一起判 —— 否则线上多蹭进两辆散车就把整支车队否掉了。
            run = _even_run(cand, rule)
            if run is None:
                continue
            ev = _check_column(scene, run, rule, cls_id)
            if ev is not None:
                found = (run, ev)
                break
        if found is None:
            break
        run, ev = found
        out.append(ev)
        # **只摘掉判定通过的那些目标。** 早先是不管过没过都从 remaining 里删掉,
        # 于是一条蹭出来的假线会把真车队的成员一起吃掉。
        ids = {id(o) for o in run}
        covered |= ids
        remaining = [o for o in remaining if id(o) not in ids]

    if not out:
        return _veto(cls_id, "不共线")

    # **最后一道闸: 画面里的车必须基本都编进了队列。**
    # 实测过: 15 辆随机散车就能凑出一条间距 1.49 车长的线, 而最紧的真车队九成
    # 分位才 1.63; 80 辆以上时随机线能压到 0.8 车长, 比任何真车队都紧。也就是说
    # 靠共线度/间距/整齐度这些几何量**分不开**车队和密集车流 —— 停车场的一排车
    # 本身就是共线 + 等距 + 细长。
    # 唯一还站得住的信号是覆盖率: 一支车队开在空旷路段上, 画面里的车就是它;
    # 而从满图散车里凑出来的线永远只占一小撮。
    # 按**所有队列的并集**算, 不是单条 —— 一图两支车队(此外无车)应当算数。
    # 代价说清楚: 城市密集车流里的车队一律抓不到。这是有意的取舍, 那种场景本来
    # 就分不出车队和车流, 硬判只会制造假数据, 与"不要为了凑数据破坏精度"冲突。
    ratio = len(covered) / max(1, len(all_objs))
    if ratio < float(rule.get("min_inlier_ratio", 0.6)):
        # 记账按档分桶。早先把百分比直接写进原因, 于是每个百分比占一行,
        # 一份报告里冒出十几行同一类否决, 反而看不出重点。
        b = "不足两成" if ratio < 0.2 else "两到四成" if ratio < 0.4 else "四到六成"
        _veto(cls_id, f"成队的车只占同类目标的{b}(周围散车太多, 分不出车队)")
        return []
    return out


def _even_run(objs: list[Obj], rule: dict[str, Any]) -> list[Obj] | None:
    """一条线上间距最均匀的最长连续一段。

    车队的定义就是"沿线首尾相接、间距大致相等的一串"。线上蹭进来的散车会让
    整体间距变异系数爆掉, 但把它们排除后中间那一段仍然是车队 —— 所以这里按
    投影排序后滑窗, 找最长的、间距 CV 达标的连续段。
    """
    if len(objs) < rule["min_count"]:
        return None
    proj, _ = _axis_project([o.center for o in objs])
    order = sorted(range(len(objs)), key=lambda k: proj[k])
    ps = [proj[k] for k in order]
    thr = float(rule.get("max_spacing_cv", 0.45))
    best: tuple[int, int] = (0, 0)
    for i in range(len(ps)):
        for j in range(i + int(rule["min_count"]), len(ps) + 1):
            gaps = [b - a for a, b in zip(ps[i:j], ps[i + 1:j])]
            m = sum(gaps) / len(gaps)
            if m <= 1e-6:
                continue
            cv = (sum((g - m) ** 2 for g in gaps) / len(gaps)) ** 0.5 / m
            if cv <= thr and (j - i) > (best[1] - best[0]):
                best = (i, j)
    if best[1] - best[0] < rule["min_count"]:
        return None
    return [objs[order[k]] for k in range(best[0], best[1])]


def _line_candidates(objs: list[Obj], rule: dict[str, Any]) -> list[list[Obj]]:
    """RANSAC: 按内点数从多到少给出若干条候选线。

    **不能只返回内点最多的那一条。** 三十辆散车里总能拉出一条蹭到十辆的线,
    它的内点比七辆车的真车队还多; 只看最多的那条, 真车队永远轮不上。
    所以给出前 K 条候选, 由调用方逐条跑完整判据, 谁先通过用谁。

    带宽取目标尺寸的若干倍 —— 车队沿路行进时中心点不会正好共线, 但偏离不会
    超过一两个车身宽。按画幅比例取带宽会随裁剪尺寸漂移, 和 cluster_eps 同一个道理。
    随机数固定种子, 保证重跑结果一致。
    """
    n = len(objs)
    if n < rule["min_count"]:
        return []
    cs = [o.center for o in objs]
    sizes = sorted(max(o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]) for o in objs)
    band = sizes[len(sizes) // 2] * float(rule.get("band_obj_mult", 1.5))

    pairs: list[tuple[int, int]] = []
    budget = int(rule.get("ransac_pairs", 3000))
    if n * (n - 1) // 2 <= budget:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        rng = random.Random(0)                 # 固定种子: 同一份数据重跑结果一致
        seen = set()
        while len(pairs) < budget:
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j and (key := (min(i, j), max(i, j))) not in seen:
                seen.add(key)
                pairs.append(key)

    seen_sets: set[tuple[int, ...]] = set()
    cands: list[list[int]] = []
    for i, j in pairs:
        x1, y1 = cs[i]
        x2, y2 = cs[j]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        ux, uy = dx / L, dy / L
        inl = tuple(k for k, (x, y) in enumerate(cs)
                    if abs(-(x - x1) * uy + (y - y1) * ux) <= band)
        if len(inl) >= rule["min_count"] and inl not in seen_sets:
            seen_sets.add(inl)
            cands.append(list(inl))
    cands.sort(key=len, reverse=True)
    return [[objs[k] for k in c] for c in cands[: int(rule.get("max_candidates", 40))]]


def _check_column(scene: Scene, objs: list[Obj], rule: dict[str, Any],
                  cls_id: str) -> "Event | None":
    """对一条候选车列跑完整判据。不合格返回 None 并记账。"""
    centers = [o.center for o in objs]
    r2 = _r_squared(centers)
    if r2 < rule["collinearity_r2"]:
        _veto(cls_id, "不共线")
        return None

    proj, axis = _axis_project(centers)
    gaps = [b - a for a, b in zip(proj, proj[1:])]
    if not gaps:
        return None
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 1e-6:
        return None
    cv = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean_gap
    if cv > rule.get("max_spacing_cv", 0.45):
        _veto(cls_id, "间距不均(是车流不是车队)")
        return None

    # **车队是"挨着走的一串", 这一条才是真正分得开的判据。**
    # 实测过: 拿 40 辆随机散车去凑最整齐的一条线, 它的间距变异系数能低到 0.077,
    # 比带抖动的真车队(中位 0.156)还"整齐" —— 所以收紧 CV 只会误杀真车队,
    # 分不开假的。真正拉开差距的是**平均间距相对车身**:
    #     真车队   中位 3.8 车长, 最大 4.0
    #     随机凑线 最小 5.8 车长, 中位 9.3
    # 物理上也说得通: 保持队形就得跟得紧, 而散布在全画幅的车沿任一条线都拉得很开。
    sizes = sorted(max(o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]) for o in objs)
    car = sizes[len(sizes) // 2] or 1.0
    gap_mult = mean_gap / car
    if gap_mult > float(rule.get("max_gap_obj_mult", 5.0)):
        _veto(cls_id, f"间距过大({gap_mult:.1f}倍车身, 是散布不是车队)")
        return None

    # 细长度: 沿主轴的跨度 / 垂直方向的跨度
    ux, uy = axis
    perp = [-(p[0] - centers[0][0]) * uy + (p[1] - centers[0][1]) * ux for p in centers]
    span_long = max(proj) - min(proj)
    span_perp = max(perp) - min(perp)
    elong = span_long / max(1.0, span_perp)
    if elong < rule.get("min_elongation", 4.0):
        _veto(cls_id, "不够细长(是一片, 归集结)")
        return None

    by_cls: dict[str, int] = {}
    for o in objs:
        by_cls[o.cls.lower()] = by_cls.get(o.cls.lower(), 0) + 1
    main_cls, main_n = max(by_cls.items(), key=lambda kv: kv[1])
    if main_n / len(objs) < rule.get("min_same_class_ratio", 0.7):
        _veto(cls_id, "车型杂(是车流不是车队)")
        return None

    require = {c.lower() for c in rule.get("require_any", [])}
    if require and not (set(by_cls) & require):
        if rule.get("on_require_fail") == "hard_negative":
            scene.meta["hard_negative"] = True
            scene.meta.setdefault("hard_negative_reason", []).append(
                f"{cls_id}: {len(objs)} 个目标列队行进且间距均匀, 但均为民用车辆")
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            scene.meta.setdefault("hard_negative_bbox",
                                  [min(xs), min(ys), max(xs), max(ys)])
        _veto(cls_id, "不含军事目标(require_any)")
        return None

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    # 军用还是民用不再决定"算不算 convoy", 但必须记下来 —— 描述题要靠它区分
    # 「装甲车队」和「卡车车队」, 配额也要靠它保证两种都有。
    mil = {c.lower() for c in rule.get("military_classes", [])}
    n_mil = sum(1 for o in objs if o.cls.lower() in mil)
    return Event(
        type=cls_id,
        conf=round(min(1.0, r2 * (1.0 - cv)), 3),
        evidence={"rule": "linear_formation", "count": len(objs), "r2": round(r2, 4),
                  "spacing_cv": round(cv, 3), "elongation": round(elong, 2),
                  "gap_per_object": round(gap_mult, 2),
                  "main_class": main_cls,
                  "military_grade": ("military" if n_mil >= max(1, len(objs) // 2)
                                     else "mixed" if n_mil else "civil"),
                  "n_military": n_mil,
                  "cluster_bbox": [min(xs), min(ys), max(xs), max(ys)],
                  "object_ids": [o.id for o in objs]},
    )


# ---------------------------------------------------------------- 越界
def _seg_intersect(p1, p2, p3, p4) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _side(p, a, b) -> int:
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _point_in_polygon(p, poly) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def derive_boundary_cross(scene: Scene, rule: dict[str, Any], cls_id: str,
                         subtype: str | None = None) -> list[Event]:
    if not scene.regions or not scene.tracks:
        return []
    targets = {c.lower() for c in rule["target_classes"]}
    cls_by_track = {str(o.track_id): o.cls.lower() for o in scene.objects if o.track_id is not None}
    min_disp = rule.get("min_displacement_ratio", 0.02) * scene.diag

    events = []
    for tid, pts in scene.tracks.items():
        if cls_by_track.get(tid) not in targets:
            continue
        path = [(p[1], p[2]) for p in sorted(pts, key=lambda p: p[0])]
        if len(path) < 2:
            continue
        if math.dist(path[0], path[-1]) < min_disp:
            continue          # 位移过小, 视为原地抖动

        for region in scene.regions:
            crossed, direction = False, None
            if region.type == "polyline":
                for i in range(len(path) - 1):
                    for j in range(len(region.points) - 1):
                        if _seg_intersect(path[i], path[i + 1], region.points[j], region.points[j + 1]):
                            crossed = True
                            break
                    if crossed:
                        break
                if crossed:
                    a, b = region.points[0], region.points[-1]
                    direction = f"{_side(path[0], a, b)}->{_side(path[-1], a, b)}"
            elif region.type == "polygon":
                was_in = _point_in_polygon(path[0], region.points)
                now_in = _point_in_polygon(path[-1], region.points)
                crossed = was_in != now_in
                direction = "enter" if (not was_in and now_in) else "exit"

            if crossed:
                events.append(Event(
                    type=cls_id,
                    conf=1.0,
                    evidence={"rule": "boundary_cross", "track_id": tid,
                              "region": region.name, "region_type": region.type,
                              "direction": direction, "cls": cls_by_track.get(tid),
                              "start": list(path[0]), "end": list(path[-1])},
                ))
    return events


# ---------------------------------------------------------------- 驱动
_DISPATCH = {
    "density_cluster": derive_density_cluster,
    "linear_formation": derive_linear_formation,
    "boundary_cross": derive_boundary_cross,
}


def collect_rules(ontology: dict[str, Any]) -> list[tuple[str, str | None, dict[str, Any]]]:
    """展开为 (class_id, subtype, rule) 列表。停用的类别(enabled: false)跳过。"""
    out = []
    for cls in ontology["classes"]:
        if not cls.get("enabled", True):
            continue
        if cls.get("rule", {}).get("kind") in _DISPATCH:
            out.append((cls["id"], None, cls["rule"]))
        for st in cls.get("subtypes", []):
            if st.get("rule", {}).get("kind") in _DISPATCH:
                out.append((cls["id"], st["name"], st["rule"]))
    return out


def resolve_overlap(scene: Scene) -> None:
    """车队优先于集结。

    同一批车排成一列, 密度聚类同样会判成"密集成簇" —— 但"聚成一片"和"拉成一条"
    是两回事, 判定答案说"异常聚集"就错了。两个事件覆盖同一批目标时, 保留更具体
    的那个(车队)。

    **放宽档补量之后必须再调一次**: relax_pass 跑在 derive 之后, 会把同一批目标
    重新补一个集结事件回去, 只在第一遍做互斥是兜不住的。
    """
    conv_ids = {i for e in scene.events if e.type == "convoy"
                for i in e.evidence.get("object_ids", [])}
    if not conv_ids:
        return
    kept = []
    for e in scene.events:
        ids = set(e.evidence.get("object_ids", []))
        overlap = len(ids & conv_ids) / max(1, len(ids))
        if e.type != "convoy" and e.evidence.get("rule") == "density_cluster" \
                and overlap >= 0.6:
            continue
        kept.append(e)
    scene.events = kept


def derive(scenes: list[Scene], ontology: dict[str, Any], overwrite: bool = False) -> list[Scene]:
    rules = collect_rules(ontology)
    # **本体的 enabled 要在这里统一执行, 不能只在 collect_rules 里管。**
    # collect_rules 只管本文件自己算的那几条规则; 而适配器在 prepare 阶段会
    # 直接把事件挂在 scene 上(era.py 的 ANOMALY 映射就是), 完全绕过本体。
    # 于是 disaster 明明 enabled: false, 全库照样出了 528 条事件, 一路会走到
    # build_vqa 生成灾害题 —— 而灾害这一类已经决定不做了。
    enabled = {c["id"] for c in ontology["classes"] if c.get("enabled", True)}
    dropped: Counter = Counter()
    for scene in scenes:
        kept = [e for e in scene.events if e.type in enabled]
        if len(kept) != len(scene.events):
            dropped.update(e.type for e in scene.events if e.type not in enabled)
            scene.events = kept
    if dropped:
        print(f"按本体的 enabled 丢弃了 {sum(dropped.values())} 条停用类别的事件: "
              f"{dict(dropped.most_common())}")
        print("  (这些是适配器在 prepare 阶段直接挂上的, 不经过本文件的规则;"
              "\n   要保留就把 configs/ontology.yaml 里对应类的 enabled 改回 true)")
    for scene in scenes:
        if overwrite:
            # **只清本文件能重算的那几种事件**。原来的写法是"丢掉所有带 rule 字段的",
            # 但适配器给的标签迁移事件(烟雾/爆炸/灾害)同样带 rule: label_transfer,
            # 而 _DISPATCH 里没有它 —— 清掉就再也算不回来了。
            # 实测: --overwrite 一跑, smoke/explosion/disaster 三类全部消失。
            scene.events = [e for e in scene.events
                            if e.evidence.get("rule") not in _DISPATCH]
            scene.meta.pop("hard_negative", None)
            scene.meta.pop("hard_negative_reason", None)
            scene.meta.pop("hard_negative_bbox", None)
        existing = {(e.type, str(sorted(e.evidence.items()))) for e in scene.events}
        for cls_id, subtype, rule in rules:
            for ev in _DISPATCH[rule["kind"]](scene, rule, cls_id, subtype):
                key = (ev.type, str(sorted(ev.evidence.items())))
                if key not in existing:
                    existing.add(key)
                    scene.events.append(ev)
        resolve_overlap(scene)

        # 事件与困难负样本互斥: 有真事件就不是负样本
        if scene.events:
            scene.meta.pop("hard_negative", None)
            scene.meta.pop("hard_negative_reason", None)
    return scenes


def _class_key(e: Event) -> str:
    return e.type + (f"/{e.evidence['subtype']}" if "subtype" in e.evidence else "")


def relax_pass(scenes: list[Scene], ontology: dict[str, Any], below: int,
               max_ratio: float = 0.5) -> dict[str, int]:
    """补量: 只对产量不够的类别启用 relax 档, 且只在还没出事件的图上跑。

    「多的筛精、少的放宽」里放宽的这一半。要点有三:
      1. 只放宽不够的类 —— explosion/smoke 本来就富余, 放宽只会拉低质量;
      2. 放宽出来的事件打 relaxed 标记, 下游答案改用"小规模/迹象"这类说法,
         不冒充典型样本;
      3. 补量上限 max_ratio: 放宽样本最多占该类的一半, 否则这一类的分布
         会被边缘样本主导, 模型学到的就是"稍微聚一下就算集结"。
    """
    rules = [(c, st, r) for c, st, r in collect_rules(ontology) if r.get("relax")]
    if not rules:
        return {}
    have: dict[str, int] = {}
    for s in scenes:
        for k in {_class_key(e) for e in s.events}:
            have[k] = have.get(k, 0) + 1

    added: dict[str, int] = {}
    for cls_id, subtype, rule in rules:
        key = cls_id + (f"/{subtype}" if subtype else "")
        cur = have.get(key, 0)
        if cur >= below:
            continue
        budget = int(max(below - cur, 0) if cur == 0 else
                     min(below - cur, cur * max_ratio / (1 - max_ratio)))
        if budget <= 0:
            continue
        merged = {**rule, **rule["relax"]}
        merged.pop("relax", None)
        merged.pop("on_require_fail", None)   # 放宽档不再制造困难负样本, 避免与严格档打架
        n = 0
        for scene in scenes:
            if n >= budget:
                break
            if any(_class_key(e) == key for e in scene.events):
                continue                       # 严格档已经命中, 不重复补
            evs = _DISPATCH[merged["kind"]](scene, merged, cls_id, subtype)
            if not evs:
                continue
            for ev in evs:
                ev.evidence["relaxed"] = True
                ev.conf = round(ev.conf * 0.7, 3)
                scene.events.append(ev)
            scene.meta.pop("hard_negative", None)
            scene.meta.pop("hard_negative_reason", None)
            n += 1
        if n:
            added[key] = n
    for scene in scenes:
        resolve_overlap(scene)      # 补量可能把让过位的集结又加回来了
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="从检测/跟踪标注派生异常事件标签")
    ap.add_argument("--scenes", required=True, help="输入 scene jsonl")
    ap.add_argument("--out", required=True, help="输出 scene jsonl")
    ap.add_argument("--ontology", default="configs/ontology.yaml")
    ap.add_argument("--overwrite", action="store_true", help="丢弃已有的规则派生事件后重算")
    ap.add_argument("--relax-below", type=int, default=0,
                    help="产量低于这个图数的类别启用 ontology 里的 relax 档补量, "
                         "0 表示不补。补出来的事件带 relaxed 标记")
    ap.add_argument("--relax-max-ratio", type=float, default=0.5,
                    help="放宽样本在该类中的占比上限, 默认最多一半")
    args = ap.parse_args()

    ontology = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    scenes = derive(load_scenes(args.scenes), ontology, args.overwrite)
    if args.relax_below:
        added = relax_pass(scenes, ontology, args.relax_below, args.relax_max_ratio)
        if added:
            print("放宽档补量(仅限产量不够的类, 事件标 relaxed):")
            for k, v in sorted(added.items(), key=lambda kv: -kv[1]):
                print(f"  {k:24s} +{v}")
        else:
            print("放宽档未启用: 各类产量都够, 或没有可补的图")
    dump_scenes(scenes, args.out)

    hist: dict[str, int] = {}
    for s in scenes:
        for e in s.events:
            key = e.type + (f"/{e.evidence['subtype']}" if "subtype" in e.evidence else "")
            hist[key] = hist.get(key, 0) + 1
    n_normal = sum(1 for s in scenes if not s.events)
    n_hard = sum(1 for s in scenes if s.meta.get("hard_negative"))
    print(f"scenes={len(scenes)}  无事件(正常)={n_normal}  其中困难负样本={n_hard}")
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v}")
    if n_hard:
        print(f"\n  困难负样本 = 规模达标但不含军事目标的聚集(如民用停车场), "
              f"占正常样本 {n_hard / max(1, n_normal):.1%}")
    # 本体里启用了却一条都没产出的类, 必须点名 —— 这是开跑前最该拦住的事。
    enabled = {c["id"] for c in ontology.get("classes", [])
               if c["id"] != "normal" and c.get("enabled", True)}
    got = {k.split("/")[0] for k in hist}
    if (zero := enabled - got):
        print(f"\n⚠ 启用了但一条都没触发的类: {sorted(zero)}")
    if (rep := reject_report()):
        print(rep)


if __name__ == "__main__":
    main()

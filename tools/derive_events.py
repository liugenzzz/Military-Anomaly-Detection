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
from pathlib import Path
from typing import Any

import yaml

from scene import Event, Scene, dump_scenes, load_scenes


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
        return []

    require = {c.lower() for c in rule.get("require_any", [])}
    on_fail = rule.get("on_require_fail")
    eps = cluster_eps(objs, scene.diag, rule)
    centers = [o.center for o in objs]
    events = []
    for group in _union_find_cluster(centers, eps):
        if len(group) < rule["min_cluster_size"]:
            continue
        cls_in_group = {objs[i].cls.lower() for i in group}
        if require and not (cls_in_group & require):
            if on_fail == "hard_negative":
                scene.meta["hard_negative"] = True
                scene.meta.setdefault("hard_negative_reason", []).append(
                    f"{cls_id}/{subtype or '-'}: {len(group)} 个目标密集成簇但不含军事目标")
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


def derive_linear_formation(scene: Scene, rule: dict[str, Any], cls_id: str,
                           subtype: str | None = None) -> list[Event]:
    targets = {c.lower() for c in rule["target_classes"]}
    objs = scene.of_classes(targets)
    if len(objs) < rule["min_count"]:
        return []
    centers = [o.center for o in objs]
    r2 = _r_squared(centers)
    if r2 < rule["collinearity_r2"]:
        return []
    return [Event(
        type=cls_id,
        conf=float(r2),
        evidence={"rule": "linear_formation", "count": len(objs), "r2": round(r2, 4),
                  "object_ids": [o.id for o in objs]},
    )]


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


def derive(scenes: list[Scene], ontology: dict[str, Any], overwrite: bool = False) -> list[Scene]:
    rules = collect_rules(ontology)
    for scene in scenes:
        if overwrite:
            scene.events = [e for e in scene.events if e.evidence.get("rule") is None]
            scene.meta.pop("hard_negative", None)
            scene.meta.pop("hard_negative_reason", None)
        existing = {(e.type, str(sorted(e.evidence.items()))) for e in scene.events}
        for cls_id, subtype, rule in rules:
            for ev in _DISPATCH[rule["kind"]](scene, rule, cls_id, subtype):
                key = (ev.type, str(sorted(ev.evidence.items())))
                if key not in existing:
                    existing.add(key)
                    scene.events.append(ev)
        # 事件与困难负样本互斥: 有真事件就不是负样本
        if scene.events:
            scene.meta.pop("hard_negative", None)
            scene.meta.pop("hard_negative_reason", None)
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description="从检测/跟踪标注派生异常事件标签")
    ap.add_argument("--scenes", required=True, help="输入 scene jsonl")
    ap.add_argument("--out", required=True, help="输出 scene jsonl")
    ap.add_argument("--ontology", default="configs/ontology.yaml")
    ap.add_argument("--overwrite", action="store_true", help="丢弃已有的规则派生事件后重算")
    args = ap.parse_args()

    ontology = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    scenes = derive(load_scenes(args.scenes), ontology, args.overwrite)
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


if __name__ == "__main__":
    main()

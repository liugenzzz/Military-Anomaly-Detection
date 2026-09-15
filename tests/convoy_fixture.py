"""convoy（车队）判定的对照样本。

这条规则最早的致命缺陷是**对全图所有车一起算共线度** —— 一支七辆车的车队旁边
只要有十辆散车，整体 R² 就掉下去，车队被判"不共线"。真实航拍图里车队周围永远
有别的车，所以全库跑下来 convoy 只有 49 条，而"不共线"的否决多达 13983。

改成搜共线子集之后，反过来出现假阳性：四十辆纯随机散车里能"找出"两支车队。
实测量过两者的分布（`python tests/convoy_fixture.py --measure`）：

    真车队(间距抖动±20%)     CV 中位 0.156
    随机散车凑出的最好线      CV 最小 0.077   ← 比真车队还整齐

也就是说**靠共线度/间距整齐度分不开**，收紧 CV 只会误杀真车队。停车场的一排车
本身就是共线 + 等距 + 细长。唯一还站得住的信号是覆盖率：画面里的车基本都编进了
队列。代价是密集车流里的车队一律抓不到 —— 这是有意的取舍。

    python tests/convoy_fixture.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import yaml  # noqa: E402

from derive_events import derive  # noqa: E402
from scene import Obj, Scene  # noqa: E402

RNG = random.Random(3)
W, H, CW, CL = 1400, 1050, 26, 50


def _scene(sid: str, objs: list[tuple[str, list[float]]]) -> Scene:
    return Scene(image_id=sid, image_path=f"/x/{sid}.png", width=W, height=H,
                 source_dataset="PROBE", view="uav",
                 objects=[Obj(id=i, cls=c, bbox=b) for i, (c, b) in enumerate(objs)])


def column(n: int, x0: float, y0: float, dx: float, dy: float,
           cls: str = "truck") -> list[tuple[str, list[float]]]:
    return [(cls, [x0 + i * dx, y0 + i * dy, x0 + i * dx + CW, y0 + i * dy + CL])
            for i in range(n)]


def scatter(n: int, cls: str = "vehicle") -> list[tuple[str, list[float]]]:
    out = []
    for _ in range(n):
        x, y = RNG.randint(50, W - 100), RNG.randint(50, H - 100)
        out.append((cls, [x, y, x + 24, y + 45]))
    return out


# (场景, 期望的车队数) —— 期望值是判定意图, 不是"当前跑出来的结果"
CASES: list[tuple[Scene, int, str]] = [
    (_scene("A_pure_column", column(7, 200, 600, 95, -18)), 1,
     "纯车队: 全图只有这一列"),
    (_scene("E_two_columns", column(6, 150, 200, 90, 12) + column(6, 300, 800, 85, -10)), 2,
     "一图两支车队, 此外无车 —— 覆盖率要按并集算, 不能按单条"),
    (_scene("B_column_in_traffic", column(7, 200, 600, 95, -18) + scatter(30)), 0,
     "车队淹在散车里 —— **有意放弃**: 这种场景分不出车队和车流"),
    (_scene("D_scatter_only", scatter(40)), 0,
     "纯随机散车 —— 绝不能'找出'车队"),
    (_scene("F_parking", [("tank", [300 + (i % 3) * 70, 300 + (i // 3) * 70,
                                    300 + (i % 3) * 70 + CW, 300 + (i // 3) * 70 + CL])
                          for i in range(9)]), 0,
     "停车场式一片: 不细长, 该归 massing"),
    (_scene("G_only4", column(4, 200, 600, 95, -18)), 0,
     "只有四辆, 低于 min_count"),
    (_scene("H_uneven", [("truck", [200 + x, 600, 200 + x + CW, 650])
                         for x in (0, 95, 260, 300, 520, 560)]), 0,
     "间距忽大忽小: 是走走停停的车流"),
    (_scene("I_mixed", [(c, [200 + i * 95, 600 - i * 18, 200 + i * 95 + CW,
                             600 - i * 18 + CL])
                        for i, c in enumerate(["truck", "vehicle", "truck",
                                               "vehicle", "truck", "vehicle"])]), 0,
     "车型杂: 是混杂车流"),
]


def main() -> int:
    onto = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs/ontology.yaml")
        .read_text(encoding="utf-8"))
    fail = []
    for sc, want, why in CASES:
        derive([sc], onto, overwrite=True)
        got = sum(1 for e in sc.events if e.type == "convoy")
        ok = got == want
        if not ok:
            fail.append(f"{sc.image_id}: 期望 {want} 支车队, 实得 {got}  ({why})")
        print(f"  {'✓' if ok else '✗'} {sc.image_id:22s} {got} 支  {why}")
    if fail:
        print("\n" + "\n".join("  ✗ " + f for f in fail))
        return 1
    print(f"\n{len(CASES)} 个对照场景全部判定正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

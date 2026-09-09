"""自动构建 golden set —— 不依赖人工精修。

三层来源, 前两层零人工成本:
  L1 自动可验证    计数题/grounding 题, 答案 100% 由 bbox 标注决定, 天然是真值
  L2 借来的人工标注 从 CapERA / HIVAU-70k / ERA 等本就人工精标的数据里切出保留集
  L3 二选一审核    LLM 生成的描述/推理题, 人只判"合格/不合格", 不写答案(可选)

关键约束: golden set 里的 image_id 必须从训练数据中彻底排除。本脚本同时输出
exclude_ids.txt, build_vqa.py 与 llm_qa.py 用 --exclude-ids 读取它。

用法:
  python tools/make_golden.py --scenes data/screened/scenes_kept.jsonl --out-dir data/golden
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from build_vqa import RuleBuilder
from scene import Scene, dump_scenes, load_scenes

# 这些数据集自带人工精标, L2 直接从中借用, 无需自己标注
HUMAN_ANNOTATED = {"CapERA", "ERA", "HIVAU-70k", "UCA", "UCF-Crime-UCA", "MOCO"}


def stratified_pick(scenes: list[Scene], per_cell: int, seed: int) -> list[Scene]:
    """按 (数据源 x 异常类) 分层取样。随机取样会让小类别完全抽不到。"""
    cells: dict[tuple[str, str], list[Scene]] = defaultdict(list)
    for s in scenes:
        for a in (s.anomaly_types or ["normal"]):
            cells[(s.source_dataset, a)].append(s)

    rng = random.Random(seed)
    picked: dict[str, Scene] = {}
    for key in sorted(cells):
        pool = cells[key][:]
        rng.shuffle(pool)
        for s in pool[:per_cell]:
            picked[s.image_id] = s          # 同一图可能命中多格, 去重
    return list(picked.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="自动构建 golden set(零人工)")
    ap.add_argument("--scenes", required=True, help="筛选后的 scene jsonl")
    ap.add_argument("--out-dir", default="data/golden")
    ap.add_argument("--ontology", default="configs/ontology.yaml")
    ap.add_argument("--prompt-dir", default="configs/prompts")
    ap.add_argument("--per-cell", type=int, default=12,
                    help="每个 (数据源 x 异常类) 单元取多少张图")
    ap.add_argument("--l2-sources", nargs="*", default=sorted(HUMAN_ANNOTATED),
                    help="视为人工精标、可直接借用的数据源名")
    ap.add_argument("--seed", type=int, default=20240101)
    args = ap.parse_args()

    onto = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    builder = RuleBuilder(onto, args.prompt_dir, seed=args.seed)
    scenes = load_scenes(args.scenes)

    l2_names = set(args.l2_sources)
    l1_pool = [s for s in scenes if s.objects]                       # 有框才能出真值题
    l2_pool = [s for s in scenes if s.source_dataset in l2_names]

    l1 = stratified_pick(l1_pool, args.per_cell, args.seed)
    l2 = stratified_pick(l2_pool, args.per_cell, args.seed + 1)

    l2_ids = {s.image_id for s in l2}
    l1 = [s for s in l1 if s.image_id not in l2_ids]                 # 两层不重叠

    samples: list[dict] = []
    for level, group in (("L1", l1), ("L2", l2)):
        for s in group:
            # L1 只取答案由标注唯一决定的题; L2 用人工精标源的判定与方位题
            if level == "L1":
                pairs = [(builder.count(s), "count"),
                         (builder.locate_box(s), "locate_box")]
            else:
                pairs = [(builder.judge(s), "judge"),
                         (builder.locate_verbal(s), "locate_verbal")]
            for turn, task in pairs:
                if not turn:
                    continue
                qa = builder._mk(s, [turn], task, golden_level=level,
                                 human_verified=(level == "L2"))
                samples.append(qa)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    exclude = sorted({s.image_id for s in l1 + l2})
    (out / "exclude_ids.txt").write_text("\n".join(exclude) + "\n", encoding="utf-8")
    dump_scenes(l1 + l2, out / "golden_scenes.jsonl")

    hist = Counter(s["extra"]["task"] for s in samples)
    by_src = Counter(s["extra"]["source_dataset"] for s in samples)
    print(f"golden set: {len(samples)} 条 QA / {len(exclude)} 张图")
    print(f"  L1 自动可验证(计数+grounding, 零人工): {sum(1 for s in samples if s['extra']['golden_level'] == 'L1')}")
    print(f"  L2 借用人工精标({', '.join(sorted(l2_names & {x.source_dataset for x in scenes})) or '本批无'}): "
          f"{sum(1 for s in samples if s['extra']['golden_level'] == 'L2')}")
    print("\n题型:", dict(hist))
    print("来源:", dict(by_src))
    print(f"\n已写出 {out / 'exclude_ids.txt'}")
    print("下一步务必带上它, 否则 golden set 会泄漏进训练集:")
    print(f"  python tools/build_vqa.py --scenes ... --exclude-ids {out / 'exclude_ids.txt'}")
    if not l2:
        print("\n[warn] 本批没有命中任何人工精标数据源, golden set 只有 L1 层。")
        print("       接入 ERA/CapERA/HIVAU-70k 后重跑, 才能覆盖描述与判定类题目。")


if __name__ == "__main__":
    main()

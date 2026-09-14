"""每种题型/描述侧面各抽一条, 导出成一个 jsonl 供人工过目。

    python tools/sample_examples.py --vqa-dir data/vqa --out examples.jsonl

用途是**验收**: 全量几万条没法逐条看, 但每个题型看一条, 问法对不对、答案形态
对不对、该给框的有没有给框, 一眼就能判出来。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="每种题型抽一条样例")
    ap.add_argument("--vqa-dir", nargs="+", required=True)
    ap.add_argument("--out", default="examples.jsonl")
    ap.add_argument("--per-kind", type=int, default=1, help="每种抽几条")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows: list[dict] = []
    for d in args.vqa_dir:
        for f in sorted(Path(d).glob("*.json")):
            if f.name in ("dataset_info.json", "report.json"):
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows += data

    # 分组键: 规则题按 task_type, 描述题按 facet —— describe 这一个 task 底下有
    # 二十多个侧面, 只按 task 抽的话它们全被折叠成一条, 恰恰是最该逐个过目的部分
    groups: dict[str, list[dict]] = {}
    for r in rows:
        ex = r.get("metadata") or r.get("extra") or {}
        key = ex.get("facet") if ex.get("gen") == "llm" else ex.get("task_type", ex.get("task", "?"))
        groups.setdefault(str(key), []).append(r)

    rng = random.Random(args.seed)
    out = []
    for key in sorted(groups):
        pool = groups[key][:]
        rng.shuffle(pool)
        # **原样导出训练文件里的那一行**, 不做任何改写 —— 验收看的就得是真格式,
        # 翻译成中文键看着舒服, 但看不出 conversations / from / value 对不对
        out += pool[:args.per_kind]

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n",
                 encoding="utf-8")
    print(f"{len(groups)} 种题型 / 共 {len(rows)} 条样本 -> 抽出 {len(out)} 条到 {p}")
    for k in sorted(groups):
        print(f"  {k:22s} {len(groups[k]):6d} 条")


if __name__ == "__main__":
    main()

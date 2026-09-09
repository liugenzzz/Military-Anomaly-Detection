"""VQA 数据体检: 图像可达性、坐标范围、类别与题型分布、正负样本比。

发布/开训前跑一遍, 大部分低级问题(路径错、坐标越界、负样本太少)都能在这里拦住。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BOX_RE = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="待检查的 VQA json")
    ap.add_argument("--box-scale", type=int, default=1000)
    ap.add_argument("--check-images", action="store_true", help="逐条检查图像文件是否存在(慢)")
    ap.add_argument("--min-negative-ratio", type=float, default=0.30)
    args = ap.parse_args()

    problems: list[str] = []
    qa_types, anomalies, sources = Counter(), Counter(), Counter()
    styles, turns = Counter(), Counter()
    facets, gens, imgs = Counter(), Counter(), Counter()
    total = 0
    missing = set()

    for fp in args.files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        for i, s in enumerate(data):
            total += 1
            msgs = s.get("messages", [])
            body = [m for m in msgs if m.get("role") != "system"]
            if len(body) < 2 or len(body) % 2 or any(
                    m["role"] != ("user" if k % 2 == 0 else "assistant")
                    for k, m in enumerate(body)):
                problems.append(f"{fp}#{i}: messages 必须是 system? + user/assistant 交替")
                continue
            turns[(len(body) // 2)] += 1
            n_tok = sum(m["content"].count("<image>") for m in body)
            if n_tok != len(s.get("images", [])):
                problems.append(f"{fp}#{i}: <image> 数量({n_tok}) 与 images 数量({len(s.get('images', []))}) 不一致")
            if body[0]["content"].count("<image>") != n_tok:
                problems.append(f"{fp}#{i}: <image> 应全部出现在首轮 user 消息中")
            answers = " ".join(m["content"] for m in body[1::2])
            for box in BOX_RE.findall(answers):
                x1, y1, x2, y2 = map(int, box)
                if not (0 <= x1 < x2 <= args.box_scale and 0 <= y1 < y2 <= args.box_scale):
                    problems.append(f"{fp}#{i}: 坐标越界或反向 {box}")
            if args.check_images:
                for im in s.get("images", []):
                    if not Path(im).exists():
                        missing.add(im)
            ex = s.get("extra", {})
            qa_types[ex.get("task", ex.get("qa_type", "?"))] += 1
            if ex.get("facet"):
                facets[ex["facet"]] += 1
            gens[ex.get("gen", "-")] += 1
            imgs[ex.get("n_images", 1)] += 1
            for a in (ex.get("anomaly") or ["?"]):
                anomalies[a] += 1
            sources[ex.get("source_dataset", "?")] += 1


    def show(title: str, c: Counter) -> None:
        print(f"\n{title}")
        for k, v in c.most_common():
            print(f"  {str(k):24s} {v:7d}  {v / max(1, total):6.1%}")

    print(f"样本总数 {total}")
    show("题型分布", qa_types)
    show("异常类别分布", anomalies)
    show("数据来源分布", sources)
    if facets:
        show("描述侧面分布", facets)
    show("生成方式", Counter({"规则" if k == "rule" else ("LLM" if k == "llm" else k): v
                              for k, v in gens.items()}))
    show("对话轮数分布", Counter({f"{k} 轮": v for k, v in turns.items()}))
    show("图片数分布", Counter({f"{k} 图": v for k, v in imgs.items()}))

    neg = anomalies.get("normal", 0) / max(1, total)
    print(f"\n正常(负)样本占比 {neg:.1%}  阈值 {args.min_negative_ratio:.0%}"
          f"  -> {'OK' if neg >= args.min_negative_ratio else '偏低, 请补充正常场景样本'}")
    if missing:
        print(f"\n缺失图像 {len(missing)} 个, 例如: {list(sorted(missing))[:5]}")
    if problems:
        print(f"\n发现 {len(problems)} 处问题:")
        for p in problems[:20]:
            print("  " + p)
    else:
        print("\n结构检查未发现问题。")


if __name__ == "__main__":
    main()

"""VQA 数据体检: 图像可达性、坐标范围、类别与题型分布、正负样本比。

发布/开训前跑一遍, 大部分低级问题(路径错、坐标越界、负样本太少)都能在这里拦住。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sharegpt import GPT, HUMAN, meta_of, turns_of  # noqa: E402

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
    facets, gens, imgs, mods = Counter(), Counter(), Counter(), Counter()
    cover = Counter()          # 训练要求的三件事各覆盖了多少条
    total = 0
    missing = set()

    for fp in args.files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        for i, s in enumerate(data):
            total += 1
            body = turns_of(s)
            if len(body) < 2 or len(body) % 2 or any(
                    m["role"] != (HUMAN if k % 2 == 0 else GPT)
                    for k, m in enumerate(body)):
                problems.append(f"{fp}#{i}: conversations 必须是 human/gpt 交替")
                continue
            turns[(len(body) // 2)] += 1
            field = "videos" if "videos" in s else "images"
            tok = "<video>" if field == "videos" else "<image>"
            n_tok = sum(m["content"].count(tok) for m in body)
            if n_tok != len(s.get(field, [])):
                problems.append(f"{fp}#{i}: {tok} 数量({n_tok}) 与 {field} 数量"
                                f"({len(s.get(field, []))}) 不一致")
            if body[0]["content"].count(tok) != n_tok:
                problems.append(f"{fp}#{i}: {tok} 应全部出现在首轮 user 消息中")
            if "images" in s and "videos" in s:
                problems.append(f"{fp}#{i}: 同时含 images 与 videos, LLaMA-Factory 无法加载")
            answers = " ".join(m["content"] for m in body[1::2])
            # 训练要求的三件事分别数一下 —— 光看题型分布看不出"文字+图像区域"覆盖到没有
            has_box = "bbox_2d" in answers
            prose = len(re.findall(r"[\u4e00-\u9fff]", re.sub(r"\{[^{}]*\}", "", answers)))
            if has_box and prose >= 30:
                cover["多模态描述(文字+图像区域)"] += 1
            elif has_box:
                cover["纯坐标(grounding)"] += 1
            elif prose >= 30:
                cover["纯文字描述/说明"] += 1
            else:
                cover["短答(判定/计数等)"] += 1
            for box in BOX_RE.findall(answers):
                x1, y1, x2, y2 = map(int, box)
                if not (0 <= x1 < x2 <= args.box_scale and 0 <= y1 < y2 <= args.box_scale):
                    problems.append(f"{fp}#{i}: 坐标越界或反向 {box}")
            if args.check_images:
                for im in s.get(field, []):
                    if not Path(im).exists():
                        missing.add(im)
            ex = meta_of(s)
            qa_types[ex.get("task_type", ex.get("task", "?"))] += 1
            if ex.get("facet"):
                facets[ex["facet"]] += 1
            gens[ex.get("gen", "-")] += 1
            imgs[ex.get("n_media", ex.get("n_images", 1))] += 1
            mods[ex.get("modality", "image")] += 1
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
    show("训练要求覆盖", cover)
    show("生成方式", Counter({"规则" if k == "rule" else ("LLM" if k == "llm" else k): v
                              for k, v in gens.items()}))
    show("对话轮数分布", Counter({f"{k} 轮": v for k, v in turns.items()}))
    show("输入形态", mods)
    show("媒体数分布", Counter({f"{k} 个": v for k, v in imgs.items()}))

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

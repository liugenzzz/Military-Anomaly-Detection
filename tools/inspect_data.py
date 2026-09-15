"""数据体检: 先看清楚数据长什么样, 再决定该问什么。

分两部分:
  A. 纯标注统计(不用模型, 秒级)  —— 类别分布、每图目标数、框大小、共现、事件触发率
  B. 让 VLM 自由描述抽样图像(用端点池) —— **不给任何约束, 就问"你看到了什么"**

B 是关键。设计问题之前得先知道图里到底有什么: 是俯拍机场停机坪还是街景?
目标占几个像素还是占半幅画? 烟是一柱还是漫天? 不看图就设计问法, 就会写出
"清点一下画面中的烟雾"这种问法 —— 标注里 smoke 有个框, 于是想当然地去数它。

    python tools/inspect_data.py --scenes data/all_ev.jsonl --out-dir data/inspect
    python tools/inspect_data.py --scenes data/all_ev.jsonl --out-dir data/inspect \
        --endpoints configs/endpoints.local.yaml --per-class 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import Scene, load_scenes  # noqa: E402

FREE_PROMPT = """你面前是一张航拍或监控画面。请用中文如实描述你看到的内容。

要求：
1. 先说这是什么场景（俯视角度？大概多高？城市/野外/机场/港口/道路？）；
2. 再说画面里有哪些东西，大致多少，占画面多大；
3. 如果有烟、火、人群、车队、飞机这类，说清楚它们的形态——是一柱烟还是一片，
   是几个人还是密密麻麻一片，看不看得清单个目标；
4. 最后说一句：这张图**看得清什么、看不清什么**。

不要猜测军事含义，不要推断国别或事件起因。看不清就说看不清。"""


def stat_annotations(scenes: list[Scene]) -> dict:
    by_ds: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "cls": Counter(), "objs": [], "area": [], "size": Counter(),
        "events": Counter(), "modality": Counter()})
    cooc = Counter()
    for s in scenes:
        d = by_ds[s.source_dataset]
        d["n"] += 1
        d["objs"].append(len(s.objects))
        d["modality"][s.modality] += 1
        d["size"][f"{s.width}x{s.height}"] += 1
        for o in s.objects:
            d["cls"][o.cls] += 1
            if s.width and s.height:
                d["area"].append(o.area / (s.width * s.height))
        types = sorted({e.type for e in s.events})
        for t in types:
            d["events"][t] += 1
        if len(types) > 1:
            cooc["+".join(types)] += 1
        if not types:
            d["events"]["(无事件)"] += 1
    return {"by_dataset": by_ds, "cooccurrence": cooc}


def _pct(v: list[float], q: float) -> float:
    if not v:
        return 0.0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


def report(stats: dict) -> str:
    out = ["# 标注统计\n"]
    for ds, d in sorted(stats["by_dataset"].items(), key=lambda kv: -kv[1]["n"]):
        objs = d["objs"]
        out.append(f"\n## {ds}   {d['n']} 个样本")
        out.append(f"- 每图目标数: 中位 {int(_pct(objs, .5))}, "
                   f"九成分位 {int(_pct(objs, .9))}, 最多 {max(objs) if objs else 0}")
        if d["area"]:
            a = d["area"]
            out.append(f"- 单目标占画幅: 中位 {_pct(a, .5):.2%}, "
                       f"一成分位 {_pct(a, .1):.3%}, 九成分位 {_pct(a, .9):.2%}")
            tiny = sum(1 for x in a if x < 0.001) / len(a)
            if tiny > 0.3:
                out.append(f"  - **{tiny:.0%} 的目标小于画幅千分之一** —— "
                           f"这类目标问颜色、部件、朝向都是为难模型")
        out.append(f"- 输入形态: {dict(d['modality'])}")
        out.append(f"- 图像尺寸(前 3): {dict(d['size'].most_common(3))}")
        out.append(f"- 类别(前 8): {dict(d['cls'].most_common(8))}")
        out.append(f"- 事件: {dict(d['events'].most_common(6))}")
    if stats["cooccurrence"]:
        out.append(f"\n## 多类共现\n{dict(stats['cooccurrence'].most_common(10))}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="数据体检: 标注统计 + VLM 自由描述抽样图")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out-dir", default="data/inspect")
    ap.add_argument("--per-class", type=int, default=0,
                    help="每个异常类让模型自由描述几张图。0 = 只做标注统计, 不调模型")
    ap.add_argument("--endpoints", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--inline-images", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = load_scenes(args.scenes)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = stat_annotations(scenes)
    (out / "annotation_stats.md").write_text(report(stats), encoding="utf-8")
    print(report(stats))
    print(f"\n-> {out / 'annotation_stats.md'}")

    if not args.per_class:
        print("\n(未指定 --per-class, 跳过 VLM 自由描述。"
              "加上它才能知道图里到底长什么样)")
        return

    if not args.endpoints and Path("configs/generate.yaml").exists():
        args.endpoints = "configs/generate.yaml"       # 与 ping 一致, 免得忘了传
        print(f"  端点池: {args.endpoints}")
    from llm_qa import DEFAULT_MODEL, LLM, image_message
    from concurrent.futures import ThreadPoolExecutor

    by_cls: dict[str, list[Scene]] = defaultdict(list)
    for s in scenes:
        by_cls[(s.anomaly_types or ["normal"])[0]].append(s)
    rng = random.Random(args.seed)
    picked: list[Scene] = []
    for cls, pool in by_cls.items():
        p = pool[:]
        rng.shuffle(p)
        picked += p[:args.per_class]
    print(f"\n抽 {len(picked)} 张图让模型自由描述 ({len(by_cls)} 个类)")

    llm = LLM(args.model or DEFAULT_MODEL, args.base_url,
              endpoints_file=args.endpoints, role="generate", temperature=0.2)
    print(f"  {llm.describe()}")
    workers = args.workers or llm.total_concurrency

    def run(s: Scene):
        msg = [{"role": "user",
                "content": image_message(FREE_PROMPT, s.image_path, args.inline_images)}]
        try:
            ans = llm.chat(msg, json_mode=False).strip()
        except Exception as e:                            # noqa: BLE001
            ans = f"[调用失败] {type(e).__name__}: {e}"
        return {"image_id": s.image_id, "source_dataset": s.source_dataset,
                "anomaly": s.anomaly_types or ["normal"], "image_path": s.image_path,
                "n_objects": len(s.objects), "modality": s.modality,
                "classes": dict(Counter(o.cls for o in s.objects).most_common(5)),
                "model_sees": ans}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(run, picked))
    (out / "vlm_free_look.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n", encoding="utf-8")

    md = ["# 模型自由描述抽样图（无任何约束，只问「你看到了什么」）\n"]
    for cls in sorted({"+".join(r["anomaly"]) for r in res}):
        md.append(f"\n## {cls}\n")
        for r in [x for x in res if "+".join(x["anomaly"]) == cls]:
            md.append(f"**{r['image_id']}**　{r['source_dataset']}　"
                      f"{r['n_objects']} 个目标 {r['classes']}\n")
            md.append(f"> {r['model_sees']}\n")
    (out / "vlm_free_look.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> {out / 'vlm_free_look.md'}  (这份是重点, 先读它再谈问法)")


if __name__ == "__main__":
    main()

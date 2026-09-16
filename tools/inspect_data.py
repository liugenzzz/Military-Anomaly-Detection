"""数据体检: 先看清楚数据长什么样, 再决定该问什么。

分两部分:
  A. 纯标注统计(不用模型, 秒级)  —— 类别分布、每图目标数、框大小、共现、事件触发率
  B. 让 VLM 自由描述抽样图像(用端点池) —— **不给任何约束, 就问"你看到了什么"**

B 是关键。设计问题之前得先知道图里到底有什么: 是俯拍机场停机坪还是街景?
目标占几个像素还是占半幅画? 烟是一柱还是漫天? 不看图就设计问法, 就会写出
"清点一下画面中的烟雾"这种问法 —— 标注里 smoke 有个框, 于是想当然地去数它。

先跑 ping 确认端点通, 再跑这个:

    python tools/llm_qa.py ping
    python tools/inspect_data.py --scenes data/all_ev.jsonl --out-dir data/inspect \
        --per-class 12

--endpoints 不写就默认读 configs/generate.yaml, 与 ping 用同一份配置。
图像默认 base64 内联; 只有在推理机挂了同一块盘、且 vLLM 起服务时带了
--allowed-local-media-path 的情况下, 才用 --no-inline-images 省带宽。
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


def stat_annotations(scenes: list[Scene], onto: dict | None = None) -> dict:
    by_ds: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "cls": Counter(), "objs": [], "area": [], "size": Counter(),
        "events": Counter(), "modality": Counter(), "hard_neg": 0, "normal": 0})
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
            # **"(无事件)" 不等于"没用"。** 困难负样本(规模达标但不含军事目标的
            # 聚集, 比如 DOTA 的民用停车场)也是无事件, 但它恰恰是最该有的负样本。
            # 早先报告把两者混在一起显示, 结果 DOTA 的 3181 条看上去像是规则没
            # 跑通, 差点按"漏了"去改规则。
            d["events"]["(无事件)"] += 1
            if s.meta.get("hard_negative"):
                d["hard_neg"] += 1
            else:
                d["normal"] += 1
    # 各异常类的产量。**按 image_id 去重** —— ERA 和 ERA-SingleFrames 是同一批
    # 2173 条素材的两种形态(video / image), 按 scene 数直接加会把它们算两次,
    # 于是集结、爆炸的产量凭空多出一截, 配额算下来就是虚的。
    per_class: dict[str, set[str]] = defaultdict(set)
    for s in scenes:
        for e in s.events:
            per_class[e.type].add(s.image_id)
    counts = {k: len(v) for k, v in per_class.items()}
    zero = set()
    if onto:
        enabled = {c["id"] for c in onto.get("classes", [])
                   if c["id"] != "normal" and c.get("enabled", True)}
        zero = enabled - set(counts)
    twins = [(a, b) for a in by_ds for b in by_ds
             if a < b and by_ds[a]["n"] == by_ds[b]["n"]
             and (a.startswith(b) or b.startswith(a))]
    return {"by_dataset": by_ds, "cooccurrence": cooc,
            "per_class": counts, "zero_classes": zero, "twins": twins}


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
        if d["hard_neg"] or d["normal"]:
            out.append(f"  - 无事件的 {d['hard_neg'] + d['normal']} 条里: "
                       f"困难负样本 {d['hard_neg']}, 普通正常样本 {d['normal']}")
            if d["hard_neg"] and not d["events"].get("(无事件)", 0) - d["hard_neg"]:
                out.append("    (这一批全是困难负样本 —— 规模够但不含军事目标, "
                           "是负样本不是漏判)")
    if stats["cooccurrence"]:
        out.append(f"\n## 多类共现\n{dict(stats['cooccurrence'].most_common(10))}")
    if stats.get("per_class"):
        out.append("\n## 各异常类产量（去重后的独立素材数）")
        for cls, n in sorted(stats["per_class"].items(), key=lambda kv: -kv[1]):
            out.append(f"- {cls}: {n}")
        for a, b in stats.get("twins", []):
            out.append(f"\n**{a} 与 {b} 条数相同且同名前缀 —— 很可能是同一批素材的"
                       f"两种形态(video / 抽帧)。上面的产量已按 image_id 去重, "
                       f"但排配额时也别把它们当成两份独立数据。**")
        if stats.get("zero_classes"):
            out.append(f"\n**本体里启用但一条都没触发的类: "
                       f"{sorted(stats['zero_classes'])}** —— 要么规则太严, "
                       f"要么根本没有数据源。开跑前必须先解决, 否则产出的数据集"
                       f"少了这几类。")
    return "\n".join(out)


def apply_filters(scenes: list[Scene], only_class, only_dataset) -> list[Scene]:
    """定向复查用的过滤。

    改完判定规则之后要拿**同一批**素材再看一次模型怎么说, 全类均匀抽样做不到
    —— 想复查 DroneCrowd 的 massing, 抽出来的十二张里只有两三张是它。
    这是"改规则 -> 看模型还说不说稀疏"这个闭环的必要一环。

    **过滤要在最前面做**, 标注统计也跟着一起过滤。早先我把它放在
    `--per-class` 的提前返回之后, 于是不加那个参数时过滤根本不执行,
    "过滤到空"也不报错 —— 又一个静默降级。
    """
    out = scenes
    if only_class:
        want = set(only_class)
        out = [s for s in out
               if want & set(s.anomaly_types or ["normal"])
               or want & {e.type for e in s.events}
               or want & {f"{e.type}/{e.evidence.get('subtype')}"
                          for e in s.events if e.evidence.get("subtype")}]
    if only_dataset:
        want = set(only_dataset)
        out = [s for s in out if s.source_dataset in want]
    if (only_class or only_dataset) and not out:
        raise SystemExit(
            f"过滤后一个 scene 都不剩。\n"
            f"  --only-class {only_class or '(未指定)'}   "
            f"--only-dataset {only_dataset or '(未指定)'}\n"
            f"  可选的数据源: {sorted({s.source_dataset for s in scenes})[:12]}\n"
            f"  可选的类别:   {sorted({e.type for s in scenes for e in s.events})}")
    if out is not scenes:
        print(f"过滤后 {len(out)}/{len(scenes)} 个 scene（--only-class / --only-dataset）\n")
    return out


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
    # **默认内联**。不内联就得靠 file:// 让推理机自己去读盘, 要求 vLLM 起服务时
    # 带 --allowed-local-media-path 且挂了同一块盘 —— 条件不满足时每一张都失败。
    # base64 虽然多传点字节, 但它总是能用。先能跑通, 再谈省带宽。
    ap.add_argument("--inline-images", dest="inline_images",
                    action="store_true", default=None)
    ap.add_argument("--no-inline-images", dest="inline_images", action="store_false",
                    help="改用 file:// 让推理机自己读盘(需要 --allowed-local-media-path)")
    ap.add_argument("--only-class", nargs="*", default=None,
                    help="只看这些异常类(可写 massing 或 massing/personnel)。"
                         "改完判定规则之后定向复查用")
    ap.add_argument("--only-dataset", nargs="*", default=None,
                    help="只看这些数据源, 如 DroneCrowd VisDrone2019-MOT")
    ap.add_argument("--ontology", default="configs/ontology.yaml",
                    help="用来核对「启用了但一条都没触发」的类")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = apply_filters(load_scenes(args.scenes), args.only_class, args.only_dataset)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onto = None
    op = Path(args.ontology)
    if op.exists():
        import yaml
        onto = yaml.safe_load(op.read_text(encoding="utf-8"))
    stats = stat_annotations(scenes, onto)
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
    # 先查一遍文件在不在。模型调用是贵的, 路径检查是免费的 —— 拿几十次
    # 失败的推理去发现"图根本不在盘上", 是这个项目已经栽过的坑。
    novideo = [s for s in picked if s.still is None]
    if novideo:
        print(f"\n  {len(novideo)}/{len(picked)} 个是纯视频且没抽帧, 跳过自由看图"
              f"({novideo[0].source_dataset} 等)。视频当图发只会撞 400。")
        picked = [s for s in picked if s.still is not None]
    missing = [s for s in picked if not Path(s.still).exists()]
    if missing:
        print(f"\n⚠ {len(missing)}/{len(picked)} 张抽中的图在盘上找不到, 已剔除:")
        for s in missing[:5]:
            print(f"    {s.image_id}: {s.still}")
        if len(missing) > 5:
            print(f"    … 还有 {len(missing) - 5} 张")
        picked = [s for s in picked if Path(s.still).exists()]
    if not picked:
        raise SystemExit("抽中的图一张都不在盘上, 先确认 scenes 里的 image_path "
                         "是相对哪个目录写的。")
    print(f"\n抽 {len(picked)} 张图让模型自由描述 ({len(by_cls)} 个类)")

    llm = LLM(args.model or DEFAULT_MODEL, args.base_url,
              endpoints_file=args.endpoints, role="generate", temperature=0.2)
    print(f"  {llm.describe()}")
    workers = args.workers or llm.total_concurrency

    if args.inline_images is None:              # 命令行没指定就读配置, 再没有就内联
        import yaml
        cfg = (yaml.safe_load(Path(args.endpoints).read_text(encoding="utf-8")) or {}
               if args.endpoints else {})
        args.inline_images = bool((cfg.get("llm") or {}).get("inline_images", True))
    print(f"  图像传法: {'base64 内联' if args.inline_images else 'file:// 由推理机读盘'}")
    if args.inline_images:
        import llm_qa
        try:
            import PIL  # noqa: F401  只为探测是否装了 Pillow
        except ImportError:
            if llm_qa.MAX_IMAGE_SIDE:
                print(f"  ⚠ 没装 Pillow, max_image_side={llm_qa.MAX_IMAGE_SIDE} 不生效, "
                      f"内联的是原图。\n    3072x2048 那批会明显变慢: pip install Pillow")

    def run(s: Scene):
        msg = [{"role": "user",
                "content": image_message(FREE_PROMPT, s.still, args.inline_images)}]
        try:
            ans = llm.chat(msg, json_mode=False).strip()
        except Exception as e:                            # noqa: BLE001
            ans = f"[调用失败] {type(e).__name__}: {e}"
        return {"image_id": s.image_id, "source_dataset": s.source_dataset,
                "anomaly": s.anomaly_types or ["normal"], "image_path": s.image_path,
                "n_objects": len(s.objects), "modality": s.modality,
                "classes": dict(Counter(o.cls for o in s.objects).most_common(5)),
                "model_sees": ans}

    # **先拿一张真图试一次再开工。** 上一版是直接 60 张并发跑, 每条失败都被
    # 吞成 "[调用失败]" 写进结果, 于是跑完才发现整份报告 100% 是错误信息 ——
    # 既浪费了一轮, 又让一次全废的运行看起来像"有结果了"。
    probe = run(picked[0])
    # 服务端读不了本地路径时自动切回内联。配置里把 inline_images 写成 false 很容易,
    # 而这个错只在真正发图的时候才暴露 —— 与其让人去翻配置, 不如就地改对再往下跑。
    if (not args.inline_images
            and "allowed-local-media-path" in probe["model_sees"]):
        print("\n  推理机读不了本地路径(vLLM 没带 --allowed-local-media-path), "
              "自动改用 base64 内联重试。")
        print("  想省这份带宽的话, 起服务时加上该参数并确认推理机挂了同一块盘;"
              "\n  否则把 configs/generate.yaml 的 llm.inline_images 设成 true。")
        args.inline_images = True
        probe = run(picked[0])
    if probe["model_sees"].startswith("[调用失败]"):
        print(f"\n预检失败, 已中止 —— 不再把 {len(picked)} 条错误信息写成报告。")
        print(f"  图: {picked[0].still}")
        print(f"  {probe['model_sees']}")
        print("\n先跑 `python tools/llm_qa.py ping` 确认端点; 如果 ping 全绿而这里"
              "仍失败,\n  多半是图像传法不对: 默认 base64 内联, 若你显式加了"
              " --no-inline-images,\n  推理机必须挂到同一块盘并带"
              " --allowed-local-media-path。")
        raise SystemExit(1)

    with ThreadPoolExecutor(workers) as ex:
        res = [probe] + list(ex.map(run, picked[1:]))
    bad = [r for r in res if r["model_sees"].startswith("[调用失败]")]
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
    if bad:
        # 部分失败也要显眼地说出来。抽样报告里混着几条错误信息, 读的人很容易
        # 当成"模型看不清这张图"的结论, 实际上是根本没调通。
        print(f"\n⚠ {len(bad)}/{len(res)} 条调用失败, 这几条在报告里是错误信息不是描述:")
        for r in bad[:3]:
            print(f"    {r['image_id']}: {r['model_sees'][:110]}")
        if len(bad) > 3:
            print(f"    … 还有 {len(bad) - 3} 条")


if __name__ == "__main__":
    main()

"""把各适配器产出的 interim/*.jsonl 合并成一个 all.jsonl，供 derive_events 用。

这一步以前是手敲 `cat data/interim/*.jsonl > data/all.jsonl`，已经坑过两次:

  1. 通配符把不是 scene 的文件也卷了进来 —— llm_requests.jsonl、gen_req.jsonl
     这些同样有 image_id 和 image_path，光看字段名分不出来；
  2. 某个文件末尾缺一个换行，cat 直接把两条记录粘成一行，那一条静默丢失。

所以改成: **按内容判定是不是 scene 文件**(要有 image_id/image_path/width/height),
逐行解析并报出坏行的 文件:行号, 按 image_id 去重并报告冲突, 最后按数据源打表。

    python tools/merge_scenes.py                       # 默认合 data/interim -> data/all.jsonl
    python tools/merge_scenes.py --in data/interim --out data/all.jsonl
    python tools/merge_scenes.py --in a.jsonl b.jsonl --out data/all.jsonl
    python tools/merge_scenes.py --dry-run             # 只看会合哪些, 不写文件
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

NEEDED = {"image_id", "image_path", "width", "height"}
# 一眼就该跳过的: derive 的产物、LLM 请求体、演示与测试数据
SKIP_SUFFIX = ("_ev.jsonl",)
SKIP_STEMS = {"llm_requests", "gen_req", "rev_req", "requests", "responses"}
DEMO_HINTS = ("demo", "test", "sample", "ontology_test")


def looks_like_scenes(path: Path) -> tuple[bool, str]:
    """按内容判定。**不按文件名** —— 名字是最不可靠的依据。"""
    try:
        with path.open(encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                miss = NEEDED - set(d)
                return (not miss, "" if not miss else f"缺字段 {sorted(miss)}")
        return False, "空文件"
    except json.JSONDecodeError as e:
        return False, f"首行不是 JSON: {e}"
    except OSError as e:
        return False, str(e)


def pick_files(inputs: list[str], keep_ev: bool, keep_demo: bool) -> tuple[list[Path], list[tuple[Path, str]]]:
    cands: list[Path] = []
    for x in inputs:
        p = Path(x)
        cands += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]

    take, skip = [], []
    for p in cands:
        if not p.exists():
            skip.append((p, "文件不存在"))
        elif not keep_ev and p.name.endswith(SKIP_SUFFIX):
            skip.append((p, "derive_events 的产物(加 --include-ev 可保留)"))
        elif p.stem in SKIP_STEMS:
            skip.append((p, "不是 scene 文件(LLM 请求体之类)"))
        elif not keep_demo and any(h in p.stem.lower() for h in DEMO_HINTS):
            skip.append((p, "演示/测试数据(加 --include-demo 可保留)"))
        else:
            ok, why = looks_like_scenes(p)
            (take if ok else skip).append(p if ok else (p, why))
    return take, skip


def main() -> None:
    ap = argparse.ArgumentParser(description="合并 scene jsonl -> all.jsonl")
    ap.add_argument("--in", dest="inputs", nargs="+", default=["data/interim"],
                    help="目录或文件列表, 默认 data/interim")
    ap.add_argument("--out", default="data/all.jsonl")
    ap.add_argument("--include-ev", action="store_true",
                    help="连 *_ev.jsonl 一起合(默认跳过 —— 那是 derive 的产物, "
                         "合进来会和源文件重复)")
    ap.add_argument("--include-demo", action="store_true", help="连 demo/test 数据一起合")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="要排除的文件名(可写 stem, 如 d 或 d.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="只报告会合哪些, 不写文件")
    args = ap.parse_args()

    take, skip = pick_files(args.inputs, args.include_ev, args.include_demo)
    if args.exclude:
        drop = {x.removesuffix(".jsonl") for x in args.exclude}
        skip += [(p, "--exclude 排除") for p in take if p.stem in drop]
        take = [p for p in take if p.stem not in drop]
    if skip:
        print("跳过:")
        for p, why in skip:
            print(f"  {p}  —— {why}")
    if not take:
        raise SystemExit("\n没有可合并的 scene 文件。先跑 tools/prepare.py 产出 "
                         "data/interim/*.jsonl。")
    print(f"\n合并 {len(take)} 个文件:")

    seen: dict[str, Path] = {}
    rows: list[str] = []
    by_ds: Counter = Counter()
    dup: dict[str, list[str]] = defaultdict(list)
    bad: list[str] = []

    for p in take:
        n_ok = n_dup = 0
        with p.open(encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                if not ln.strip():
                    continue
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError as e:
                    # 逐行解析才能定位。cat 粘行的那种错, 整份文件读进来是看不出的
                    bad.append(f"{p}:{i}  {e}")
                    continue
                if NEEDED - set(d):
                    bad.append(f"{p}:{i}  缺字段 {sorted(NEEDED - set(d))}")
                    continue
                iid = d["image_id"]
                if iid in seen:
                    dup[iid].append(f"{p.name}")
                    n_dup += 1
                    continue
                seen[iid] = p
                by_ds[d.get("source_dataset", "unknown")] += 1
                rows.append(json.dumps(d, ensure_ascii=False))
                n_ok += 1
        print(f"  {p.name:32s} {n_ok:7d} 条" + (f"  (重复丢弃 {n_dup})" if n_dup else ""))

    if bad:
        print(f"\n⚠ {len(bad)} 行有问题, 已跳过:")
        for b in bad[:10]:
            print(f"    {b}")
        if len(bad) > 10:
            print(f"    … 还有 {len(bad) - 10} 行")
    if dup:
        print(f"\n⚠ {len(dup)} 个 image_id 重复(只留了第一次出现的):")
        for iid, where in list(dup.items())[:5]:
            print(f"    {iid}  又出现在 {', '.join(where[:3])}")

    print(f"\n共 {len(rows)} 个 scene / {len(by_ds)} 个数据源")
    for ds, n in by_ds.most_common():
        print(f"  {ds:28s} {n:7d}")
    # 文件名看不出来、但 source_dataset 露馅的演示数据。d.jsonl 就是这种 ——
    # 名字毫无特征, 内容却是 DEMO-synthetic, 混进真实语料会污染整批训练数据。
    fake = [ds for ds in by_ds if any(h in ds.lower() for h in ("demo", "synthetic", "test"))]
    if fake and not args.include_demo:
        print(f"\n⚠ 这些数据源看着是演示/合成数据: {fake}")
        print("  文件名看不出来, 但 source_dataset 露馅了。确认要把它们训进去吗?")
        print("  不要的话从 --in 里去掉对应文件, 或用 --exclude 排除。")

    if args.dry_run:
        print("\n(--dry-run, 没有写文件)")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 每行都带换行, 最后一行也是 —— 这正是 cat 粘行事故的根源
    out.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    print(f"\n-> {out}")
    print(f"下一步: python tools/derive_events.py --scenes {out} "
          f"--out {out.with_name(out.stem + '_ev.jsonl')} --overwrite")


if __name__ == "__main__":
    main()

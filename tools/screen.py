"""质量筛选: 五道闸中的前四道(纯规则), 以及第五道 VLM 复核结果的合并。

用法:
  python tools/screen.py --scenes data/interim/all.jsonl --out-dir data/screened
  python tools/screen.py ... --vlm-review data/screened/vlm_review.jsonl   # 合并闸5结果

输出 scenes_kept.jsonl / scenes_dropped.jsonl(带 drop_reason) / screen_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from scene import Scene, dump_scenes, load_scenes

try:
    from PIL import Image, ImageFilter, ImageStat
    HAS_PIL = True
except ImportError:                                    # 无 PIL 时跳过像素级检查
    HAS_PIL = False

# 按数据源校准的阈值。卫星图天然锐利, 夜间监控天然偏糊, 一刀切会误杀。
PROFILES: dict[str, dict[str, float]] = {
    "default":   {"min_side": 224, "blur": 60,  "std": 8},
    "satellite": {"min_side": 512, "blur": 100, "std": 8},
    "cctv":      {"min_side": 224, "blur": 30,  "std": 6},
    "uav":       {"min_side": 224, "blur": 60,  "std": 8},
}
ASPECT_RANGE = (0.2, 5.0)
BRIGHTNESS_RANGE = (15, 240)
MIN_BOX_SIDE = 4.0
MAX_OBJECTS = 2000
DHASH_THRESHOLD = 5


# ------------------------------------------------------------ 闸1 图像质量
def image_stats(path: Path) -> dict[str, Any] | None:
    """返回 {w,h,blur,brightness,std,dhash}; 文件不可读时返回 None。"""
    if not HAS_PIL:
        return {}
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            g = im.convert("L")
            small = g.resize((256, 256))
            lap = small.filter(ImageFilter.Kernel(
                (3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
            st_l, st_g = ImageStat.Stat(lap), ImageStat.Stat(g)
            # dHash: 相邻像素比较, 64 bit
            d = g.resize((9, 8)).tobytes()          # mode L, 行主序 9x8
            bits = 0
            for r in range(8):
                for c in range(8):
                    bits = (bits << 1) | (1 if d[r * 9 + c] > d[r * 9 + c + 1] else 0)
            return {"w": w, "h": h,
                    "blur": st_l.var[0], "brightness": st_g.mean[0],
                    "std": st_g.stddev[0], "dhash": bits}
    except Exception:
        return None


def gate1(scene: Scene, stats: dict[str, Any] | None, prof: dict[str, float]) -> str | None:
    if stats is None:
        return "unreadable_image"
    if not stats:
        return None                                     # 无 PIL, 跳过
    w, h = stats["w"], stats["h"]
    if min(w, h) < prof["min_side"]:
        return f"too_small({min(w, h)}px)"
    ar = w / max(1, h)
    if not (ASPECT_RANGE[0] <= ar <= ASPECT_RANGE[1]):
        return f"bad_aspect({ar:.2f})"
    if not (BRIGHTNESS_RANGE[0] <= stats["brightness"] <= BRIGHTNESS_RANGE[1]):
        return f"bad_exposure({stats['brightness']:.0f})"
    if stats["std"] < prof["std"]:
        return f"flat_image(std={stats['std']:.1f})"    # 须先于模糊判定, 否则纯色图会被误报为 blurry
    if stats["blur"] < prof["blur"]:
        return f"blurry({stats['blur']:.0f})"
    return None


# ------------------------------------------------------------ 闸2 标注合法性
def gate2(scene: Scene, vocab: set[str]) -> tuple[str | None, list[str]]:
    """就地修正可修的问题, 返回 (整图丢弃原因 or None, 警告列表)。"""
    warns: list[str] = []
    kept = []
    for o in scene.objects:
        x1, y1, x2, y2 = o.bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(scene.width), x2), min(float(scene.height), y2)
        if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
            warns.append("tiny_box_dropped")
            continue                                    # 丢框不丢图
        o.bbox = [x1, y1, x2, y2]
        if vocab and o.cls.lower() not in vocab:
            warns.append(f"unmapped_class:{o.cls}")
        kept.append(o)
    scene.objects = kept

    if len(scene.objects) > MAX_OBJECTS:
        warns.append("extremely_dense")                 # 标记而非丢弃, 走计数任务

    # 事件与标注自洽: 声称集结却数不出那么多目标, 说明派生规则出了问题
    for e in scene.events:
        if e.evidence.get("rule") == "density_cluster":
            n = e.evidence.get("count", 0)
            if n > len(scene.objects):
                return "event_annotation_mismatch", warns
    return None, warns


# ------------------------------------------------------------ 闸3 近重复
def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def gate3_dedup(items: list[tuple[Scene, dict[str, Any] | None]],
                threshold: int = DHASH_THRESHOLD) -> set[str]:
    """跨数据集统一去重, 返回应丢弃的 image_id 集合。同簇保留最清晰的一张。

    先按 dhash 高 16 位分桶再桶内两两比, 避免 O(n^2) 全量比较。
    """
    buckets: dict[int, list[tuple[Scene, dict]]] = defaultdict(list)
    for s, st in items:
        if st and "dhash" in st:
            buckets[st["dhash"] >> 48].append((s, st))

    drop: set[str] = set()
    for group in buckets.values():
        used = set()
        for i in range(len(group)):
            if i in used:
                continue
            cluster = [i]
            for j in range(i + 1, len(group)):
                if j not in used and hamming(group[i][1]["dhash"], group[j][1]["dhash"]) <= threshold:
                    cluster.append(j)
                    used.add(j)
            if len(cluster) > 1:
                best = max(cluster, key=lambda k: group[k][1].get("blur", 0))
                for k in cluster:
                    if k != best:
                        drop.add(group[k][0].image_id)
    return drop


# ------------------------------------------------------------ 闸4/5
VALID_VIEWS = {"uav", "satellite", "cctv"}


def gate4(scene: Scene) -> str | None:
    if scene.view not in VALID_VIEWS:
        return f"bad_view({scene.view})"
    return None


def gate5(scene: Scene, review: dict[str, Any] | None, min_score: int) -> str | None:
    """合并 VLM 复核结果。'不确定'不丢, 进 uncertain 池等人工抽检。"""
    if review is None:
        return None
    if review.get("view") and review["view"] not in VALID_VIEWS:
        return f"vlm_bad_view({review['view']})"
    if review.get("watermark"):
        return "vlm_watermark"
    if review.get("consistent") == "no":
        return "vlm_label_inconsistent"
    if review.get("consistent") == "unsure":
        scene.meta["uncertain"] = True                  # 标记, 不丢
    score = review.get("score")
    if score is not None and score < min_score:
        return f"vlm_low_score({score})"
    return None


# ------------------------------------------------------------ 驱动
def main() -> None:
    ap = argparse.ArgumentParser(description="Scene 质量筛选(闸1-5)")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out-dir", default="data/screened")
    ap.add_argument("--ontology", default="configs/ontology.yaml")
    ap.add_argument("--vlm-review", default=None, help="闸5 结果 jsonl, 每行 {image_id, view, consistent, score, watermark}")
    ap.add_argument("--min-vlm-score", type=int, default=3)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--dedup-threshold", type=int, default=DHASH_THRESHOLD)
    args = ap.parse_args()

    onto = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    vocab = {c.lower() for cls in onto["classes"] for c in
             (cls.get("rule", {}) or {}).get("target_classes", [])}

    reviews: dict[str, dict] = {}
    if args.vlm_review:
        for ln in Path(args.vlm_review).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                reviews[r["image_id"]] = r

    scenes = load_scenes(args.scenes)
    if not HAS_PIL:
        print("[warn] 未安装 Pillow, 闸1/闸3 的像素级检查将被跳过。pip install Pillow")

    kept, dropped = [], []
    reasons, warn_counter = Counter(), Counter()
    by_ds_total, by_ds_kept = Counter(), Counter()
    staged: list[tuple[Scene, dict | None]] = []

    for s in scenes:
        by_ds_total[s.source_dataset] += 1
        prof = PROFILES.get(s.view, PROFILES["default"])
        st = image_stats(Path(s.image_path))

        reason = gate1(s, st, prof)
        if not reason:
            reason, warns = gate2(s, vocab)
            warn_counter.update(warns)
        if not reason:
            reason = gate4(s)
        if not reason:
            reason = gate5(s, reviews.get(s.image_id), args.min_vlm_score)

        if reason:
            s.meta["drop_reason"] = reason
            reasons[reason.split("(")[0]] += 1
            dropped.append(s)
        else:
            staged.append((s, st))

    if not args.no_dedup:
        dup = gate3_dedup(staged, args.dedup_threshold)
        for s, _ in staged:
            if s.image_id in dup:
                s.meta["drop_reason"] = "near_duplicate"
                reasons["near_duplicate"] += 1
                dropped.append(s)
            else:
                kept.append(s)
    else:
        kept = [s for s, _ in staged]

    for s in kept:
        by_ds_kept[s.source_dataset] += 1

    out = Path(args.out_dir)
    dump_scenes(kept, out / "scenes_kept.jsonl")
    dump_scenes(dropped, out / "scenes_dropped.jsonl")

    n = len(scenes)
    report = {
        "total": n, "kept": len(kept), "dropped": len(dropped),
        "keep_rate": round(len(kept) / max(1, n), 4),
        "drop_reasons": dict(reasons.most_common()),
        "warnings": dict(warn_counter.most_common()),
        "uncertain_pool": sum(1 for s in kept if s.meta.get("uncertain")),
        "by_dataset": {d: {"total": by_ds_total[d], "kept": by_ds_kept[d],
                           "keep_rate": round(by_ds_kept[d] / max(1, by_ds_total[d]), 4)}
                       for d in by_ds_total},
    }
    (out / "screen_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"总计 {n} -> 保留 {len(kept)} ({report['keep_rate']:.1%})")
    for k, v in reasons.most_common():
        print(f"  剔除 {k:28s} {v}")
    if report["uncertain_pool"]:
        print(f"  待人工抽检(VLM 判为不确定) {report['uncertain_pool']}")
    print("\n按数据集保留率:")
    for d, v in report["by_dataset"].items():
        flag = ""
        if v["keep_rate"] > 0.90:
            flag = "  <- 偏高, 阈值可能过松"
        elif v["keep_rate"] < 0.50:
            flag = "  <- 偏低, 检查是否误杀"
        print(f"  {d:28s} {v['kept']:6d}/{v['total']:<6d} {v['keep_rate']:.1%}{flag}")


if __name__ == "__main__":
    main()

"""数据体检：扫一遍数据根目录，报告每个数据集是否可用、卡在哪。

下载工具的失败信息往往指向它自己的取数逻辑（比如去 HF 找镜像失败），
未必说明数据不可用。这个脚本只看**磁盘上实际有什么**，并直接拿本项目的
适配器试跑一小批，结论比下载报告可靠。

    python tools/doctor.py --root /mnt/.../military
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMG = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
VID = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".mpg", ".m4v")

# 数据集名 -> (目录名候选, 供哪个异常类, 是不是单点依赖)
SPEC = {
    "MAR20": (["MAR20"], "massing/equipment", False),
    "Mendeley-UAV-Military": (["Mendeley-UAV-Military", "mendeley"], "massing/equipment", False),
    "FASDD_UAV": (["FASDD_UAV", "FASDD"], "smoke + explosion", True),
    "ERA": (["ERA"], "massing/personnel + explosion + normal", False),
    "CapERA": (["CapERA", "CapEra"], "描述语料", False),
    "DroneCrowd": (["DroneCrowd"], "massing/personnel", False),
    "VisDrone-MOT": (["VisDrone2019-MOT-train", "VisDrone-MOT", "VisDrone"], "border_crossing", True),
    "VisDrone-DET": (["VisDrone2019-DET-train", "VisDrone-DET"], "normal", False),
    "DOTA": (["DOTA", "DOTA-v2.0"], "困难负样本", False),
    "Drone-Anomaly": (["Drone-Anomaly"], "normal", False),
}


def _find(root: Path, names: list[str]) -> Path | None:
    for n in names:
        p = root / n
        if p.is_dir():
            return p
    for n in names:
        hits = [d for d in root.rglob(n) if d.is_dir()]
        if hits:
            return hits[0]
    return None


def _count(d: Path, exts: tuple[str, ...], cap: int = 200000) -> int:
    n = 0
    for p in d.rglob("*"):
        if p.suffix.lower() in exts:
            n += 1
            if n >= cap:
                break
    return n


def check_visdrone_mot(d: Path) -> list[str]:
    """VisDrone MOT 标注体检。下载器报的 '缺少有效 track_id' 多半是它自己的
    校验口径问题 —— MOT 的 target_id 在第 2 列，这里直接按列解析确认。"""
    msgs = []
    anns = sorted(d.rglob("annotations/*.txt")) or sorted(d.rglob("*.txt"))
    anns = [a for a in anns if a.stat().st_size > 0][:5]
    if not anns:
        return ["未找到 MOT 标注 txt"]
    for a in anns[:3]:
        rows, tids, cols = 0, set(), Counter()
        for ln in a.read_text(encoding="utf-8", errors="ignore").splitlines()[:5000]:
            p = ln.strip().rstrip(",").split(",")
            cols[len(p)] += 1
            if len(p) >= 8:
                try:
                    tids.add(int(float(p[1])))
                    rows += 1
                except ValueError:
                    pass
        ok = rows > 0 and len(tids) > 1
        msgs.append(f"{a.name}: {rows} 行可解析, {len(tids)} 个 track_id, "
                    f"列数分布 {dict(cols.most_common(3))} -> {'可用 ✓' if ok else '异常 ✗'}")
    return msgs


def check_era(d: Path) -> list[str]:
    """直接复用适配器的类别识别逻辑, 免得体检与实际处理两套口径对不上。"""
    from ds.era import ANOMALY, EXCLUDE, NORMAL, _class_dirs_all, _norm_label
    from ds.common import VID_EXT

    cls_dirs = _class_dirs_all(d)
    if not cls_dirs:
        return [f"未找到含视频的类别目录（子目录: {[x.name for x in d.iterdir() if x.is_dir()][:6]}）"]

    known = set(ANOMALY) | NORMAL | EXCLUDE
    tally: dict[str, int] = {}
    for cd in cls_dirs:
        n = sum(1 for x in cd.iterdir() if x.is_file() and x.suffix.lower() in VID_EXT)
        tally[_norm_label(cd.name)] = tally.get(_norm_label(cd.name), 0) + n

    a = sum(v for k, v in tally.items() if k in ANOMALY)
    n = sum(v for k, v in tally.items() if k in NORMAL)
    e = sum(v for k, v in tally.items() if k in EXCLUDE)
    u = {k: v for k, v in tally.items() if k not in known}
    out = [f"{len(tally)} 个类别, 共 {sum(tally.values())} 段视频",
           f"映射为异常 {a} 段 / 安全负样本 {n} 段 / 歧义已排除 {e} 段"]
    if u:
        out.append(f"未归档类别 {sum(u.values())} 段: {list(u)[:6]} —— 需在 ds/era.py 里补映射")
    sf = d / "SingleFrames"
    if sf.is_dir():
        out.append(f"另有官方 SingleFrames/ 单帧数据 {_count(sf, IMG)} 张, "
                   f"用 --single-frames 一并采集")
    return out


def check_coco(d: Path) -> list[str]:
    from ds.common import looks_like_coco
    js = [p for p in d.rglob("*.json")
          if not p.name.startswith(".") and looks_like_coco(p)]
    if not js:
        return ["未找到 COCO 标注 json"]
    out = []
    for p in js[:3]:
        try:
            dd = json.loads(p.read_text(encoding="utf-8"))
            cats = [c["name"] for c in dd.get("categories", [])]
            out.append(f"{p.name}: {len(dd.get('images', []))} 图 / "
                       f"{len(dd.get('annotations', []))} 标注 / 类别 {cats}")
        except Exception as e:
            out.append(f"{p.name}: 解析失败 {e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="数据体检: 看磁盘上实际有什么")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    ready, blocked = [], []
    print(f"扫描 {root}\n" + "═" * 78)
    for name, (dirs, feeds, critical) in SPEC.items():
        d = _find(root, dirs)
        mark = "★" if critical else " "
        if d is None:
            print(f"{mark} {name:24s} ✗ 目录不存在")
            blocked.append((name, feeds, critical, "目录不存在"))
            continue

        n_img, n_vid = _count(d, IMG), _count(d, VID)
        n_ann = _count(d, (".txt", ".xml", ".json", ".mat"))
        only_archives = {x.name for x in d.iterdir() if x.is_dir()} <= {"archives"}
        detail = f"{n_img} 图 / {n_vid} 视频 / {n_ann} 标注文件"

        if only_archives:
            print(f"{mark} {name:24s} ⚠ 只有 archives/，未解压   ({detail})")
            blocked.append((name, feeds, critical, "未解压"))
            continue
        if n_img == 0 and n_vid == 0:
            print(f"{mark} {name:24s} ✗ 没有图像也没有视频   ({detail})")
            blocked.append((name, feeds, critical, "无图像"))
            continue

        print(f"{mark} {name:24s} ✓ {detail}")
        extra: list[str] = []
        if name == "ERA":
            extra = check_era(d)
        elif name == "VisDrone-MOT":
            extra = check_visdrone_mot(d)
        elif name in ("FASDD_UAV",):
            extra = check_coco(d)
        for line in extra:
            print(f"{'':27s}{line}")
        ready.append((name, feeds))

    print("═" * 78)
    print(f"\n可用 {len(ready)} 个:")
    for n, f in ready:
        print(f"  {n:24s} -> {f}")
    if blocked:
        print(f"\n待处理 {len(blocked)} 个:")
        for n, f, c, why in blocked:
            print(f"  {'[单点依赖] ' if c else ''}{n:24s} {why:10s} -> 影响 {f}")

    have = {n for n, _ in ready}
    print("\n现在能产出的异常类:")
    for cls, need, alt in [
        ("massing/personnel", {"ERA"}, {"DroneCrowd"}),
        ("massing/equipment", {"Mendeley-UAV-Military"}, {"MAR20"}),
        ("explosion", {"ERA"}, {"FASDD_UAV"}),
        ("smoke", {"FASDD_UAV"}, set()),
        ("border_crossing", {"VisDrone-MOT"}, set()),
        ("normal", {"ERA"}, {"VisDrone-DET", "DOTA", "Drone-Anomaly"}),
    ]:
        got = need & have
        plus = alt & have
        state = "✓ 可产出" if got else "✗ 缺主力源"
        src = "、".join(sorted(got | plus)) or "无"
        print(f"  {cls:20s} {state}   来源: {src}")


if __name__ == "__main__":
    main()

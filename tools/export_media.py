"""把数据集引用到的图片/视频归集到语料库目录, 并改写 json 里的路径。

产出的目录结构(与 corpus_media 下 book/journal/video 的分法一致):

    <media-root>/<name>/images/<数据源>/<image_id>.jpg
    <media-root>/<name>/videos/<数据源>/<video_id>.mp4

图片与视频分在两个二级目录下, 再按数据源分目录 —— 不同数据集重名的文件很多
(DroneCrowd 和 VisDrone 都有 img0001.jpg), 不按源分会互相覆盖。

    python tools/export_media.py --vqa-dir data/vqa_rule data/vqa_llm \
        --media-root /mnt/.../corpus_media --name military_anomaly

默认建硬链接: 同一块盘上不占额外空间, 删原文件也不影响。跨盘时自动退回复制。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

MEDIA_KEYS = ("images", "videos")
EXT_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _sub(path: Path, field: str) -> str:
    """落到 images/ 还是 videos/。以 json 字段为准, 扩展名只用来兜底。"""
    if field == "videos":
        return "videos"
    if field == "images":
        return "images"
    return "images" if path.suffix.lower() in EXT_IMAGE else "videos"


def _link(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst) if mode == "hardlink" else os.symlink(src, dst)
        return mode
    except OSError:
        shutil.copy2(src, dst)               # 跨设备硬链接会失败, 退回复制
        return "copy"


def main() -> None:
    ap = argparse.ArgumentParser(description="归集媒体文件到语料库目录并改写路径")
    ap.add_argument("--vqa-dir", nargs="+", required=True, help="含 train/val/test.json 的目录")
    ap.add_argument("--media-root", required=True,
                    help="语料库根目录, 例如 /mnt/.../data_process/corpus_media")
    ap.add_argument("--name", default="military_anomaly", help="本数据集在语料库下的目录名")
    ap.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    ap.add_argument("--path-style", choices=["abs", "rel"], default="abs",
                    help="改写后写绝对路径, 还是相对 media-root 的路径")
    ap.add_argument("--out-dir", default=None,
                    help="改写后的 json 落在哪; 不填就原地覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不动文件")
    args = ap.parse_args()

    root = Path(args.media_root) / args.name
    stat = {"copy": 0, "hardlink": 0, "symlink": 0, "skip": 0, "missing": 0}
    mapping: dict[str, str] = {}             # 原路径 -> 新路径, 同一文件只搬一次

    for d in args.vqa_dir:
        d = Path(d)
        for jf in sorted(d.glob("*.json")):
            if jf.name in ("dataset_info.json", "report.json"):
                continue
            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for row in data:
                src_ds = (row.get("extra") or {}).get("source_dataset", "misc")
                for field in MEDIA_KEYS:
                    if field not in row:
                        continue
                    new_paths = []
                    for old in row[field]:
                        if old in mapping:
                            new_paths.append(mapping[old])
                            continue
                        sp = Path(old)
                        dst = root / _sub(sp, field) / src_ds / sp.name
                        if not sp.exists():
                            stat["missing"] += 1
                            new_paths.append(old)     # 文件不在就保留原路径, 不伪造
                            continue
                        if not args.dry_run:
                            stat[_link(sp, dst, args.mode)] += 1
                        else:
                            stat["skip"] += 1
                        out = (str(dst) if args.path_style == "abs"
                               else str(dst.relative_to(Path(args.media_root))))
                        mapping[old] = out
                        new_paths.append(out)
                    row[field] = new_paths
            if not args.dry_run:
                outp = Path(args.out_dir) / jf.name if args.out_dir else jf
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            print(f"  {jf}  {len(data)} 条")

    print(f"\n媒体根目录: {root}")
    print(f"  images/  videos/  两个二级目录, 其下按数据源分目录")
    print(f"  归集 {len(mapping)} 个文件: " +
          ", ".join(f"{k}={v}" for k, v in stat.items() if v))
    if stat["missing"]:
        print(f"  [注意] {stat['missing']} 个引用的文件不存在, 路径按原样保留未改写")


if __name__ == "__main__":
    main()

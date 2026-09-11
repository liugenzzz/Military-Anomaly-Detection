"""数据集预处理统一入口: 各公开数据集 -> 统一的 Scene jsonl。

  python tools/prepare.py mar20        --root data/raw/MAR20 --out data/interim/mar20.jsonl
  python tools/prepare.py mendeley     --root data/raw/mendeley --out data/interim/mendeley.jsonl
  python tools/prepare.py fasdd        --root data/raw/FASDD_UAV --out data/interim/fasdd.jsonl
  python tools/prepare.py era          --root data/raw/ERA --frames-dir data/frames/era \
                                       --capera data/raw/CapEra/captions.json --out data/interim/era.jsonl
  python tools/prepare.py dronecrowd   --root data/raw/DroneCrowd --out data/interim/dronecrowd.jsonl
  python tools/prepare.py visdrone-det --root data/raw/VisDrone-DET-train --out data/interim/vd_det.jsonl
  python tools/prepare.py visdrone-mot --root data/raw/VisDrone-MOT-train --out data/interim/vd_mot.jsonl
  python tools/prepare.py dota         --root data/raw/DOTA --tiles-dir data/tiles/dota --out data/interim/dota.jsonl
  python tools/prepare.py drone-anomaly --root data/raw/Drone-Anomaly --out data/interim/da.jsonl
  python tools/prepare.py hivau        --video-root data/raw/UCF-Crime --frames-dir data/frames/ucf \
                                       --temporal-ann .../Temporal_Anomaly_Annotation.txt --out data/interim/hivau.jsonl

处理完把所有 jsonl 合并, 再走 derive_events -> screen -> make_golden -> build_vqa / llm_qa。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ds import dota, drone_anomaly, dronecrowd, era, fasdd, hivau, mar20, mendeley_mil, visdrone  # noqa: E402
from ds.boxes import audit_scene_boxes  # noqa: E402
from scene import dump_scenes  # noqa: E402


def _report(scenes, out: str) -> None:
    dump_scenes(scenes, out)
    by_src = Counter(s.source_dataset for s in scenes)
    ev = Counter(e.type + (f"/{e.evidence['subtype']}" if "subtype" in e.evidence else "")
                 for s in scenes for e in s.events)
    n_obj = sum(len(s.objects) for s in scenes)
    n_cap = sum(1 for s in scenes if s.caption)
    mod = Counter(s.modality for s in scenes)
    n_hn = sum(1 for s in scenes if s.meta.get("hard_negative_source"))
    print(f"\n写出 {len(scenes)} 个 scene -> {out}")
    print(f"  目标框合计 {n_obj}" + (f", 带 caption {n_cap}" if n_cap else "")
          + (f", 同场景困难负样本 {n_hn}" if n_hn else ""))
    if len(mod) > 1 or "image" not in mod:
        print("  输入形态:", dict(mod))
    if ev:
        print("  自带事件标签:", dict(ev))
    if len(by_src) > 1:
        print("  来源:", dict(by_src))
    for w in audit_scene_boxes(scenes):
        print(f"  [坐标自检] {w}")
    print("  下一步: python tools/derive_events.py --scenes "
          f"{out} --out {out.replace('.jsonl', '_ev.jsonl')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="公开数据集预处理 -> Scene jsonl",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_out(p):
        p.add_argument("--out", required=True)

    p = sub.add_parser("mar20", help="MAR20 军机遥感(VOC XML 或 YOLO)")
    p.add_argument("--root", required=True)
    p.add_argument("--img-dir"); p.add_argument("--ann-dir")
    p.add_argument("--format", dest="fmt", default="voc", choices=["voc", "yolo"])
    p.add_argument("--classes"); p.add_argument("--view", default="satellite")
    p.add_argument("--keep-augmented", action="store_true",
                   help="保留 Roboflow 的增强副本(默认折叠, 同一源图只留一份)")
    add_out(p)

    p = sub.add_parser("mendeley", help="Mendeley UAV 军事目标(自动探测格式)")
    p.add_argument("--root", required=True)
    p.add_argument("--format", dest="fmt", default=None, choices=["coco", "voc", "yolo"])
    p.add_argument("--classes"); p.add_argument("--view", default="uav")
    p.add_argument("--keep-classes", nargs="*", default=None,
                   help="保留哪些类别, 默认 tank soldier(drone 与任务无关, people 让位给 DroneCrowd)")
    add_out(p)

    p = sub.add_parser("fasdd", help="FASDD_UAV 火焰烟雾(COCO)")
    p.add_argument("--root", required=True)
    p.add_argument("--drop-negatives", action="store_true", help="丢弃无火无烟的图(默认保留作负样本)")
    p.add_argument("--view", default="uav")
    add_out(p)

    p = sub.add_parser("era", help="ERA 航拍事件视频 + CapERA caption(需抽帧)")
    p.add_argument("--root", required=True)
    p.add_argument("--modality", default="video", choices=["video", "frame", "both"],
                   help="video 保留整段视频(默认) / frame 抽帧 / both 两种都产")
    p.add_argument("--frames-dir", default=None, help="modality 含 frame 时必填")
    p.add_argument("--n-frames", type=int, default=3)
    p.add_argument("--capera", default=None, help="CapERA caption json")
    p.add_argument("--no-normal", action="store_true", help="不采集正常类别")
    p.add_argument("--single-frames", action="store_true",
                   help="同时采集官方 SingleFrames/ 单帧分类数据(与视频样本互补)")
    p.add_argument("--view", default="uav")
    add_out(p)

    p = sub.add_parser("dronecrowd", help="DroneCrowd 密集人群(点标注)")
    p.add_argument("--root", required=True)
    p.add_argument("--ann-dir"); p.add_argument("--stride", type=int, default=8)
    p.add_argument("--head-half", type=float, default=8.0)
    p.add_argument("--view", default="uav")
    add_out(p)

    p = sub.add_parser("visdrone-det", help="VisDrone DET 检测")
    p.add_argument("--root", required=True); p.add_argument("--view", default="uav")
    add_out(p)

    p = sub.add_parser("visdrone-mot", help="VisDrone MOT 跟踪(越界派生, 支持自动放置边界)")
    p.add_argument("--root", required=True)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--boundaries", default=None, help='人工边界 json: {"序列名": [[x,y],...]}')
    p.add_argument("--no-auto-boundary", action="store_true")
    p.add_argument("--view", default="uav")
    add_out(p)

    p = sub.add_parser("dota", help="DOTA v2.0 遥感大图(切片, 主要产出困难负样本)")
    p.add_argument("--root", required=True)
    p.add_argument("--tiles-dir", required=True)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--min-objects", type=int, default=5)
    p.add_argument("--empty-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", default="satellite")
    add_out(p)

    p = sub.add_parser("drone-anomaly", help="Drone-Anomaly(只取正常帧, 见模块说明)")
    p.add_argument("--root", required=True); p.add_argument("--stride", type=int, default=10)
    p.add_argument("--view", default="uav")
    add_out(p)

    p = sub.add_parser("hivau", help="UCF-Crime / XD-Violence 抽帧(需时间边界)")
    p.add_argument("--video-root", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--ann", default=None, help="HIVAU-70k 标注 json")
    p.add_argument("--temporal-ann", default=None, help="UCF-Crime Temporal_Anomaly_Annotation.txt")
    p.add_argument("--fps-anomaly", type=float, default=2.0)
    p.add_argument("--n-normal-frames", type=int, default=4)
    p.add_argument("--ucf-fps", type=float, default=30.0)
    p.add_argument("--view", default="cctv")
    add_out(p)

    p = sub.add_parser("hivau-inspect", help="打印 HIVAU 标注结构(格式不符时先跑这个)")
    p.add_argument("--ann", required=True); p.add_argument("--n", type=int, default=3)

    p = sub.add_parser("hivau-export", help="导出 HIVAU 自带指令标注为 ShareGPT")
    p.add_argument("--ann", required=True); p.add_argument("--out", required=True)
    p.add_argument("--frames-index", default=None, help="hivau 子命令产出的 scenes jsonl")
    p.add_argument("--video-key", default="video")

    a = ap.parse_args()

    if a.cmd == "hivau-inspect":
        hivau.inspect(a.ann, a.n)
        return
    if a.cmd == "hivau-export":
        hivau.export_instructions(a.ann, a.out, video_key=a.video_key,
                                  frames_index=a.frames_index)
        return

    if a.cmd == "mar20":
        scenes = mar20.build(a.root, a.img_dir, a.ann_dir, a.fmt, a.classes, a.view,
                             keep_augmented=a.keep_augmented)
    elif a.cmd == "mendeley":
        scenes = mendeley_mil.build(a.root, a.fmt, a.classes, a.view,
                                    tuple(a.keep_classes) if a.keep_classes else None)
    elif a.cmd == "fasdd":
        scenes = fasdd.build(a.root, a.view, keep_negatives=not a.drop_negatives)
    elif a.cmd == "era":
        scenes = era.build(a.root, a.frames_dir, a.n_frames, a.capera, a.view,
                           include_normal=not a.no_normal, modality=a.modality)
        if a.single_frames:
            scenes += era.build_single_frames(a.root, a.view,
                                              include_normal=not a.no_normal)
    elif a.cmd == "dronecrowd":
        scenes = dronecrowd.build(a.root, a.ann_dir, a.stride, a.head_half, a.view)
    elif a.cmd == "visdrone-det":
        scenes = visdrone.build_det(a.root, a.view)
    elif a.cmd == "visdrone-mot":
        scenes = visdrone.build_mot(a.root, a.stride, a.boundaries,
                                    auto=not a.no_auto_boundary, view=a.view)
    elif a.cmd == "dota":
        scenes = dota.build(a.root, a.tiles_dir, a.tile, a.overlap, a.min_objects,
                            a.empty_ratio, a.seed, a.view)
    elif a.cmd == "drone-anomaly":
        scenes = drone_anomaly.build(a.root, a.stride, a.view)
    else:
        scenes = hivau.build(a.video_root, a.frames_dir, a.ann, a.temporal_ann,
                             a.fps_anomaly, a.n_normal_frames, a.ucf_fps, a.view)
    _report(scenes, a.out)


if __name__ == "__main__":
    main()

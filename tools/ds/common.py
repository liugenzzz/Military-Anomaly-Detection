"""各数据集适配器的共享工具：抽帧、标注格式解析、大图切片、类名归一。

设计要点: **类名映射是按数据集分别定义的, 不做全局映射**。
原因很实际 —— MAR20 的 plane 全是军机, DOTA 的 plane 包含民航客机。
两者映射到同一个类名, 会让民用机场的停机坪被判成装备集结。
所以每个 ds 模块自带 CLASS_MAP, 军用与民用分开命名。
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
VID_EXT = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".mpg", ".mpeg", ".m4v")


# ---------------------------------------------------------------- 图像
def image_size(path: str | Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return 0, 0


def iter_images(root: str | Path, recursive: bool = True) -> list[Path]:
    p = Path(root)
    it = p.rglob("*") if recursive else p.iterdir()
    return sorted(x for x in it if x.suffix.lower() in IMG_EXT)


def iter_videos(root: str | Path) -> list[Path]:
    return sorted(x for x in Path(root).rglob("*") if x.suffix.lower() in VID_EXT)


# ---------------------------------------------------------------- 抽帧
def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def video_duration(path: str | Path) -> float | None:
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True, timeout=60)
            return float(out.stdout.strip())
        except Exception:
            pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return n / fps if fps > 0 else None
    except Exception:
        return None


def extract_frames(video: str | Path, out_dir: str | Path, *,
                   n_frames: int | None = 3, fps: float | None = None,
                   span: tuple[float, float] | None = None,
                   prefix: str | None = None, quality: int = 2) -> list[Path]:
    """抽帧。n_frames 均匀取 N 帧(默认首/中/尾), fps 按帧率抽; 二者取其一。

    span 限定时间区间(秒), 用于只抽异常段或只抽正常段 —— 这是切困难负样本的关键。
    优先用 ffmpeg, 没有则退回 OpenCV; 两者都没有会抛出带安装提示的异常。
    """
    video, out = Path(video), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = prefix or video.stem

    if _has_ffmpeg():
        return _ffmpeg_frames(video, out, stem, n_frames, fps, span, quality)
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "抽帧需要 ffmpeg 或 OpenCV。请任选其一安装：\n"
            "  apt-get install -y ffmpeg      (推荐, 更快更稳)\n"
            "  pip install opencv-python") from e
    return _cv2_frames(video, out, stem, n_frames, fps, span)


def _ffmpeg_frames(video, out, stem, n_frames, fps, span, quality) -> list[Path]:
    dur = video_duration(video)
    made: list[Path] = []

    if fps is not None:
        args = ["ffmpeg", "-v", "error", "-y"]
        if span:
            args += ["-ss", f"{span[0]:.3f}", "-to", f"{span[1]:.3f}"]
        args += ["-i", str(video), "-vf", f"fps={fps}", "-q:v", str(quality),
                 str(out / f"{stem}_f%05d.jpg")]
        subprocess.run(args, check=False, timeout=1800)
        return sorted(out.glob(f"{stem}_f*.jpg"))

    # 均匀取 N 帧: 逐帧 seek, 比 select 滤镜更可控
    lo, hi = span if span else (0.0, dur or 5.0)
    if hi <= lo:
        hi = lo + 0.1
    n = max(1, int(n_frames or 1))
    for i in range(n):
        t = lo + (hi - lo) * ((i + 0.5) / n)
        dst = out / f"{stem}_t{int(t * 1000):07d}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}",
                        "-i", str(video), "-frames:v", "1", "-q:v", str(quality), str(dst)],
                       check=False, timeout=300)
        if dst.exists() and dst.stat().st_size > 0:
            made.append(dst)
    return made


def _cv2_frames(video, out, stem, n_frames, fps, span) -> list[Path]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    lo_f = int((span[0] if span else 0) * src_fps)
    hi_f = int((span[1] if span else (total / src_fps if total else 5)) * src_fps)
    hi_f = min(hi_f, total - 1) if total else hi_f

    if fps is not None:
        step = max(1, int(round(src_fps / fps)))
        idxs = list(range(lo_f, max(lo_f + 1, hi_f + 1), step))
    else:
        n = max(1, int(n_frames or 1))
        idxs = [int(lo_f + (hi_f - lo_f) * ((i + 0.5) / n)) for i in range(n)]

    made = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        dst = out / f"{stem}_f{i:06d}.jpg"
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        made.append(dst)
    cap.release()
    return made


# ---------------------------------------------------------------- 标注格式
def detect_ann_format(root: str | Path) -> str:
    """探测标注格式: coco | voc | yolo | dota | unknown。

    很多社区数据集(尤其 Mendeley/Kaggle)不声明格式, 探测比让用户猜靠谱。
    """
    p = Path(root)
    if any(p.rglob("*.json")):
        for j in list(p.rglob("*.json"))[:5]:
            try:
                d = json.loads(j.read_text(encoding="utf-8"))
                if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
                    return "coco"
            except Exception:
                continue
    if any(p.rglob("*.xml")):
        return "voc"
    txts = list(p.rglob("*.txt"))[:20]
    for t in txts:
        for ln in t.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = ln.split()
            if len(parts) >= 9 and not parts[0].replace(".", "").isdigit():
                return "dota"                       # x1 y1 ... x4 y4 category difficult
            if len(parts) == 5 and all(_isnum(x) for x in parts):
                return "yolo"
            break
    return "unknown"


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_voc_xml(path: str | Path) -> tuple[int, int, list[tuple[str, list[float]]]]:
    """VOC XML -> (w, h, [(cls, [x1,y1,x2,y2])])。兼容缺失 size 节点的文件。"""
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "object").strip()
        bb = o.find("bndbox")
        if bb is None:
            continue
        try:
            objs.append((name, [float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                                float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0"))]))
        except (TypeError, ValueError):
            continue
    return w, h, objs


def parse_yolo_txt(path: str | Path, w: int, h: int,
                   names: list[str]) -> list[tuple[str, list[float]]]:
    out = []
    for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            ci = int(float(p[0]))
            cx, cy, bw, bh = (float(x) for x in p[1:5])
        except ValueError:
            continue
        out.append((names[ci] if 0 <= ci < len(names) else f"class_{ci}",
                    [(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def parse_dota_txt(path: str | Path) -> list[tuple[str, list[float], int]]:
    """DOTA OBB: x1 y1 x2 y2 x3 y3 x4 y4 category difficult -> (cls, 外接HBB, difficult)。"""
    out = []
    for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        p = ln.split()
        if len(p) < 9 or not _isnum(p[0]):
            continue                                 # 跳过 imagesource/gsd 头部行
        xs = [float(p[i]) for i in (0, 2, 4, 6)]
        ys = [float(p[i]) for i in (1, 3, 5, 7)]
        diff = int(p[9]) if len(p) > 9 and p[9].isdigit() else 0
        out.append((p[8], [min(xs), min(ys), max(xs), max(ys)], diff))
    return out


# ---------------------------------------------------------------- 大图切片
def slice_image(img_path: str | Path, out_dir: str | Path, *, tile: int = 1024,
                overlap: int = 200, min_content: float = 0.5) -> list[dict[str, Any]]:
    """把超大遥感图切成 tile x tile 的块。返回 [{path,x,y,w,h}]。

    DOTA 原图可达 20000x20000, 不切片无法送入模型, 且小目标会被缩放到消失。
    min_content 过滤掉边缘那些大部分是空白填充的块。
    """
    from PIL import Image
    img_path, out = Path(img_path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = max(1, tile - overlap)
    made = []
    with Image.open(img_path) as im:
        W, H = im.size
        if W <= tile and H <= tile:                  # 本来就不大, 直接拷一份记录
            dst = out / f"{img_path.stem}__0_0{img_path.suffix}"
            if not dst.exists():
                im.save(dst)
            return [{"path": str(dst), "x": 0, "y": 0, "w": W, "h": H}]
        for y in range(0, max(1, H - overlap), step):
            for x in range(0, max(1, W - overlap), step):
                x2, y2 = min(x + tile, W), min(y + tile, H)
                if (x2 - x) * (y2 - y) < min_content * tile * tile:
                    continue
                dst = out / f"{img_path.stem}__{x}_{y}{img_path.suffix}"
                if not dst.exists():
                    im.crop((x, y, x2, y2)).save(dst)
                made.append({"path": str(dst), "x": x, "y": y, "w": x2 - x, "h": y2 - y})
    return made


def clip_boxes_to_tile(objs: list[tuple[str, list[float]]], tile: dict[str, Any],
                       min_keep: float = 0.6) -> list[tuple[str, list[float]]]:
    """把原图坐标的框裁到切片局部坐标。保留面积占比 >= min_keep 的框。

    被切片边缘截断的目标, 保留半个框会教出错误的尺寸先验, 所以按面积占比丢弃。
    """
    ox, oy, tw, th = tile["x"], tile["y"], tile["w"], tile["h"]
    out = []
    for cls, (x1, y1, x2, y2) in objs:
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue
        cx1, cy1 = max(x1, ox), max(y1, oy)
        cx2, cy2 = min(x2, ox + tw), min(y2, oy + th)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if ((cx2 - cx1) * (cy2 - cy1)) / area < min_keep:
            continue
        out.append((cls, [cx1 - ox, cy1 - oy, cx2 - ox, cy2 - oy]))
    return out


# ---------------------------------------------------------------- 杂项
def points_to_boxes(points: list[tuple[float, float]], half: float = 8.0,
                    w: int = 0, h: int = 0) -> list[list[float]]:
    """点标注(如 DroneCrowd 的人头点)转小方框, 便于统一走 bbox 逻辑。"""
    out = []
    for x, y in points:
        x1, y1, x2, y2 = x - half, y - half, x + half, y + half
        if w:
            x1, x2 = max(0.0, x1), min(float(w), x2)
        if h:
            y1, y2 = max(0.0, y1), min(float(h), y2)
        out.append([x1, y1, x2, y2])
    return out


def chunk_span(total: float, seg: float) -> Iterator[tuple[float, float]]:
    n = max(1, int(math.ceil(total / seg)))
    for i in range(n):
        yield i * seg, min(total, (i + 1) * seg)

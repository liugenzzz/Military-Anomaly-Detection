"""用大模型生成指令型 VQA —— 但把它约束在结构化事实之上。

核心思路: LLM 不做自由发挥, 只做"把已核实的事实改写成多样化的指令问答"。
  规则先算出 FACTS(目标数量/坐标/事件/边界关系)  ->  LLM 依据 FACTS + 图像生成 QA
  ->  第二遍 LLM 校验 QA 与 FACTS 是否一致 -> pass/fix/drop
这样既拿到了 LLM 的语言多样性, 又不会引入幻觉。

三个子命令:
  review    闸5 图像质检 (输出喂给 screen.py --vlm-review)
  generate  生成指令型 QA
  verify    校验并落盘为 LLaMA-Factory ShareGPT

离线批推理: 加 --dry-run, 只写出 requests jsonl, 自己拿 vLLM 批跑完再用
--responses 回灌, 不必让本脚本联网。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds.boxes import BBOX_SCALE, COORD_MODE, box_json, to_bbox2d  # noqa: E402
from build_vqa import region_box_of, scene_quality  # noqa: E402  与规则侧共用一套口径
from sharegpt import make_row  # noqa: E402
from facets import Facet, load_all, load_tool
from scene import Scene, load_scenes

# 描述侧面的抽取权重。**position 不在其中** —— 方位表述由 build_vqa 的
# locate_verbal 用规则精确生成, 零幻觉且免费, 没必要再花 LLM 算力重做一遍。
FACET_WEIGHTS = {
    "evidence": 25, "full": 8,
    # 各类专属侧面, 顺序即权重 18/18/15/12
    "formation": 18, "composition": 18, "scale": 15, "site": 12,
    "intensity": 18, "debris": 18, "extent": 15, "stage": 12,
    "morphology": 18, "color": 18, "drift": 15, "occlusion": 12,
    "trajectory": 18, "timing": 18, "boundary_relation": 15, "group": 12,
    "pace": 14, "aftermath": 14,
    "formation_line": 18, "vehicle_mix": 18, "route": 15, "column_scale": 12,
    "extent_damage": 18, "terrain_change": 18, "affected_objects": 15, "access": 12,
    "hard_neg": 40, "scan": 30,
    # 带框描述: 训练要求里"多模态描述(文字+图像区域)"直接对应这一类, 权重给到最高
    "grounded": 30,
}
SKIP_FACETS = {"position"}
BOX_SCALE = BBOX_SCALE


def facet_applicable(kind: str, s: Scene) -> bool:
    """needs 前置条件的机器可判部分。判不了的交给模型自己返回空。"""
    ev = {e.type: e for e in s.events}
    cluster = next((e for e in s.events if "cluster_bbox" in e.evidence), None)
    # 帧在 s.frames 里, 不在 meta 里。读错地方的后果是 n_frames 恒为 1,
    # 于是 trajectory / timing 这两个越界专属侧面从来没触发过 ——
    # 越界本来就是图最少的一类, 四个专属侧面还被静默砍掉两个。
    n_frames = len(s.frames) or (len(s.media[1]) if s.modality != "image" else 1)
    crossed = [e for e in s.events if e.evidence.get("rule") == "boundary_cross"]
    return {
        "formation": bool(cluster and cluster.evidence.get("count", 0) >= 6),
        "composition": len(s.objects) > 0,
        "scale": cluster is not None,
        "site": True,
        "intensity": "explosion" in ev,
        "debris": "explosion" in ev,
        "extent": "explosion" in ev,
        "stage": "explosion" in ev,
        "morphology": "smoke" in ev,
        "color": "smoke" in ev,
        "drift": "smoke" in ev,
        "occlusion": "smoke" in ev,
        "trajectory": n_frames > 1 and bool(crossed),
        "timing": n_frames > 1 and bool(crossed),
        "boundary_relation": bool(crossed) and bool(s.regions),
        "group": len(crossed) >= 2,
        "pace": bool(crossed) and n_frames > 1,
        "aftermath": bool(crossed) and n_frames > 1,
        # 车队: 有列队事件才出
        "formation_line": "convoy" in ev,
        "vehicle_mix": "convoy" in ev,
        "route": "convoy" in ev,
        "column_scale": "convoy" in ev,
        # 灾害: 有灾害事件才出。access 还要求画面里真有道路类地物, 否则"路通不通"
        # 无从谈起 —— 一片水面里问道路, 模型只能编
        "extent_damage": "disaster" in ev,
        "terrain_change": "disaster" in ev,
        "affected_objects": "disaster" in ev and len(s.objects) > 0,
        "access": "disaster" in ev,
        "hard_neg": bool(s.meta.get("hard_negative")),
        "scan": not s.events,
        "evidence": bool(s.events),
        "grounded": bool(s.events) and region_box_of(s, {}) is not None,
        "full": True,
    }.get(kind, True)


# ---------------------------------------------------------------- FACTS
def _box1000(bbox: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [max(0, min(BOX_SCALE, round(x1 / w * BOX_SCALE))),
            max(0, min(BOX_SCALE, round(y1 / h * BOX_SCALE))),
            max(0, min(BOX_SCALE, round(x2 / w * BOX_SCALE))),
            max(0, min(BOX_SCALE, round(y2 / h * BOX_SCALE)))]


def _quadrant(cx: float, cy: float, w: int, h: int) -> str:
    v = "上" if cy < h / 3 else ("下" if cy > h * 2 / 3 else "中")
    hz = "左" if cx < w / 3 else ("右" if cx > w * 2 / 3 else "中")
    return "画面中央" if (v == "中" and hz == "中") else f"画面{v}{hz}方"


def build_facts(scene: Scene, onto: dict[str, Any], max_objects: int = 30) -> dict[str, Any]:
    """从 Scene 抽出可核查的事实包。只放规则算得出来的东西, 一个字都不许猜。"""
    zh = {c["id"]: c["zh"] for c in onto["classes"]}
    cues = {c["id"]: c.get("cues", []) for c in onto["classes"]}

    summary: dict[str, int] = {}
    for o in scene.objects:
        summary[o.cls] = summary.get(o.cls, 0) + 1

    # 目标多时按面积取前 N 个给坐标, 但 summary 里的总数始终是全量真值
    objs = sorted(scene.objects, key=lambda o: -o.area)[:max_objects]
    obj_list = []
    for o in objs:
        cx, cy = o.center
        item = {"cls": o.cls, "box_1000": _box1000(o.bbox, scene.width, scene.height),
                "position": _quadrant(cx, cy, scene.width, scene.height)}
        if "moving" in o.attrs:
            item["moving"] = bool(o.attrs["moving"])
        obj_list.append(item)

    events = []
    for e in scene.events:
        ev: dict[str, Any] = {"type": e.type, "zh": zh.get(e.type, e.type),
                              "cues": cues.get(e.type, [])}
        d = e.evidence
        if "count" in d:
            ev["object_count"] = d["count"]
        if "cluster_bbox" in d:
            ev["region_box_1000"] = _box1000(d["cluster_bbox"], scene.width, scene.height)
        if d.get("rule") == "boundary_cross":
            ev["boundary"] = d.get("region")
            ev["direction"] = d.get("direction")
            ev["object_class"] = d.get("cls")
        if "r2" in d:
            ev["collinearity"] = d["r2"]
        events.append(ev)

    return {
        "view": scene.view,
        "image_size": [scene.width, scene.height],
        "coordinate_note": f"box_1000 为归一化到 0-{BOX_SCALE} 的 [x1,y1,x2,y2]",
        "objects_summary": summary,
        "objects_total": len(scene.objects),
        "objects": obj_list,
        "objects_truncated": len(scene.objects) > max_objects,
        "events": events,
        "regions": [{"name": r.name, "type": r.type} for r in scene.regions],
        "source_caption": scene.caption,
        "absent_anomaly_types": [c["id"] for c in onto["classes"]
                                 if c["id"] != "normal" and c.get("enabled", True)
                                 and c["id"] not in {e.type for e in scene.events}],
        "hard_negative": bool(scene.meta.get("hard_negative")),
        "hard_negative_reason": scene.meta.get("hard_negative_reason", []),
    }


# ---------------------------------------------------------------- LLM 客户端
DEFAULT_MODEL = "Qwen3.8-27B"        # 与自建推理池同一套服务

# 内联大图前先缩到的长边。0 = 不缩。由 load_endpoints 从 llm.max_image_side 读入,
# 这样四个调用点不用各传一遍。
MAX_IMAGE_SIDE = 0
IMAGE_QUALITY = 88


@dataclass
class Endpoint:
    """一路端点。**url 原样使用, 不做任何猜测。**

    以前这里有一套 _norm_base: 把 /chat/completions 剥掉, 调用时再拼回去。
    绕了一圈还是同一个地址, 却多出三种出错的可能。现在的规矩只有一条:
      - 你填的是完整路径(.../v1/chat/completions) -> 原样用
      - 你只填到 /v1                              -> 补上 /chat/completions
    /v1/models 由 chat 地址换个尾巴得到, 不另外配。
    """
    url: str
    model: str
    key: str = ""
    concurrency: int = 4
    name: str = ""
    max_tokens: int = 4096
    timeout: int = 600
    # 推理型模型(Qwen3 系列)默认会先输出思维链。不关掉的话 content 里全是
    # "我们需要回答用户：..." 这种推理痕迹, json_mode 直接解析失败。
    # 服务端开了 reasoning parser 时思维链落在 reasoning_content, content 是干净的;
    # 没开 parser 时 <think>...</think> 原样留在 content 里 —— 两种都要防。
    chat_template_kwargs: dict | None = None

    def __post_init__(self):
        u = self.url.strip().rstrip("/")
        self.chat_url = u if u.endswith("/chat/completions") else u + "/chat/completions"
        self.models_url = self.chat_url[: -len("/chat/completions")] + "/models"
        self.name = self.name or self.chat_url
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {"enable_thinking": False}


def load_endpoints(path: str | None, role: str, model: str,
                   base_url: str | None) -> list[Endpoint]:
    """端点池。来源只有两个: 配置文件, 或 --base-url。**没有环境变量兜底** ——
    那条链是"静默连上外网"的根源: 配置读不到时它不报错, 而是去连 api.openai.com,
    跑完一整轮才在报告里看到满屏 SSL 失败。

    配置按角色分组, generate / review / screen。**审稿必须换模型** ——
    同一个模型审自己写的答案基本全过。review 留空会退回 generate 并告警。
    """
    if path:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        # endpoints: 底下分组(configs/generate.yaml), 或直接放在顶层(老格式)
        src = cfg.get("endpoints") if isinstance(cfg.get("endpoints"), dict) else cfg
        group = src.get(role)
        # `review: []` 写的是空列表不是缺省, 判 `is None` 落不进回退, 于是
        # 一路掉到最后那句 SystemExit —— 明明 generate 三路都好好的。
        # 非 generate 角色只要是空的(缺省 / [] / null)就一律退回 generate。
        if not group and role != "generate":
            group = src.get("generate") or []
        llm_cfg = cfg.get("llm") or {}
        global MAX_IMAGE_SIDE, IMAGE_QUALITY
        MAX_IMAGE_SIDE = int(llm_cfg.get("max_image_side", 0) or 0)
        IMAGE_QUALITY = int(llm_cfg.get("image_jpeg_quality", 88))
        out = [Endpoint(url=e["url"], model=e.get("model", model),
                        key=str(e.get("key", "")),
                        concurrency=int(e.get("concurrency", 4)),
                        name=e.get("name", ""),
                        max_tokens=int(e.get("max_tokens",
                                             llm_cfg.get("max_tokens", 4096))),
                        timeout=int(e.get("timeout", llm_cfg.get("timeout", 600))),
                        chat_template_kwargs=e.get(
                            "chat_template_kwargs",
                            llm_cfg.get("chat_template_kwargs")))
               for e in group if e.get("enabled") is not False]
        if out:
            return out
    if base_url:
        return [Endpoint(url=u, model=model) for u in base_url.split(",") if u.strip()]
    raise SystemExit(
        "没有可用的推理端点。\n"
        "  - 配置文件:  --endpoints configs/generate.yaml   (默认就读它)\n"
        "  - 或单地址:  --base-url http://10.107.226.27:8001/v1/chat/completions\n"
        "先跑 `python tools/llm_qa.py ping` 确认每一路通不通。")


# ---------------------------------------------------------------- 思维链 / 报错体
# 推理型模型默认吐思维链。服务端开了 reasoning parser 时它落在
# message.reasoning_content, content 干净; 没开 parser 时 <think>...</think>
# 原样留在 content 里, 必须在解析 JSON 之前剥掉, 否则 json_mode 永远失败。
_THINK_TAG = r"think|thinking|reasoning|reason"
_THINK_BLOCK = re.compile(rf"<\s*({_THINK_TAG})\s*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_THINK_CLOSE = re.compile(rf"^.*<\s*/\s*(?:{_THINK_TAG})\s*>", re.I | re.S)
_THINK_OPEN = re.compile(rf"<\s*(?:{_THINK_TAG})\s*>", re.I)


def strip_reasoning(text: str) -> str:
    """剥掉思维链, 只留最终回答。"""
    if not text:
        return ""
    out = _THINK_BLOCK.sub("", text)
    if _THINK_CLOSE.search(out):                 # 只有闭合标签: 丢掉它之前的一切
        out = _THINK_CLOSE.sub("", out, count=1)
    m = _THINK_OPEN.search(out)
    if m:                                        # 只有开标签: 被 max_tokens 截断了
        out = out[: m.start()]
    return out.strip()


def http_detail(e: Exception) -> str:
    """把服务端的报错体抠出来。

    urllib 的 HTTPError 字符串只有 "HTTP Error 500: Internal Server Error",
    真正有用的 traceback 在 body 里。以前这里只打 str(e)[:90], 于是图像检查
    报 500 时完全看不出是图太小、是 --limit-mm-per-prompt 没开、还是模型没视觉层。
    """
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:                        # noqa: BLE001
            body = ""
        hint = {401: "  ← key 不对或没发送",
                403: "  ← key 没权限",
                404: "  ← url 或 model 名不对, 确认与 --served-model-name 一致",
                400: "  ← 请求体被拒, 常见是超 max_model_len / max_tokens 过大",
                500: "  ← 服务端内部异常, 报错体见下"}.get(e.code, "")
        return f"HTTP {e.code}{hint}: {body[:600] or '(空)'}"
    return f"{type(e).__name__}: {e}"


class LLM:
    """OpenAI 兼容接口, 支持多端点池。

    每一路各带自己的 model 和 key —— 局域网那十几路共用一个 local-pool-key,
    云端那几路各有各的 sk, 一个全局 key 配不下来。
    请求按各路的 concurrency 加权轮转; 某一路连不上就摘掉, 其余照跑。
    """

    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "VLM_API_KEY", cache_dir: str | None = None,
                 temperature: float = 0.7, max_retries: int = 3,
                 endpoints_file: str | None = None, role: str = "generate"):
        self.pool = load_endpoints(endpoints_file, role, model, base_url)
        # 按 concurrency 加权: 能扛 8 并发的那路就该多分到一倍的请求
        self.ring: list[Endpoint] = []
        for e in self.pool:
            self.ring += [e] * max(1, e.concurrency)
        self.model = model
        self.dead: set[str] = set()
        self._rr = 0
        self._lock = threading.Lock()
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    @property
    def total_concurrency(self) -> int:
        return sum(e.concurrency for e in self.pool)

    def describe(self) -> str:
        models = sorted({e.model for e in self.pool})
        return (f"{len(self.pool)} 路端点 / 总并发 {self.total_concurrency} / "
                f"模型 {'、'.join(models)}")

    def pick(self) -> Endpoint:
        with self._lock:
            alive = [e for e in self.ring if e.chat_url not in self.dead] or self.ring
            self._rr = (self._rr + 1) % len(alive)
            return alive[self._rr]

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         ensure_ascii=False).encode()).hexdigest()[:32]

    def chat(self, messages: list[dict], json_mode: bool = True) -> str:
        import urllib.request                            # 惰性导入, dry-run 时用不到

        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "temperature": self.temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        payload.setdefault("model", self.model)
        cache_file = self.cache / f"{self._key(payload)}.json" if self.cache else None
        if cache_file and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["content"]

        last = None
        for attempt in range(self.max_retries):
            ep = self.pick()
            url = ep.chat_url                    # 原样用, 不拼不猜
            payload["model"] = ep.model          # 每路的模型名可能不同
            payload["max_tokens"] = ep.max_tokens
            if ep.chat_template_kwargs:          # 关思考, 见 Endpoint 注释
                payload["chat_template_kwargs"] = ep.chat_template_kwargs
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {ep.key}"})
                with urllib.request.urlopen(req, timeout=ep.timeout) as r:
                    msg = json.loads(r.read())["choices"][0].get("message") or {}
                content = strip_reasoning(msg.get("content") or "")
                if not content:
                    # 回答全落在思维链里 —— 没关思考且被 max_tokens 截断
                    if str(msg.get("reasoning_content") or "").strip():
                        raise RuntimeError(
                            f"{ep.name} 只返回了思维链没返回答案 —— "
                            f"确认端点的 chat_template_kwargs.enable_thinking=false, "
                            f"或把 max_tokens({ep.max_tokens}) 调大")
                    raise RuntimeError(f"{ep.name} 返回了空内容")
                if cache_file:
                    cache_file.write_text(json.dumps({"content": content}, ensure_ascii=False),
                                          encoding="utf-8")
                return content
            except Exception as e:                       # noqa: BLE001
                last = http_detail(e)
                import urllib.error
                # 连不上的那一路摘掉, 别再往里发。**HTTPError 不算连不上** ——
                # 服务端应答了, 只是这一条请求被拒, 摘掉整路是误伤。
                if (isinstance(e, OSError)
                        and not isinstance(e, urllib.error.HTTPError)
                        and len(self.pool) > 1):
                    self.dead.add(url)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败({url}): {last}")


def _shrink(path: Path, max_side: int, quality: int) -> tuple[bytes, str] | None:
    """大图先缩到 max_side 再内联。

    DOTA 那种 4000x4000 的原图 base64 之后是十几 MB, 一是网络来回慢, 二是
    服务端 Qwen-VL 本来就会按 max_pixels = 1280*28*28 缩回去, 传全尺寸纯属浪费。
    没装 Pillow 就原样传, 不因为缺个依赖就跑不动。
    """
    try:
        from PIL import Image                            # type: ignore
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            if max(im.size) <= max_side:
                return None
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except Exception:                                    # noqa: BLE001
        return None                                      # 读不了就原样传, 让服务端报


def image_message(text: str, image_path: str | None, inline: bool,
                  max_side: int | None = None, quality: int | None = None) -> list[dict]:
    if not image_path:
        return [{"type": "text", "text": text}]
    max_side = MAX_IMAGE_SIDE if max_side is None else max_side
    quality = IMAGE_QUALITY if quality is None else quality
    p = Path(image_path)
    if inline and p.exists():
        small = _shrink(p, max_side, quality) if max_side else None
        if small:
            raw, mime = small
        else:
            raw = p.read_bytes()
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(raw).decode()
        url = f"data:{mime};base64,{b64}"
    else:
        url = str(p)
    # 文字在前、图在后。Qwen 的 chat template 对两种顺序都能渲染, 但参考项目
    # (book_cpt/services/clients.py)跑通的是这个顺序, 保持一致省得踩模板差异。
    return [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}}]


def parse_json(text: str) -> dict[str, Any] | None:
    """LLM 输出常带 markdown 围栏或前后废话, 尽量抠出 JSON。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------- 子命令
def load_prompt(name: str, prompt_dir: str) -> str:
    return Path(prompt_dir, name).read_text(encoding="utf-8")


def load_scenes_excluding(path: str, exclude_file: str | None) -> list[Scene]:
    scenes = load_scenes(path)
    if not exclude_file:
        return scenes
    excl = {ln.strip() for ln in Path(exclude_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
    kept = [s for s in scenes if s.image_id not in excl]
    print(f"排除 golden set: {len(scenes)} -> {len(kept)} 个 scene")
    return kept


def cmd_screen(args, onto):
    tmpl = load_prompt("screen_image.txt", args.prompt_dir + "/_tools")
    scenes = load_scenes_excluding(args.scenes, args.exclude_ids)
    zh = {c["id"]: c["zh"] for c in onto["classes"]}

    def make(s: Scene) -> dict:
        claimed = "、".join(zh.get(t, t) for t in s.anomaly_types) or "无异常的正常场景"
        return {"image_id": s.image_id, "image_path": s.image_path,
                "prompt": tmpl.format(claimed=claimed)}

    reqs = [make(s) for s in scenes]
    if args.dry_run:
        _write_requests(reqs, args.out)
        return

    llm = LLM(args.model, args.base_url, cache_dir=args.cache_dir, temperature=0.0)

    def run(r):
        msg = [{"role": "user", "content": image_message(r["prompt"], r["image_path"], args.inline_images)}]
        d = parse_json(llm.chat(msg)) or {}
        d["image_id"] = r["image_id"]
        return d

    _run_and_write(reqs, run, args.out, args.workers)


# 描述侧面只需要"数了多少、有什么事件", 不需要每个目标的坐标。
# **坐标出现在 FACTS 里会诱导模型把坐标写进答案**, 而描述题恰恰禁止输出坐标 ——
# 与其生成完再靠 must-not 拦, 不如一开始就不给它看。
FACTS_KEEP_BOXES = {"reason"}


def _modality_note(s: Scene) -> str:
    """告诉模型这是视频还是静帧 —— 不说明的话, 模型不知道自己能不能谈运动。"""
    if s.modality == "video":
        return ("## 输入形态\n这是一段约 5 秒的视频。你可以描述运动、变化与持续过程，"
                "但**不要凭空推断视频之外的时间**。\n")
    if s.modality == "multi_image":
        return (f"## 输入形态\n这是 {len(s.frames)} 帧按时间先后排列的画面。"
                "可以描述帧与帧之间的变化，帧序即时序。\n")
    return "## 输入形态\n这是一张静止画面。**不要描述运动、变化或持续过程**，单帧看不出这些。\n"


def trim_facts(facts: dict[str, Any], facet: str) -> dict[str, Any]:
    """按侧面裁剪事实包: 描述类去掉逐目标坐标, 只留汇总与事件。"""
    if facet in FACTS_KEEP_BOXES:
        return facts
    out = {k: v for k, v in facts.items()
           if k not in ("objects", "coordinate_note", "objects_truncated")}
    evs = []
    for e in out.get("events", []):
        evs.append({k: v for k, v in e.items() if k != "region_box_1000"})
    if evs:
        out["events"] = evs
    return out


def _pick_facets(s: Scene, facets: dict[str, list[Facet]], anomaly: str,
                 rng: random.Random, k: int) -> list[Facet]:
    """按权重抽 k 个适用的侧面。不适用的直接排除, 不浪费一次调用。"""
    pool = [f for f in facets.get(anomaly, [])
            if f.kind not in SKIP_FACETS and facet_applicable(f.kind, s)]
    if not pool:
        return []
    # 至少留一个不抽: 正常图的可用侧面本来就少(scan/hard_neg/full),
    # 每次都抽干会让 full 这种"什么图都能出"的侧面占比虚高。
    # 异常图的池子有 6 个侧面, 这条限制不影响把 facets-per-image 开到 4~5。
    k = min(k, max(1, len(pool) - 1))
    picked: list[Facet] = []
    for _ in range(k):
        rest = [f for f in pool if f not in picked]
        w = [FACET_WEIGHTS.get(f.kind, 10) for f in rest]
        picked.append(rng.choices(rest, weights=w)[0])
    return picked


MAX_FACETS_PER_IMAGE = 8   # 越界类补了两个侧面后池子到 9 个   # 侧面池最多 6 个(3 通用 + 4 专属, 再减去至少留一个不抽)


def _pool_size(s: Scene, facets: dict[str, list[Facet]], anomaly: str) -> int:
    return len([f for f in facets.get(anomaly, [])
                if f.kind not in SKIP_FACETS and facet_applicable(f.kind, s)])


def _yield_of(sizes: list[int], k: int) -> int:
    """k 个侧面时这批 scene 能出多少条描述题。

    每张图实际能抽的数量受侧面池限制(见 _pick_facets 里那条"至少留一个不抽"),
    所以不能简单地 len(scenes) * k —— 正常图池子只有 3 个, 开到 5 也白开。
    """
    return sum(min(k, max(1, n - 1)) for n in sizes if n)


def plan_quota(scenes: list[Scene], facets: dict[str, list[Facet]], target: int,
               base_k: int, reason_ratio: float, seed: int) -> dict[str, tuple[list[Scene], int]]:
    """把"每类 N 条"换算成每类的 (参与的 scene, 每图抽几个侧面)。

    两个方向都要调:
      - 稀缺类(border_crossing 这种源数据本来就少的)把 facets-per-image 往上顶,
        顶到侧面池抽干为止, 还不够就如实报缺口, 不靠复制样本凑数;
      - 富余类(explosion/smoke)按 **图** 下采样, 不是逐条丢, 同图的几条要一起走。
    """
    # 归类口径与规则侧一致: 同时属于多类的图归到最稀缺的那一类。
    # 取 anomaly_types[0] 是错的 —— 烟雾与爆炸同框的图会全被算成第一个类,
    # 另一类的配额凭空少掉一大块。
    total: dict[str, int] = {}
    for s in scenes:
        for c in (s.anomaly_types or ["normal"]):
            total[c] = total.get(c, 0) + 1
    by_cls: dict[str, list[Scene]] = {}
    for s in scenes:
        cls = min(s.anomaly_types or ["normal"], key=lambda c: (total.get(c, 0), c))
        by_cls.setdefault(cls, []).append(s)

    rng = random.Random(seed)
    plan: dict[str, tuple[list[Scene], int]] = {}
    report: list[tuple[str, int, int, int, int]] = []   # 类, scene 数, k, 预估条数, 目标
    for cls, group in by_cls.items():
        # 正常样本按 3:7 配到异常总量上, 不单独设目标
        tgt = target if cls != "normal" else int(target * max(1, len(by_cls) - 1) * 3 / 7)
        sizes = [_pool_size(s, facets, cls) for s in group]
        extra = reason_ratio * sum(1 for s in group if s.events or s.meta.get("hard_negative"))

        k = base_k
        while k < MAX_FACETS_PER_IMAGE and _yield_of(sizes, k) + extra < tgt:
            k += 1
        est = _yield_of(sizes, k) + extra

        keep = group
        if est > tgt * 1.05 and k == base_k:
            # 先把 k 压到 1 再看, 能靠少抽侧面满足就不丢图 —— 图的多样性比每图条数值钱
            while k > 1 and _yield_of(sizes, k - 1) + extra >= tgt:
                k -= 1
                est = _yield_of(sizes, k) + extra
            if est > tgt * 1.05:
                # 富余的不随机丢, 按质量分在各数据源之间轮着取 —— 同 build_vqa 的口径
                q = scene_quality(group)
                by_src: dict[str, list[Scene]] = {}
                for sc in group:
                    by_src.setdefault(sc.source_dataset, []).append(sc)
                for src in by_src:
                    by_src[src].sort(key=lambda sc: (-q.get(sc.image_id, 0.5), sc.image_id))
                srcs = sorted(by_src)
                rng.shuffle(srcs)
                cur = {src: 0 for src in srcs}
                acc, picked = 0.0, []
                per = est / max(1, len(group))
                while acc < tgt and any(cur[src] < len(by_src[src]) for src in srcs):
                    for src in srcs:
                        if cur[src] >= len(by_src[src]) or acc >= tgt:
                            continue
                        picked.append(by_src[src][cur[src]])
                        cur[src] += 1
                        acc += per
                keep = picked
                est = _yield_of([_pool_size(s, facets, cls) for s in keep], k) + \
                      reason_ratio * sum(1 for s in keep if s.events or s.meta.get("hard_negative"))
        plan[cls] = (keep, k)
        report.append((cls, len(keep), k, int(est), tgt))

    print("\n按类别配额(LLM 描述侧):")
    for cls, n, k, est, tgt in sorted(report, key=lambda r: -r[3]):
        # 侧面数已经顶到池子上限还不够, 说明是这一类的图就这么多, 不是配额砍的
        flag = ("✅" if est >= tgt * 0.8 else
                "⚠ 数据见底(每图侧面已抽满)" if k >= MAX_FACETS_PER_IMAGE else
                f"⚠ 差 {tgt - est}")
        print(f"  {cls:18s} scene {n:6d} × {k} 侧面 ≈ {est:7d} 条 / 目标 {tgt:6d}  {flag}")
    anom = {c: e for c, _, _, e, _ in
            ((r[0], r[1], r[2], r[3], r[4]) for r in report) if c != "normal"}
    if len(anom) >= 2:
        hi, lo = max(anom.items(), key=lambda kv: kv[1]), min(anom.items(), key=lambda kv: kv[1])
        ratio = hi[1] / max(1, lo[1])
        verdict = ("✅ 均衡" if ratio <= 2 else
                   "✅ 轻微不均, 不影响训练" if ratio <= 3 else
                   "⚠ 偏斜明显, 建议训练时对少的那类加采样权重")
        print(f"  类间比例: 最多 {hi[0]} {hi[1]} / 最少 {lo[0]} {lo[1]} = "
              f"{ratio:.1f}:1  {verdict}")
    return plan


def _solid_png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    """纯 stdlib 生成一张 w*h 的单色 PNG, 不依赖 Pillow。ping 的测试图用。"""
    import struct
    import zlib
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def cmd_ping(args, onto):
    if not args.endpoints and Path("configs/generate.yaml").exists():
        args.endpoints = "configs/generate.yaml"

    """逐路体检端点池。**开跑之前先跑这个** —— 七万条请求跑到一半才发现
    某一路的模型名不对, 那一路的产出全是废的。

    检四件事:
      1. 通不通            —— 连不上还是 401/403
      2. 服务的模型名叫什么 —— vLLM 的 model 字段必须和 --served-model-name 完全一致,
                             常见坑是服务端用的是模型路径而配置里写的是简称
      3. 文本能不能生成
      4. **认不认图**       —— 发一张 504x504 的纯红测试图, 问它什么颜色。
                             答不上来的那一路只能写纯文本, 描述题不能用它。

    测试图为什么是 504x504 而不是随手一张 1x1:
      Qwen-VL 的图像预处理有个下限 min_pixels = 256*28*28 = 200704。
      小于这个尺寸的图进到 patch embedding 会直接在服务端炸掉, vLLM 回的是
      **HTTP 500**(服务端异常)而不是 400(请求非法) —— 看起来就像"多模态模型
      读不了图", 其实是测试图不合法。504 = 18*28, 面积 254016, 稳稳过线。
    """
    import base64
    import urllib.request

    probe_png = base64.b64encode(_solid_png(504, 504, (214, 40, 40))).decode()

    # **先看配置再连服务**。反过来的话, review 留空时 load_endpoints 会先抛
    # SystemExit, "未配置 → 跳过"那段永远轮不到执行, generate 三路明明全绿
    # 也照样以一句"没有可用的推理端点"收场。
    raw_by_role: dict[str, list] = {}
    if args.endpoints:
        _cfg = yaml.safe_load(Path(args.endpoints).read_text(encoding="utf-8")) or {}
        _src = (_cfg.get("endpoints")
                if isinstance(_cfg.get("endpoints"), dict) else _cfg)
        raw_by_role = {r: (_src.get(r) or []) for r in ("generate", "review", "screen")}

    for role in ("generate", "review", "screen"):
        if role != "generate" and not args.endpoints:
            break
        if role != "generate" and not raw_by_role.get(role):
            print(f"\n[{role}] 未配置" +
                  ("  ← 审稿没有独立模型, 会退回 generate 池, 六维 review 形同虚设"
                   if role == "review" else "  ← 会复用 generate 池"))
            continue
        pool = load_endpoints(args.endpoints, role, args.model, args.base_url)
        print(f"\n[{role}] {len(pool)} 路")
        for e in pool:
            tag = f"  {e.name:14s} {e.chat_url}"
            # 1) /v1/models
            served = None
            try:
                req = urllib.request.Request(e.models_url,
                                             headers={"Authorization": f"Bearer {e.key}"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    served = [m.get("id") for m in json.loads(r.read()).get("data", [])]
            except Exception as ex:                      # noqa: BLE001
                print(f"{tag}\n      ✗ /v1/models 不通: {type(ex).__name__}: {ex}")
                continue
            hit = e.model in (served or [])
            mark = "✓" if hit else "✗"
            print(f"{tag}\n      {mark} 服务的模型: {served}   配置写的: {e.model}")
            if not hit:
                print(f"      ↑ **对不上**。model 字段必须和服务端完全一致, "
                      f"把配置里的 model 改成上面列出的名字")
                continue
            # 2) 文本
            one = LLM(e.model, e.chat_url, cache_dir=None, temperature=0.0)
            one.pool = [e]
            one.ring = [e]
            try:
                txt = one.chat([{"role": "user", "content": "回答两个字：收到"}],
                               json_mode=False)
                print(f"      ✓ 文本生成: {txt.strip()[:20]}")
            except Exception as ex:                      # noqa: BLE001
                print(f"      ✗ 文本生成失败: {str(ex)[:90]}")
                continue
            # 3) 认不认图
            try:
                msg = [{"role": "user", "content": [
                    {"type": "text", "text": "这张图是什么颜色？只答颜色两个字。"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{probe_png}"}}]}]
                ans = one.chat(msg, json_mode=False).strip()
                # 答对颜色才算真看见了。有些纯文本模型会把图默默丢掉然后瞎猜一个,
                # 不核对内容的话这一路会被误判成可用。
                seen = "红" in ans
                print(f"      {'✓' if seen else '?'} 接受图像输入: {ans[:30]}"
                      + ("" if seen else "   ← 测试图是纯红, 答得不对, 疑似没真看图"))
            except Exception as ex:                      # noqa: BLE001
                print(f"      ✗ **不接受图像输入**: {http_detail(ex)}")
                print("      ↑ 先看上面的报错体再下结论:")
                print("        · 报 500 且提到 pixel/patch/resize → 图尺寸问题")
                print("        · 报 400 且提到 multimodal/limit_mm → 服务端起的时候"
                      "没给 --limit-mm-per-prompt image=1")
                print("        · 报 400 且说不认识 image_url → 这一路确实是纯文本模型")
                print("        · 其他 500 → 去看 vLLM 那边的 traceback")


def cmd_generate(args, onto):
    tmpl = load_prompt("describe_gen.txt", args.prompt_dir + "/_tools")
    reason_tmpl = load_prompt("reason_gen.txt", args.prompt_dir + "/_tools")
    facets, _ = load_all(args.prompt_dir)
    zh = {c["id"]: c["zh"] for c in onto["classes"]}
    scenes = load_scenes_excluding(args.scenes, args.exclude_ids)
    rng = random.Random(args.seed)
    cls_total0: dict[str, int] = {}
    for sc in scenes:
        for c in (sc.anomaly_types or ["normal"]):
            cls_total0[c] = cls_total0.get(c, 0) + 1

    if args.sample:
        by: dict[str, list[Scene]] = {}
        for sc in scenes:
            by.setdefault(min(sc.anomaly_types or ["normal"],
                              key=lambda c: (cls_total0.get(c, 0), c)), []).append(sc)
        per = max(1, args.sample // max(1, len(by)))
        picked: list[Scene] = []
        for cls in sorted(by):
            pool = by[cls][:]
            rng.shuffle(pool)
            picked += pool[:per]
        scenes = picked[:args.sample]
        print(f"试水批: 分层抽 {len(scenes)} 个 scene, 覆盖 {len(by)} 个类别")

    plan = (plan_quota(scenes, facets, args.target_per_class, args.facets_per_image,
                       args.reason_ratio, args.seed)
            if args.target_per_class else None)
    if plan:
        keep = {id(s) for ss, _ in plan.values() for s in ss}
        scenes = [s for s in scenes if id(s) in keep]

    reqs: list[dict] = []
    n_skip = 0
    cls_total: dict[str, int] = {}
    for s in scenes:
        for c in (s.anomaly_types or ["normal"]):
            cls_total[c] = cls_total.get(c, 0) + 1
    for s in scenes:
        anomaly = min(s.anomaly_types or ["normal"],
                      key=lambda c: (cls_total.get(c, 0), c))
        facts = build_facts(s, onto, args.max_objects)
        k = plan[anomaly][1] if plan and anomaly in plan else args.facets_per_image
        picked = _pick_facets(s, facets, anomaly, rng, k)
        if not picked:
            n_skip += 1
        for fa in picked:
            q = rng.choice(fa.q_bank).replace("{zh}", zh.get(anomaly, "异常"))
            bans = fa.bans_for(anomaly)
            facts_str = json.dumps(trim_facts(facts, fa.kind), ensure_ascii=False, indent=1)
            region = None
            if fa.kind == "grounded" and (rb := region_box_of(s, zh)):
                box, label = rb
                region = {"box_1000": to_bbox2d(box, s.width, s.height), "label": label}
            reqs.append({
                "region": region,
                "image_id": s.image_id, "image_path": s.image_path,
                "media_field": s.media[0], "media": s.media[1],
                "modality": s.modality, "kind": "describe", "facet": fa.kind, "anomaly": anomaly,
                "question": q, "must_not": bans, "facts": facts,
                "source_dataset": s.source_dataset, "license": s.license,
                "view": s.view,
                "width": s.width, "height": s.height,
                "prompt": tmpl.format(
                    facts=facts_str, kind=fa.kind,
                    kind_zh=fa.meta.get("zh", fa.kind),
                    answer_spec=fa.answer_spec,
                    q_example=fa.q_example or (fa.q_bank[0] if fa.q_bank else q),
                    a_example=fa.example_for(anomaly),
                    question=q, must_not="、".join(bans) or "（无）",
                    modality_note=_modality_note(s)),
            })
        # 推理题: 有事件或困难负样本的图才出
        if (s.events or s.meta.get("hard_negative")) and rng.random() < args.reason_ratio:
            _, asks = load_all(args.prompt_dir)
            rq = rng.choice(asks["reason"].lines).replace("{zh}", zh.get(anomaly, "异常"))
            reqs.append({
                "image_id": s.image_id, "image_path": s.image_path,
                "media_field": s.media[0], "media": s.media[1],
                "modality": s.modality, "kind": "reason", "facet": "reason", "anomaly": anomaly,
                "question": rq, "must_not": [], "facts": facts,
                "source_dataset": s.source_dataset, "license": s.license,
                "view": s.view,
                "width": s.width, "height": s.height,
                "prompt": reason_tmpl.format(
                    facts=json.dumps(trim_facts(facts, "reason"), ensure_ascii=False, indent=1),
                    question=rq),
            })

    if n_skip:
        print(f"[{n_skip}] 个 scene 没有适用的侧面, 已跳过")
    if args.dry_run:
        _write_requests(reqs, args.out)
        return

    llm = LLM(args.model, args.base_url, cache_dir=args.cache_dir, temperature=args.temperature,
              endpoints_file=args.endpoints, role="generate")
    print(f"生成端: {llm.describe()}")
    if args.workers <= 0:
        args.workers = llm.total_concurrency     # 0 = 跟着池子的总并发走, 不用手算
        print(f"  并发跟随池子: {args.workers}")
    elif args.workers < llm.total_concurrency:
        print(f"  [提示] --workers {args.workers} 小于池子总并发 {llm.total_concurrency}, "
              f"这些机器没吃满; 填 0 让它自动跟随")

    def run(r):
        msg = [{"role": "user", "content": image_message(r["prompt"], r["image_path"], args.inline_images)}]
        ans = llm.chat(msg, json_mode=False).strip().strip('"')
        return {**{k: v for k, v in r.items() if k != "prompt"}, "answer": ans}

    _run_and_write(reqs, run, args.out, args.workers)


def cmd_verify(args, onto):
    """must-not 硬过滤 -> 六维 review -> 落盘 ShareGPT。

    硬过滤放在 review 之前: 它是纯字符串检查, 零成本, 先把跑题的滤掉,
    再花算力做 review。
    """
    sysdir = Path(args.prompt_dir) / "system"
    systems = ([f.read_text(encoding="utf-8").strip() for f in sorted(sysdir.glob("*.txt"))]
               if sysdir.is_dir() else
               [(Path(args.prompt_dir) / "system.txt").read_text(encoding="utf-8").strip()])
    sys_rng = random.Random(0)      # 同上: 不让模型把一段长 system 背成常量
    review_tmpl = load_prompt("review.txt", args.prompt_dir + "/_tools")
    records = [json.loads(ln) for ln in
               Path(args.generated).read_text(encoding="utf-8").splitlines() if ln.strip()]

    stat = {"total": len(records), "empty": 0, "must_not": 0, "no_facts": 0,
            "pass": 0, "fail": 0}
    kept: list[dict] = []
    violations: dict[str, int] = {}

    # ── 闸一: 纯规则硬过滤
    for r in records:
        ans = (r.get("answer") or "").strip()
        if not ans or ans in {"空", '""', "null", "无"}:
            stat["empty"] += 1
            continue
        if bad := [w for w in r.get("must_not", []) if w and w in ans]:
            stat["must_not"] += 1
            for w in bad:
                violations[w] = violations.get(w, 0) + 1
            continue
        if r["kind"] == "describe" and not r["facts"].get("events") \
                and r["facet"] not in {"scan", "hard_neg", "full"} :
            stat["no_facts"] += 1
            continue
        kept.append(r)

    print(f"硬过滤: {stat['total']} -> {len(kept)}  "
          f"(空 {stat['empty']} / 踩禁用词 {stat['must_not']} / 无事实支撑 {stat['no_facts']})")
    if violations:
        top = sorted(violations.items(), key=lambda kv: -kv[1])[:8]
        print("  最常踩的禁用词:", "、".join(f"{w}×{n}" for w, n in top))
        print("  —— 这些词属于别的侧面, 出现说明生成端跑题了, 频次高就该收紧对应的 answer-spec")

    # ── 闸二: 六维 review(按图分组, 一次审多条)
    verdicts: dict[int, dict] = {}
    if not args.no_verify and kept:
        llm = LLM(args.model, args.base_url, cache_dir=args.cache_dir, temperature=0.0,
                  endpoints_file=args.endpoints, role="review")
        print(f"审稿端: {llm.describe()}")
        if args.workers <= 0:
            args.workers = llm.total_concurrency
        gen_models = {e.model for e in LLM(args.model, args.base_url,
                                           endpoints_file=args.endpoints,
                                           role="generate").pool}
        if {e.model for e in llm.pool} == gen_models:
            print("  [注意] 审稿与生成是同一个模型, 自己审自己会虚高 —— "
                  "配置里给 review 换一个模型")
        by_img: dict[str, list[int]] = {}
        for i, r in enumerate(kept):
            by_img.setdefault(r["image_id"], []).append(i)

        def run(item):
            _, idxs = item
            batch = [{"id": j, "question": kept[j]["question"], "answer": kept[j]["answer"],
                      "facet": kept[j]["facet"]} for j in idxs[:8]]
            prompt = review_tmpl.format(width=kept[idxs[0]]["width"],
                                        height=kept[idxs[0]]["height"],
                                        samples=json.dumps(batch, ensure_ascii=False, indent=1))
            msg = [{"role": "user", "content": image_message(
                prompt, kept[idxs[0]]["image_path"], args.inline_images)}]
            d = parse_json(llm.chat(msg)) or {}
            return {r["id"]: r for r in d.get("reviews", []) if "id" in r}

        with ThreadPoolExecutor(args.workers) as ex:
            for got in ex.map(run, by_img.items()):
                verdicts.update(got)

    # ── 落盘
    dims = ("correct", "grounded", "facet", "instruction", "needs_image", "no_overclaim")
    out: list[dict] = []
    n_grounded = 0
    dim_fail: dict[str, int] = {}
    for i, r in enumerate(kept):
        v = verdicts.get(i)
        if v is not None:
            low = [d for d in dims if v.get(d, 5) < args.min_score]
            if low:
                stat["fail"] += 1
                for d in low:
                    dim_fail[d] = dim_fail.get(d, 0) + 1
                continue
        stat["pass"] += 1
        field = r.get("media_field", "images")
        paths = r.get("media") or [r["image_path"]]
        answer = r["answer"]
        if r.get("facet") == "grounded" and r.get("region"):
            # 文字是模型写的, 坐标是规则算的 —— 在这里才合成一条"文字 + 图像区域"的答案。
            # 坐标从来不经过模型, 所以不存在框报偏的问题。
            n_grounded += 1
            answer = answer.rstrip() + "\n" + box_json(r["region"]["box_1000"],
                                                       r["region"]["label"])
        out.append(make_row(
            sample_id=f"{r['image_id']}_{r['facet']}_{i}",
            media_field=field, media=paths, system=sys_rng.choice(systems),
            turns=[(r["question"], answer)],
            metadata={"image_id": r["image_id"], "task_type": r["kind"], "facet": r["facet"],
                      "gen": "llm",
                      "modality": r.get("modality", "image"), "n_media": len(paths),
                      "anomaly": [r["anomaly"]],
                      "source_dataset": r["source_dataset"], "license": r["license"],
                      "image_width": r["width"], "image_height": r["height"],
                      "coordinate_mode": COORD_MODE, "bbox_scale": BOX_SCALE,
                      "view": r.get("view", "uav"),
                      # **永远是 dict**。原来通过时写 dict、跳过时写字符串
                      # "skipped", 规则侧干脆没这个键 —— 下游一句
                      # metadata["review"]["correct"] 就炸。
                      "review": (v if isinstance(v, dict)
                                 else {"status": "skipped"})}))

    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    img = [x for x in out if "videos" not in x]
    vid = [x for x in out if "videos" in x]
    base.write_text(json.dumps(img, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"落盘 {len(img)} 条(图像) -> {base}")
    if vid:
        vp = base.with_name(base.stem + "_video" + base.suffix)
        vp.write_text(json.dumps(vid, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"落盘 {len(vid)} 条(视频) -> {vp}")
    if verdicts:
        rate = stat["fail"] / max(1, stat["fail"] + stat["pass"])
        print(f"  review: 通过 {stat['pass']} / 打回 {stat['fail']} (打回率 {rate:.1%})")
        if dim_fail:
            print("  打回原因:", "、".join(f"{k} {v}" for k, v in
                                          sorted(dim_fail.items(), key=lambda kv: -kv[1])))
        if rate > 0.15:
            print("  [warn] 打回率超过 15%, 说明生成端在编造或跑题, "
                  "建议收紧 answer-spec 或把 temperature 降到 0.5 以下")
    from collections import Counter
    print("  侧面分布:", dict(Counter(x["metadata"]["facet"] for x in out).most_common()))
    if n_grounded:
        print(f"  其中带框描述(文字+图像区域) {n_grounded} 条, 坐标由规则给出")


def _write_requests(reqs: list[dict], out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[dry-run] 写出 {len(reqs)} 条请求 -> {out}（可交给 vLLM 离线批推理）")


def _run_and_write(reqs, fn, out: str, workers: int) -> None:
    """并发跑完写盘。**单条失败不中断整批** —— 十万条的活跑几个小时,
    不能因为中间一次 500 把前面的成果全丢了。失败的记 error 字段,
    verify 那一步会当空答案滤掉; 重跑时命中缓存, 成功的不重复计费。
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    def guarded(r):
        try:
            return fn(r)
        except Exception as e:                           # noqa: BLE001
            return {**{k: v for k, v in r.items() if k != "prompt"},
                    "answer": "", "error": f"{type(e).__name__}: {e}"}

    done = n_err = 0
    with Path(out).open("w", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for res in ex.map(guarded, reqs):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            n_err += bool(res.get("error"))
            if done % 50 == 0:
                print(f"  {done}/{len(reqs)}" + (f"  (失败 {n_err})" if n_err else ""),
                      flush=True)
    print(f"完成 {done} 条 -> {out}" + (f"  其中 {n_err} 条调用失败" if n_err else ""))
    if n_err and n_err > len(reqs) * 0.1:
        print(f"  [警告] 失败率 {n_err / len(reqs):.0%}, 先查服务再往下走")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 驱动的指令型 VQA 生成")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--ontology", default="configs/ontology.yaml")
        p.add_argument("--prompt-dir", default="configs/prompts")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--base-url", default=None,
                       help="OpenAI 兼容端点, 可指向本地 vLLM。逗号分隔可写多路轮转。"
                            "不填则读环境变量 VLM_BASE_URL")
        p.add_argument("--endpoints", default=None,
                       help="端点池 yaml(generate / review 两组)。填了它就不用 --base-url; "
                            "每一路各带自己的 model 与 key")
        p.add_argument("--cache-dir", default=".llm_cache", help="按请求哈希缓存, 重跑不重复计费")
        p.add_argument("--workers", type=int, default=8,
                       help="并发数。填 0 = 跟着端点池的总并发走")
        p.add_argument("--inline-images", action="store_true", help="图像转 base64 内联(远端 API 需要)")
        p.add_argument("--dry-run", action="store_true", help="只写请求, 不调用 API")
        p.add_argument("--exclude-ids", default=None,
                       help="golden set 的 image_id 清单, 必须排除否则泄漏(make_golden.py 产出)")

    p = sub.add_parser("ping", help="逐路体检端点池(开跑前先跑这个)")
    p.add_argument("--ontology", default="configs/ontology.yaml")
    p.add_argument("--prompt-dir", default="configs/prompts")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=None)
    p.add_argument("--endpoints", default=None,
                   help="端点池 yaml。不填就读 configs/generate.yaml")
    p.set_defaults(out="/dev/null", cache_dir=None, workers=1,
                   inline_images=False, dry_run=False, exclude_ids=None)

    p = sub.add_parser("screen", help="闸5 图像质检")
    p.add_argument("--scenes", required=True); common(p)

    p = sub.add_parser("generate", help="按侧面生成描述与推理")
    p.add_argument("--scenes", required=True)
    p.add_argument("--facets-per-image", type=int, default=2,
                   help="每张图抽几个描述侧面")
    p.add_argument("--reason-ratio", type=float, default=0.35,
                   help="有事件的图里出推理题的比例")
    p.add_argument("--max-objects", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.55,
                   help="描述求准不求奇, 0.5~0.6 即可; 多样性靠侧面与问法池, 不靠高温")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", type=int, default=0,
                   help="只跑 N 个 scene 的试水批。**按类别分层抽**, 不是取前 N 条 —— "
                        "scene 是按数据源排好序的, 取前 N 条只会拿到同一个数据集的图, "
                        "看不出别的类写成什么样")
    p.add_argument("--target-per-class", type=int, default=0,
                   help="每个异常类的目标条数(描述侧)。富余的按图下采样, "
                        "稀缺的自动提高每图侧面数, 仍不足则报缺口。0 表示不限")
    common(p)

    p = sub.add_parser("verify", help="must-not 硬过滤 + 六维 review + 落盘")
    p.add_argument("--generated", required=True)
    p.add_argument("--no-verify", action="store_true",
                   help="跳过 LLM review, 只做硬过滤(调试用, 正式跑不要用)")
    p.add_argument("--min-score", type=int, default=3, help="六维中任一维低于此值即打回")
    common(p)

    args = ap.parse_args()
    onto = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    globals()["BOX_SCALE"] = int(onto.get("box_scale", BOX_SCALE))
    {"ping": cmd_ping, "screen": cmd_screen,
     "generate": cmd_generate, "verify": cmd_verify}[args.cmd](args, onto)


if __name__ == "__main__":
    main()

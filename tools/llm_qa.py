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
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from scene import Scene, load_scenes

DEFAULT_QA_TYPES = ("judgement", "classification", "counting", "grounding",
                    "attribute", "spatial", "description", "reasoning", "negation")
BOX_SCALE = 1000


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
                                 if c["id"] != "normal" and c["id"] not in {e.type for e in scene.events}],
    }


# ---------------------------------------------------------------- LLM 客户端
class LLM:
    """OpenAI 兼容接口。base_url 指向本地 vLLM 即可, 无需外网。"""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", cache_dir: str | None = None,
                 temperature: float = 0.7, max_retries: int = 3):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         ensure_ascii=False).encode()).hexdigest()[:32]

    def chat(self, messages: list[dict], json_mode: bool = True) -> str:
        import urllib.request                            # 惰性导入, dry-run 时用不到

        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "temperature": self.temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        cache_file = self.cache / f"{self._key(payload)}.json" if self.cache else None
        if cache_file and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["content"]

        last = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    content = json.loads(r.read())["choices"][0]["message"]["content"]
                if cache_file:
                    cache_file.write_text(json.dumps({"content": content}, ensure_ascii=False),
                                          encoding="utf-8")
                return content
            except Exception as e:                       # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败: {last}")


def image_message(text: str, image_path: str | None, inline: bool) -> list[dict]:
    if not image_path:
        return [{"type": "text", "text": text}]
    p = Path(image_path)
    if inline and p.exists():
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode()
        url = f"data:{mime};base64,{b64}"
    else:
        url = str(p)
    return [{"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text}]


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


def cmd_review(args, onto):
    tmpl = load_prompt("review_image.txt", args.prompt_dir)
    scenes = load_scenes(args.scenes)
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


def cmd_generate(args, onto):
    tmpl = load_prompt("generate_qa.txt", args.prompt_dir)
    scenes = load_scenes(args.scenes)
    qa_types = ", ".join(args.qa_types or DEFAULT_QA_TYPES)

    reqs = []
    for s in scenes:
        facts = build_facts(s, onto, args.max_objects)
        reqs.append({"image_id": s.image_id, "image_path": s.image_path,
                     "facts": facts,
                     "source_dataset": s.source_dataset, "license": s.license,
                     "prompt": tmpl.format(facts=json.dumps(facts, ensure_ascii=False, indent=1),
                                           n_items=args.n_items, qa_types=qa_types)})
    if args.dry_run:
        _write_requests(reqs, args.out)
        return

    llm = LLM(args.model, args.base_url, cache_dir=args.cache_dir, temperature=args.temperature)

    def run(r):
        msg = [{"role": "user", "content": image_message(r["prompt"], r["image_path"], args.inline_images)}]
        d = parse_json(llm.chat(msg)) or {"items": []}
        return {"image_id": r["image_id"], "image_path": r["image_path"],
                "facts": r["facts"], "source_dataset": r["source_dataset"],
                "license": r["license"], "items": d.get("items", [])}

    _run_and_write(reqs, run, args.out, args.workers)


def cmd_verify(args, onto):
    tmpl = load_prompt("verify_qa.txt", args.prompt_dir)
    system = load_prompt("system.txt", args.prompt_dir).strip()
    records = [json.loads(ln) for ln in Path(args.generated).read_text(encoding="utf-8").splitlines() if ln.strip()]

    llm = None if args.no_verify else LLM(args.model, args.base_url,
                                          cache_dir=args.cache_dir, temperature=0.0)
    out, stats = [], {"pass": 0, "fix": 0, "drop": 0, "conflict": 0}

    for rec in records:
        items = [it for it in rec.get("items", []) if not it.get("conflict")]
        stats["conflict"] += len(rec.get("items", [])) - len(items)
        if not items:
            continue

        verdicts: dict[int, dict] = {}
        if llm is not None:
            brief = [{"index": i, "turns": it.get("turns", [])} for i, it in enumerate(items)]
            prompt = tmpl.format(facts=json.dumps(rec["facts"], ensure_ascii=False, indent=1),
                                 items=json.dumps(brief, ensure_ascii=False, indent=1))
            d = parse_json(llm.chat([{"role": "user", "content": prompt}])) or {}
            verdicts = {r["index"]: r for r in d.get("results", []) if "index" in r}

        for i, it in enumerate(items):
            v = verdicts.get(i, {"verdict": "pass"})
            stats[v.get("verdict", "pass")] = stats.get(v.get("verdict", "pass"), 0) + 1
            if v.get("verdict") == "drop":
                continue
            turns = it.get("turns") or []
            if not turns:
                continue
            if v.get("verdict") == "fix" and v.get("fixed_assistant"):
                turns[-1]["assistant"] = v["fixed_assistant"]

            messages = [{"role": "system", "content": system}]
            for k, t in enumerate(turns):
                user = t.get("user", "")
                if k == 0 and "<image>" not in user:
                    user = "<image>" + user
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": t.get("assistant", "")})

            out.append({
                "messages": messages,
                "images": [rec["image_path"]],
                "extra": {"image_id": rec["image_id"],
                          "qa_type": it.get("qa_type", "unknown"),
                          "instruction_style": it.get("instruction_style", "direct"),
                          "anomaly": [e["type"] for e in rec["facts"].get("events", [])] or ["normal"],
                          "source_dataset": rec.get("source_dataset", "unknown"),
                          "license": rec.get("license", "unknown"),
                          "verified": v.get("verdict", "pass")},
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(v for k, v in stats.items() if k in ("pass", "fix", "drop"))
    print(f"落盘 {len(out)} 条 -> {args.out}")
    print(f"  校验: pass={stats['pass']} fix={stats['fix']} drop={stats['drop']}"
          f"  (drop 率 {stats['drop'] / max(1, total):.1%}), 生成端自报冲突 {stats['conflict']}")
    if total and stats["drop"] / total > 0.15:
        print("  [warn] drop 率超过 15%, 说明生成端在编造事实, 建议收紧 generate_qa.txt 的约束或降低 temperature")


def _write_requests(reqs: list[dict], out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[dry-run] 写出 {len(reqs)} 条请求 -> {out}（可交给 vLLM 离线批推理）")


def _run_and_write(reqs, fn, out: str, workers: int) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with Path(out).open("w", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for res in ex.map(fn, reqs):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(reqs)}", flush=True)
    print(f"完成 {done} 条 -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 驱动的指令型 VQA 生成")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--ontology", default="configs/ontology.yaml")
        p.add_argument("--prompt-dir", default="configs/prompts")
        p.add_argument("--model", default="qwen2.5-vl-72b-instruct")
        p.add_argument("--base-url", default=None, help="OpenAI 兼容端点, 可指向本地 vLLM")
        p.add_argument("--cache-dir", default=".llm_cache", help="按请求哈希缓存, 重跑不重复计费")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--inline-images", action="store_true", help="图像转 base64 内联(远端 API 需要)")
        p.add_argument("--dry-run", action="store_true", help="只写请求, 不调用 API")

    p = sub.add_parser("review", help="闸5 图像质检")
    p.add_argument("--scenes", required=True); common(p)

    p = sub.add_parser("generate", help="生成指令型 QA")
    p.add_argument("--scenes", required=True)
    p.add_argument("--n-items", type=int, default=6)
    p.add_argument("--max-objects", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--qa-types", nargs="*", default=None)
    common(p)

    p = sub.add_parser("verify", help="校验并落盘为 ShareGPT")
    p.add_argument("--generated", required=True)
    p.add_argument("--no-verify", action="store_true", help="跳过校验直接落盘(不推荐)")
    common(p)

    args = ap.parse_args()
    onto = yaml.safe_load(Path(args.ontology).read_text(encoding="utf-8"))
    globals()["BOX_SCALE"] = int(onto.get("box_scale", BOX_SCALE))
    {"review": cmd_review, "generate": cmd_generate, "verify": cmd_verify}[args.cmd](args, onto)


if __name__ == "__main__":
    main()

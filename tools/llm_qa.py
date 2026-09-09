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
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds.boxes import BBOX_SCALE, COORD_MODE
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
    "hard_neg": 40, "scan": 30,
}
SKIP_FACETS = {"position"}
BOX_SCALE = BBOX_SCALE


def facet_applicable(kind: str, s: Scene) -> bool:
    """needs 前置条件的机器可判部分。判不了的交给模型自己返回空。"""
    ev = {e.type: e for e in s.events}
    cluster = next((e for e in s.events if "cluster_bbox" in e.evidence), None)
    n_frames = len(s.meta.get("frames", [])) or 1
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
        "hard_neg": bool(s.meta.get("hard_negative")),
        "scan": not s.events,
        "evidence": bool(s.events),
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


def cmd_generate(args, onto):
    tmpl = load_prompt("describe_gen.txt", args.prompt_dir + "/_tools")
    reason_tmpl = load_prompt("reason_gen.txt", args.prompt_dir + "/_tools")
    facets, _ = load_all(args.prompt_dir)
    zh = {c["id"]: c["zh"] for c in onto["classes"]}
    scenes = load_scenes_excluding(args.scenes, args.exclude_ids)
    rng = random.Random(args.seed)

    reqs: list[dict] = []
    n_skip = 0
    for s in scenes:
        anomaly = (s.anomaly_types or ["normal"])[0]
        facts = build_facts(s, onto, args.max_objects)
        picked = _pick_facets(s, facets, anomaly, rng, args.facets_per_image)
        if not picked:
            n_skip += 1
        for fa in picked:
            q = rng.choice(fa.q_bank).replace("{zh}", zh.get(anomaly, "异常"))
            bans = fa.bans_for(anomaly)
            facts_str = json.dumps(trim_facts(facts, fa.kind), ensure_ascii=False, indent=1)
            reqs.append({
                "image_id": s.image_id, "image_path": s.image_path,
                "images": s.meta.get("frames") or [s.image_path],
                "kind": "describe", "facet": fa.kind, "anomaly": anomaly,
                "question": q, "must_not": bans, "facts": facts,
                "source_dataset": s.source_dataset, "license": s.license,
                "width": s.width, "height": s.height,
                "prompt": tmpl.format(
                    facts=facts_str, kind=fa.kind,
                    kind_zh=fa.meta.get("zh", fa.kind),
                    answer_spec=fa.answer_spec,
                    q_example=fa.q_example or (fa.q_bank[0] if fa.q_bank else q),
                    a_example=fa.example_for(anomaly),
                    question=q, must_not="、".join(bans) or "（无）"),
            })
        # 推理题: 有事件或困难负样本的图才出
        if (s.events or s.meta.get("hard_negative")) and rng.random() < args.reason_ratio:
            _, asks = load_all(args.prompt_dir)
            rq = rng.choice(asks["reason"].lines).replace("{zh}", zh.get(anomaly, "异常"))
            reqs.append({
                "image_id": s.image_id, "image_path": s.image_path,
                "images": s.meta.get("frames") or [s.image_path],
                "kind": "reason", "facet": "reason", "anomaly": anomaly,
                "question": rq, "must_not": [], "facts": facts,
                "source_dataset": s.source_dataset, "license": s.license,
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

    llm = LLM(args.model, args.base_url, cache_dir=args.cache_dir, temperature=args.temperature)

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
    system = (Path(args.prompt_dir) / "system.txt").read_text(encoding="utf-8").strip()
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
        llm = LLM(args.model, args.base_url, cache_dir=args.cache_dir, temperature=0.0)
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
        n_img = len(r["images"])
        out.append({
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "<image>" * n_img + r["question"]},
                         {"role": "assistant", "content": r["answer"]}],
            "images": r["images"],
            "extra": {"image_id": r["image_id"], "task": r["kind"], "facet": r["facet"],
                      "gen": "llm", "n_turns": 1, "n_images": n_img,
                      "anomaly": [r["anomaly"]],
                      "source_dataset": r["source_dataset"], "license": r["license"],
                      "image_width": r["width"], "image_height": r["height"],
                      "coordinate_mode": COORD_MODE, "bbox_scale": BOX_SCALE,
                      "review": v or "skipped"},
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"落盘 {len(out)} 条 -> {args.out}")
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
    print("  侧面分布:", dict(Counter(s["extra"]["facet"] for s in out).most_common()))


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
        p.add_argument("--exclude-ids", default=None,
                       help="golden set 的 image_id 清单, 必须排除否则泄漏(make_golden.py 产出)")

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
    {"screen": cmd_screen, "generate": cmd_generate, "verify": cmd_verify}[args.cmd](args, onto)


if __name__ == "__main__":
    main()

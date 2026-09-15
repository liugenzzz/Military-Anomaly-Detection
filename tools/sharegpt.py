"""ShareGPT 落盘格式。**与 target_detection_vl_dataset/qwen3vl_sft_builder 对齐**。

一条样本长这样:

    {"id": "...",
     "images": ["..."],                       # 或 "videos"
     "system": "...",                         # 单列一个字段, 不混进 conversations
     "conversations": [{"from": "human", "value": "<image>\\n问"},
                       {"from": "gpt",   "value": "答"}],
     "metadata": {"task_type": ..., "n_turns": ..., ...}}

几个约定是照抄那个项目的, 不要随手改:
  - `conversations` 而不是 `messages`; `from`/`value` 而不是 `role`/`content`;
    human/gpt 而不是 user/assistant。LLaMA-Factory 两种都认, 但两个项目产出的
    数据将来大概率要混着训, 格式不一致会踩坑。
  - 媒体占位符后面**带一个换行**(`<image>\\n`), 且只出现在首轮。
  - system 不塞进 conversations, 单列一个字段 —— 那个项目干脆没有 system,
    我们这边需要它来框定"情报标注"的角色, 所以放在 dataset_info 的 system 列里。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

HUMAN, GPT = "human", "gpt"


def _rel(paths: list[str]) -> list[str]:
    """按配置把绝对路径改写成相对 image_folder 的路径。

    训练机和生成机的挂载点往往不一样, 数据里写死绝对路径, 换台机器就得全文替换。
    LLaMA-Factory 支持 image_folder(或 media_dir), 相对路径更好搬。
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from config import CFG
        root = CFG.get("output", "image_folder", default="") or ""
        if not root or not CFG.get("output", "relative_paths", default=True):
            return paths
        out = []
        for p in paths:
            try:
                out.append(str(Path(p).resolve().relative_to(Path(root).resolve())))
            except ValueError:
                out.append(p)                  # 不在 image_folder 底下, 保留绝对路径
        return out
    except Exception:                          # noqa: BLE001
        return paths


def make_row(*, sample_id: str, media_field: str, media: list[str], system: str,
             turns: list[tuple[str, str]], metadata: dict[str, Any]) -> dict[str, Any]:
    """turns 是 [(问, 答), ...]; 媒体占位符自动加在首轮问句前面。"""
    tok = "<video>" if media_field == "videos" else "<image>"
    conv: list[dict[str, str]] = []
    for i, (q, a) in enumerate(turns):
        conv.append({"from": HUMAN,
                     "value": (tok * len(media) + "\n" + q) if i == 0 else q})
        conv.append({"from": GPT, "value": a})
    row = {"id": sample_id, media_field: _rel(media), "conversations": conv,
           "metadata": {**metadata, "n_turns": len(turns)}}
    if system:
        row["system"] = system
    return row


def turns_of(row: dict[str, Any]) -> list[dict[str, str]]:
    """兼容读取: 新的 conversations, 以及早期产出的 messages。"""
    if "conversations" in row:
        return [{"role": HUMAN if t.get("from") == HUMAN else GPT,
                 "content": t.get("value", "")} for t in row["conversations"]]
    return [{"role": m["role"], "content": m["content"]}
            for m in row.get("messages", []) if m.get("role") != "system"]


def meta_of(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") or row.get("extra") or {}


def media_of(row: dict[str, Any]) -> tuple[str, list[str]]:
    field = "videos" if "videos" in row else "images"
    return field, row.get(field, [])

"""merge_scenes.py 的对照样本 —— 复刻手敲 cat 时踩过的两种事故。

  1. 某个文件末尾缺行尾换行, cat 把两条记录粘成一行 —— 那两条静默消失;
  2. 通配符把不是 scene 的文件(LLM 请求体)一起卷进来。
外加重复 image_id 与坏 JSON 行。

    python tests/merge_fixture.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from merge_scenes import pick_files  # noqa: E402

TMP = Path(__file__).resolve().parent / "_merge_tmp"


def scene(i: str, ds: str) -> dict:
    return {"image_id": i, "image_path": f"/x/{i}.jpg", "width": 1024, "height": 1024,
            "source_dataset": ds, "objects": [], "events": []}


def build() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    # a: 末尾故意不带换行
    (TMP / "a.jsonl").write_text(
        "\n".join(json.dumps(scene(f"A_{i}", "DOTA")) for i in range(3)), encoding="utf-8")
    (TMP / "b.jsonl").write_text(
        "\n".join(json.dumps(scene(f"B_{i}", "MAR20")) for i in range(3)) + "\n",
        encoding="utf-8")
    # c: 一条重复 id + 一行坏 JSON
    (TMP / "c.jsonl").write_text("\n".join([
        json.dumps(scene("A_0", "DOTA")), json.dumps(scene("C_9", "FASDD")),
        '{"image_id": "坏行", 不是JSON}']) + "\n", encoding="utf-8")
    # 这两个必须被认出来不是 scene: 它们同样有 image_id 和 image_path
    (TMP / "gen_req.jsonl").write_text(
        json.dumps({"image_id": "A_0", "image_path": "/x/A_0.jpg", "prompt": "写描述"}) + "\n",
        encoding="utf-8")
    (TMP / "a_ev.jsonl").write_text(
        json.dumps(scene("A_0", "DOTA")) + "\n", encoding="utf-8")


def main() -> int:
    build()
    take, skip = pick_files([str(TMP)], keep_ev=False, keep_demo=False)
    names = {p.name for p in take}
    skipped = {p.name for p, _ in skip}

    fail = []
    if names != {"a.jsonl", "b.jsonl", "c.jsonl"}:
        fail.append(f"该合的文件不对: {sorted(names)}")
    if "gen_req.jsonl" not in skipped:
        fail.append("LLM 请求体没被认出来 —— 它也有 image_id/image_path, 只能靠内容判")
    if "a_ev.jsonl" not in skipped:
        fail.append("derive 的产物 *_ev.jsonl 没被跳过, 会和源文件重复")

    # cat 的对照: 缺行尾换行会粘掉两条
    glued = sum(1 for ln in "".join((TMP / f).read_text(encoding="utf-8")
                                    for f in ("a.jsonl", "b.jsonl")).splitlines()
                if _bad(ln))
    if glued != 1:
        fail.append(f"对照样本没造出粘行(应有 1 行粘坏, 实得 {glued}) —— 测试本身失效了")

    for n in sorted(names):
        print(f"  ✓ 合并 {n}")
    for n in sorted(skipped):
        print(f"  ✓ 跳过 {n}")
    print(f"  ✓ cat 会把 a/b 粘坏 {glued} 行, 静默丢掉 2 条记录")
    shutil.rmtree(TMP, ignore_errors=True)
    if fail:
        print("\n" + "\n".join("  ✗ " + f for f in fail))
        return 1
    print("\n合并器对 4 种事故全部处理正确。")
    return 0


def _bad(ln: str) -> bool:
    try:
        json.loads(ln)
        return False
    except json.JSONDecodeError:
        return True


if __name__ == "__main__":
    raise SystemExit(main())

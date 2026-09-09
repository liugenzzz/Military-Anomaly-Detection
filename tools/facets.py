"""描述侧面与问法池的加载、校验。

prompt 全部是纯 txt，与代码分离 —— 改提示词不用动代码、不用重装依赖，
服务器上直接改文件即可（这条约定取自参考项目）。

文件格式：
    # 普通注释, 加载时丢弃
    #! kind: formation            元数据, 键值对
    #! must-not: 颜色 烟 火        空格分隔的列表
    answer-spec: ...             多行, 续行缩进
    q-example: ...
    a-example: ...
    q-bank:                      其后每个非空行是一条问法
      问法一
      问法二

自检(python tools/facets.py --check)会检查:
  - q-bank 里的问法有没有踩到该任务的 forbid 词
  - a-example 有没有违反自己的 must-not(样例都违规, 生成出来只会更糟)
  - 占位符是否闭合、是否有重复问法、每个异常类的侧面是否齐全
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_DIR = Path("configs/prompts")

# 跨类互斥词: 写某一类异常的描述时, 不该出现另一类的专有词汇。
# 防止四类描述串味 —— 参考项目没有这一条, 是本项目按 4 类异常的特点补的。
CROSS_CLASS_BAN: dict[str, list[str]] = {
    "massing": ["烟雾", "烟柱", "火光", "火焰", "爆炸", "爆燃", "越界", "禁区"],
    "explosion": ["集结", "列队", "阵列", "越界", "禁区", "编队"],
    "smoke": ["爆炸", "爆燃", "火球", "集结", "列队", "阵列", "越界", "禁区"],
    "border_crossing": ["集结", "列队", "阵列", "烟雾", "烟柱", "火光", "爆炸"],
    "normal": [],
}


@dataclass
class Facet:
    kind: str
    anomaly: list[str]
    must_not: list[str]
    answer_spec: str
    a_example: str = ""
    q_example: str = ""
    a_by_anomaly: dict[str, str] = field(default_factory=dict)
    q_bank: list[str] = field(default_factory=list)
    needs: str = ""
    meta: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    def example_for(self, anomaly: str) -> str:
        """取该异常类专用的样例, 没有就用默认的。"""
        return self.a_by_anomaly.get(anomaly, self.a_example)

    def bans_for(self, anomaly: str) -> list[str]:
        """该侧面用在某个异常类上时的完整禁用词 = 自身 must-not + 跨类互斥。"""
        return list(dict.fromkeys(self.must_not + CROSS_CLASS_BAN.get(anomaly, [])))

    def violates(self, text: str, anomaly: str) -> list[str]:
        return [w for w in self.bans_for(anomaly) if w and w in text]


@dataclass
class AskBank:
    task: str
    lines: list[str]
    forbid: list[str] = field(default_factory=list)
    max_len: int = 0
    meta: dict[str, str] = field(default_factory=dict)
    path: Path | None = None


# ---------------------------------------------------------------- 解析
_META = re.compile(r"^#!\s*([\w-]+)\s*:\s*(.*)$")
# 共有侧面(evidence/position/full)被四类异常复用, 样例必须分类给 ——
# 用聚集主题的样例去写爆炸的描述, 会踩到跨类互斥词。
_FIELD = re.compile(r"^(answer-spec|q-example|a-example|q-bank)(?:\.([a-z_]+))?\s*:\s*(.*)$")


def _parse(path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """返回 (元数据, 字段, 裸行列表)。裸行用于 ask 池。"""
    meta: dict[str, str] = {}
    fields: dict[str, list[str]] = {}
    bare: list[str] = []
    cur: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        m = _META.match(raw)
        if m:
            meta[m.group(1)] = m.group(2).strip()
            cur = None
            continue
        if raw.lstrip().startswith("#"):
            continue
        m = _FIELD.match(raw)
        if m:
            cur = m.group(1) + (f".{m.group(2)}" if m.group(2) else "")
            fields[cur] = [m.group(3).strip()] if m.group(3).strip() else []
            continue
        if cur and raw.startswith((" ", "\t")):
            fields[cur].append(raw.strip())
            continue
        if cur == "q-bank":
            fields[cur].append(raw.strip())
            continue
        cur = None
        bare.append(raw.strip())

    joined = {k: (" ".join(v) if not k.startswith("q-bank") else v)
              for k, v in fields.items()}
    return meta, joined, bare


def load_facet(path: Path) -> Facet:
    meta, f, _ = _parse(path)
    if "kind" not in meta:
        raise ValueError(f"{path}: 缺少 #! kind")
    return Facet(
        kind=meta["kind"],
        anomaly=meta.get("anomaly", "").split() or ["*"],
        must_not=meta.get("must-not", "").split(),
        needs=meta.get("needs", ""),
        answer_spec=f.get("answer-spec", ""),
        q_example=f.get("q-example", ""),
        a_example=f.get("a-example", ""),
        a_by_anomaly={k.split(".", 1)[1]: v for k, v in f.items()
                      if k.startswith("a-example.")},
        q_bank=list(f.get("q-bank", [])),
        meta=meta, path=path,
    )


def load_ask(path: Path) -> AskBank:
    meta, _, bare = _parse(path)
    return AskBank(task=meta.get("task", path.stem), lines=bare,
                   forbid=meta.get("forbid", "").split(),
                   max_len=int(meta.get("max-len", 0) or 0),
                   meta=meta, path=path)


def load_all(prompt_dir: str | Path = PROMPT_DIR) -> tuple[dict[str, list[Facet]], dict[str, AskBank]]:
    """返回 ({异常类: [该类可用的侧面]}, {任务名: 问法池})。"""
    d = Path(prompt_dir)
    facets: dict[str, list[Facet]] = {}
    for p in sorted(d.glob("describe/**/*.txt")):
        fa = load_facet(p)
        for a in fa.anomaly:
            facets.setdefault(a, []).append(fa)
    asks = {b.task: b for b in (load_ask(p) for p in sorted(d.glob("ask/*.txt")))}
    return facets, asks


def load_tool(name: str, prompt_dir: str | Path = PROMPT_DIR) -> str:
    return (Path(prompt_dir) / "_tools" / f"{name}.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 自检
EXPECTED = {
    "massing": {"evidence", "position", "full", "formation", "composition", "scale", "site"},
    "explosion": {"evidence", "position", "full", "intensity", "debris", "extent", "stage"},
    "smoke": {"evidence", "position", "full", "morphology", "color", "drift", "occlusion"},
    "border_crossing": {"evidence", "position", "full", "trajectory", "timing",
                        "boundary_relation", "group"},
    "normal": {"position", "full", "hard_neg", "scan"},
}


def check(prompt_dir: str | Path = PROMPT_DIR) -> list[str]:
    facets, asks = load_all(prompt_dir)
    errs: list[str] = []

    # 1. 每个异常类的侧面是否齐全
    for anomaly, want in EXPECTED.items():
        got = {f.kind for f in facets.get(anomaly, [])}
        if miss := want - got:
            errs.append(f"[{anomaly}] 缺少侧面: {sorted(miss)}")
        if extra := got - want:
            errs.append(f"[{anomaly}] 多出未登记的侧面: {sorted(extra)}")

    seen_files: set[Path] = set()
    for anomaly, fs in facets.items():
        if anomaly == "*":
            continue
        for fa in fs:
            tag = f"{anomaly}/{fa.kind}"
            # 2. a-example 不得违反自己的 must-not —— 样例都违规, 生成只会更糟
            ex = fa.example_for(anomaly)
            if not ex:
                errs.append(f"[{tag}] 没有可用的 a-example")
            elif bad := fa.violates(ex, anomaly):
                errs.append(f"[{tag}] a-example 踩到禁用词 {bad} —— "
                            f"共有侧面请用 a-example.{anomaly} 单独给样例")
            # 3. answer-spec 不能为空
            if not fa.answer_spec:
                errs.append(f"[{tag}] answer-spec 为空")
            # 4. q-bank 至少 6 条, 且不重复
            if len(fa.q_bank) < 6:
                errs.append(f"[{tag}] q-bank 只有 {len(fa.q_bank)} 条, 少于 6 条")
            if len(fa.q_bank) != len(set(fa.q_bank)):
                errs.append(f"[{tag}] q-bank 有重复问法")
            # 5. 描述类问法不得要坐标
            for q in fa.q_bank:
                if any(w in q for w in ("坐标", "框出", "边界框", "bbox")):
                    errs.append(f"[{tag}] q-bank 出现要坐标的问法: {q}")
            if fa.path:
                seen_files.add(fa.path)

    # 6. 问法池自检
    for name, bank in asks.items():
        if len(bank.lines) < 6:
            errs.append(f"[ask/{name}] 只有 {len(bank.lines)} 条问法, 少于 6 条")
        if len(bank.lines) != len(set(bank.lines)):
            errs.append(f"[ask/{name}] 有重复问法")
        for q in bank.lines:
            if bad := [w for w in bank.forbid if w in q]:
                errs.append(f"[ask/{name}] 问法踩到 forbid {bad}: {q}")
            plain = re.sub(r"\{[a-z_]+\}", "", q)
            if bank.max_len and len(plain) > bank.max_len:
                errs.append(f"[ask/{name}] 问法超长({len(plain)}>{bank.max_len}): {q}")
            if q.count("{") != q.count("}"):
                errs.append(f"[ask/{name}] 占位符不闭合: {q}")

    # 7. 工具类 prompt 齐全
    for t in ("describe_gen", "reason_gen", "review", "screen_image"):
        if not (Path(prompt_dir) / "_tools" / f"{t}.txt").exists():
            errs.append(f"[_tools] 缺少 {t}.txt")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="侧面与问法池的加载与自检")
    ap.add_argument("--prompt-dir", default=str(PROMPT_DIR))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    facets, asks = load_all(args.prompt_dir)
    if args.list or not args.check:
        for anomaly in sorted(facets):
            if anomaly == "*":
                continue
            ks = sorted(f.kind for f in facets[anomaly])
            print(f"{anomaly:18s} {len(ks)} 个侧面: {', '.join(ks)}")
        print(f"\n问法池 {len(asks)} 个: {', '.join(sorted(asks))}")
        for n, b in sorted(asks.items()):
            print(f"  {n:16s} {len(b.lines):3d} 条问法  forbid={b.forbid or '-'}")
    if args.check:
        errs = check(args.prompt_dir)
        print("\n自检:", "通过 ✓" if not errs else f"发现 {len(errs)} 个问题")
        for e in errs:
            print("  ✗", e)
        raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    main()

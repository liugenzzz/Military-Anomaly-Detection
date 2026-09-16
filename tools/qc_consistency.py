"""语义一致性体检 —— 抓 check_dataset.py 抓不到的那一类问题。

check_dataset 看的是分布和格式: 路径通不通、坐标越不越界、题型比例、正负样本比。
这些过了, 数据照样可能是坏的 —— 坏在**条与条之间自相矛盾**:

  · 同一张图, 一条答"共 26 人", 另一条答"二三十人"        -> 模型学到数字可以随便报
  · 答案说"位于画面右上", 框却在左下                      -> 方位词与坐标解耦
  · 两张不相干的图, 出现一模一样的框                       -> 模板复制, 框根本没看图
  · 三个 image_id 共用同一句答案                           -> 同质化, 等于只有一条样本
  · 标注为正常/难负样本的图, 答案里却在断言异常             -> 自相矛盾, 最伤判定能力
  · label 里漏出英文类名 crowd_gathering区域               -> 本体 id 没翻译就落盘

这些错单看一条都挑不出毛病, 必须跨条比对。人工抽检能发现几处, 但十万条里
一条条看是不现实的 —— 所以做成脚本, 每次产出之后跑一遍。

用法:
    python tools/qc_consistency.py data/vqa/all.json
    python tools/qc_consistency.py data/vqa/*.json --out data/qc_flagged.jsonl
    python tools/qc_consistency.py data/vqa/all.json --fail-over 0.02   # 超 2% 退出码 1

--out 写出被标记的条目(带 qc 原因), 可以直接喂给下游过滤掉。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sharegpt import GPT, meta_of, media_of, turns_of  # noqa: E402
from build_vqa import CLS_ZH  # noqa: E402  主语词表与规则侧共用一份

BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
BOX_OBJ_RE = re.compile(r'\{[^{}]*"bbox_2d"[^{}]*\}')
LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]*)"')

# --- 数量抽取。这里的讲究全是被误报逼出来的 -------------------------------
# 早先只写了 (\d+)\s*(单位), 结果三分之二的告警是假的:
#   · compare 题「左半区0个目标，右半区30个」—— 分区对比, 不是矛盾
#   · correct 题「数量是2辆，不是3辆」        —— 3 是被否定掉的错数
# 所以只认两种"全图总数": 带总数前缀的, 和计数问句的直接回答;
# 且前面挂着否定词的一律不算。compare 这类分区题直接整条跳过。
TOTAL_RE = re.compile(r'(?:共|总共|一共|合计|数量是|数量为)\s*(?:观察到|检测到|有)?\s*(\d+)\s*(辆|人|名|架|艘|处|团|股|条|个)')
UNIT_RE = re.compile(r'(\d+)\s*(辆|人|名|架|艘|处|团|股|条)')
# 计数问句。纠错题问的是「画面中有158名人员——这句话准确吗」, 不含"多少/清点",
# 早先匹配不到, 于是那句准数没被当成全图总数, 矛盾就漏过去了。
COUNT_Q_RE = re.compile(r'(多少|几辆|几人|几名|几架|几艘|几处|清点|数量|计数|总数|'
                        r'准确吗|对不对|是否属实|这句话|错在哪|规模)')
NEG_BEFORE_RE = re.compile(r'(不是|并非|而非|不足|没有|少于|多于|超过)$')
REGION_TASKS = ("compare", "dense_region", "after_count")
# 概数说法, 必须带单位才算 —— 光一个"若干"不知道说的是人还是车,
# 拿它去和"3辆"比会把不相干的两句判成打架。**「数百名」里的「数」和「名」中间隔着「百」** —— 早先写成
# (数|多|少)\s*(单位) 就漏掉了这一整类, 于是同一张图上「数百名以上, 无法逐个
# 点清」和「确为158名人员」并存, QC 一声没吭。数量词部分要允许中间夹位数。
VAGUE_UNIT_RE = re.compile(
    r'(数十|十余|二三十|三四十|四五十|几十|十几|若干|数百|上百|成百|数千|上千|'
    r'成千|数万|大量|不少|许多|众多|数|多|少)[百千万余多]{0,2}\s*(辆|人|名|架|艘|处|团|股|条)')
UNIT_ALIAS = {"名": "人"}                      # 26 名 == 26 人, 同一口径

# 计数必须绑到**主语**, 光看量词会把「3辆车」和「0辆坦克」判成打架 ——
# 量词都是"辆", 问的根本不是一回事。主语词表跟规则侧共用 CLS_ZH, 再补几个
# 上位词; 长词优先匹配, 否则"车辆"会先被"车"吃掉。
SUBJECT_ALIAS = {"人员": "人", "士兵": "人", "人群": "人",
                 "大型车辆": "车辆", "军用车辆": "车辆", "机群": "飞机",
                 "军用飞机": "飞机", "民航飞机": "飞机", "军舰": "船只"}
SUBJECTS = sorted(set(CLS_ZH.values()) | set(SUBJECT_ALIAS)
                  | {"目标", "车辆", "人", "飞机", "船只", "烟雾", "火焰"},
                  key=len, reverse=True)


def subject_of(*texts: str) -> str | None:
    """问句(或答句)问的是什么。认不出来就返回 None —— 宁可不比, 不瞎比。"""
    for t in texts:
        for w in SUBJECTS:
            if w in t:
                return SUBJECT_ALIAS.get(w, w)
    return None


# 断言异常的措辞。正常图/难负样本里出现即为自相矛盾。
# **必须按句子判, 并且看否定词。** 光匹配一个"车队"会把
# 「否，未观察到车队机动的迹象。」也判成断言 —— 那恰恰是正确答案。
ASSERT_ANOMALY_RE = re.compile(r'(有异常|存在异常|发现异常|属于异常|异常聚集|正在聚集|'
                               r'发生爆炸|出现烟雾|车队|灾害|越界)')
# **否定词要就近看, 不能按整句找。** 规则侧的标准否定答案长这样:
#   「逐项核查：无军事装备集结、无人员异常聚集、无烟火、无车队机动」
# 每个否定词只管紧跟它的那一项, 按整句搜"没有/未见"根本抓不到这个「无」。
# 窗口取 6 个字: 够放下"并未观察到"这种长否定, 又不会跨到上一个顿号之外。
NEG_NEAR_RE = re.compile(r'[无未没不非否][^，,、。；;]{0,5}$')
# 这些词里的「不 / 无」是连词或副词, 不是否定。漏掉它们会把
# 「未见异常。**不过**右上角出现烟雾。」判成没问题 —— 而那正是该抓的自相矛盾。
NOT_NEG_RE = re.compile(r'不过|不但|不仅|不只|不管|不论|无论|无非|不外乎|不由得')
SENT_SPLIT_RE = re.compile(r'[。；;!?\n]+')
# 中文 label 里混进 ASCII 标识符 —— 本体 id 没翻译就落盘了
RAW_ID_RE = re.compile(r'[a-z][a-z0-9]*_[a-z0-9_]+')

# 方位词 -> (横向, 纵向)。None = 该轴不约束
DIRECTIONS: dict[str, tuple[str | None, str | None]] = {
    "左上": ("左", "上"), "右上": ("右", "上"), "左下": ("左", "下"), "右下": ("右", "下"),
    "左上角": ("左", "上"), "右上角": ("右", "上"),
    "左下角": ("左", "下"), "右下角": ("右", "下"),
    "正上方": (None, "上"), "正下方": (None, "下"),
    "上方": (None, "上"), "下方": (None, "下"), "上缘": (None, "上"), "下缘": (None, "下"),
    "顶部": (None, "上"), "底部": (None, "下"),
    "左侧": ("左", None), "右侧": ("右", None), "左缘": ("左", None), "右缘": ("右", None),
    "偏左": ("左", None), "偏右": ("右", None), "偏上": (None, "上"), "偏下": (None, "下"),
    "画面中央": ("中", "中"), "正中": ("中", "中"), "居中": ("中", "中"),
}


def axis(v: float, scale: float, lo: str, mid: str, hi: str) -> str:
    """把一个坐标分到三档。**边界用三分之一而不是二分之一** —— 二分法会把
    紧挨中线的目标硬判成左或右, 然后和答案里的"居中"打架, 全是误报。"""
    return lo if v < scale / 3 else (hi if v > scale * 2 / 3 else mid)


class Finding:
    __slots__ = ("kind", "sample_id", "detail")

    def __init__(self, kind: str, sample_id: str, detail: str):
        self.kind, self.sample_id, self.detail = kind, sample_id, detail


def gpt_text(row: dict[str, Any]) -> str:
    return "\n".join(t["content"] for t in turns_of(row) if t["role"] == GPT)


def norm_sentence(t: str) -> str:
    """比对模板重复用的归一化: 整块坐标 JSON 连同 label 一起去掉, 数字换成 #,
    剩下的散文才是句式骨架。

    **必须整块去掉而不是只去掉数字**。只抹数字的话, 「共#辆车辆。label车辆
    label车辆」会被当成一句 —— 可那几条的框各不相同, 根本不是同质化。
    """
    t = BOX_OBJ_RE.sub("", t)
    t = re.sub(r'[\[\]{}",:]', "", t)
    t = re.sub(r"\d+", "#", t)
    return re.sub(r"\s+", "", t)


def totals_in(q: str, a: str) -> list[tuple[str, int]]:
    """从一问一答里抠出"全图总数"。抠不准就宁可不抠 —— 假告警比漏报更贵,
    一个天天喊狼来了的检查器最后没人看。"""
    hits: list[tuple[str, int]] = []
    spans: list[tuple[int, str, int]] = []
    for m in TOTAL_RE.finditer(a):
        spans.append((m.start(), m.group(2), int(m.group(1))))
    if COUNT_Q_RE.search(q):                   # 计数问句的回答, 裸数字也算总数
        for m in UNIT_RE.finditer(a):
            spans.append((m.start(), m.group(2), int(m.group(1))))
    subj = subject_of(q, a)
    if subj is None:
        return []                              # 认不出主语就不参与比对
    for pos, unit, num in spans:
        if NEG_BEFORE_RE.search(a[max(0, pos - 4):pos]):
            continue                           # 「不是3辆」里的 3 不是总数
        hits.append((f"{subj}/{UNIT_ALIAS.get(unit, unit)}", num))
    return hits


def check(rows: list[dict[str, Any]], scale_default: int,
          dup_min_images: int, dup_min_len: int) -> list[Finding]:
    out: list[Finding] = []
    # 跨条累积
    box_owner: dict[tuple[int, int, int, int], set[str]] = defaultdict(set)
    sent_owner: dict[str, set[str]] = defaultdict(set)
    sent_sample: dict[str, str] = {}
    counts_by_img: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    vague_by_img: dict[str, set[tuple[str, str]]] = defaultdict(set)
    sample_of_img_count: dict[tuple[str, str, int], str] = {}

    for row in rows:
        sid = row.get("id", "?")
        meta = meta_of(row)
        img_id = meta.get("image_id") or sid
        scale = int(meta.get("bbox_scale") or scale_default)
        ans = gpt_text(row)
        boxes = [tuple(int(x) for x in m) for m in BOX_RE.findall(ans)]
        anomalies = meta.get("anomaly") or []
        hard_neg = bool(meta.get("hard_negative"))

        # --- 1) 框本身合不合法 ---
        for b in boxes:
            x1, y1, x2, y2 = b
            if x2 <= x1 or y2 <= y1:
                out.append(Finding("box_degenerate", sid, f"{list(b)} 宽或高 <= 0"))
            elif max(b) > scale or min(b) < 0:
                out.append(Finding("box_out_of_range", sid, f"{list(b)} 超出 [0,{scale}]"))

        # --- 2) 方位词 vs 框 ---
        # 只在"答案里恰好一个框"时判。多个框时"右上"可能指其中一个,
        # 拿所有框的外接矩形去比会把对的也判错。
        if len(boxes) == 1:
            x1, y1, x2, y2 = boxes[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hz = axis(cx, scale, "左", "中", "右")
            vt = axis(cy, scale, "上", "中", "下")
            for word, (want_h, want_v) in DIRECTIONS.items():
                if word not in ans:
                    continue
                bad = []
                if want_h and want_h != hz:
                    bad.append(f"横向说「{want_h}」框在「{hz}」")
                if want_v and want_v != vt:
                    bad.append(f"纵向说「{want_v}」框在「{vt}」")
                if bad:
                    out.append(Finding("direction_vs_box", sid,
                                       f"「{word}」 vs {list(boxes[0])}: " + "; ".join(bad)))
                break                      # 一条报一次就够, 不刷屏

        # --- 3) 正常图 / 难负样本却在断言异常 ---
        if (not anomalies or anomalies == ["normal"]) or hard_neg:
            # 逐个词看: 词前面没有否定词才算断言。
            hit = None
            for m in ASSERT_ANOMALY_RE.finditer(ans):
                ctxb = NOT_NEG_RE.sub("＊", ans[max(0, m.start() - 6):m.start()])
                if not NEG_NEAR_RE.search(ctxb):
                    hit = m
                    break
            if hit:
                tag = "难负样本" if hard_neg else "无异常标注"
                ctx = ans[max(0, hit.start() - 12):hit.end() + 8].replace("\n", " ")
                out.append(Finding("normal_but_asserts_anomaly", sid,
                                   f"{tag}, 却断言「{hit.group(1)}」: …{ctx}…"))

        # --- 4) label 里漏出本体 id ---
        for lab in LABEL_RE.findall(ans):
            m = RAW_ID_RE.search(lab)
            if m:
                out.append(Finding("raw_class_id_in_label", sid,
                                   f'label="{lab}" 含未翻译的本体 id「{m.group(0)}」'))

        # --- 5) 媒体占位符 ---
        field, media = media_of(row)
        tok = "<video>" if field == "videos" else "<image>"
        turns = turns_of(row)
        first = turns[0]["content"] if turns else ""
        if first.count(tok) != len(media):
            out.append(Finding("placeholder_mismatch", sid,
                               f"首轮 {first.count(tok)} 个 {tok}, 实际 {len(media)} 个媒体"))
        for t in turns[1:]:
            if tok in t["content"]:
                out.append(Finding("placeholder_not_first_turn", sid,
                                   "占位符出现在非首轮"))
                break

        # --- 累积跨条比对用的材料 ---
        for b in boxes:
            box_owner[b].add(img_id)
        for t in turns:
            if t["role"] != GPT:
                continue
            key = norm_sentence(t["content"])
            if len(key) >= dup_min_len:
                sent_owner[key].add(img_id)
                sent_sample.setdefault(key, sid)
        task = str(meta.get("task_type") or "")
        if not any(k in task for k in REGION_TASKS):
            pairs = [(turns[i]["content"], turns[i + 1]["content"])
                     for i in range(0, len(turns) - 1, 2)]
            for q, a in pairs:
                for unit, num in totals_in(q, a):
                    counts_by_img[img_id][unit].add(num)
                    sample_of_img_count.setdefault((img_id, unit, num), sid)
                # **概数也要按问答对看, 而且主语可能只在问句里。**
                # 「人员的聚集规模大概多大？」/「数百名以上, 无法逐个点清」——
                # 答句里一个"人"字都没有, 只在问句里。早先只扫答句, 于是这一条
                # 拿不到主语被跳过, 同图的"确为158名人员"就没人跟它对质。
                for m in VAGUE_UNIT_RE.finditer(a):
                    subj = (subject_of(a[m.start():m.start() + 16])
                            or subject_of(a) or subject_of(q))
                    if subj:
                        vague_by_img[img_id].add(
                            (f"{subj}/{UNIT_ALIAS.get(m.group(2), m.group(2))}", sid))

    # --- 6) 同图数字自相矛盾 ---
    for img_id, per_unit in counts_by_img.items():
        for unit, nums in per_unit.items():
            if len(nums) > 1:
                who = sorted(sample_of_img_count[(img_id, unit, n)] for n in nums)
                out.append(Finding("count_conflict_same_image", who[0],
                                   f"{img_id} 的「{unit.split('/')[0]}」同时出现 "
                                   f"{sorted(nums)}; 涉及 {', '.join(who[:4])}"))
        # 同一个单位上既报精确数又用概数: 计数口径打架。
        # 早先不分单位, 「共3辆车」配上「若干人员」也要告警, 全是噪音。
        for unit, nums in per_unit.items():
            who = sorted(s for u, s in vague_by_img.get(img_id, ()) if u == unit)
            if who:
                out.append(Finding("exact_vs_vague_same_image", who[0],
                                   f"{img_id} 的「{unit.split('/')[0]}」既报了精确数 "
                                   f"{sorted(nums)} 又用了概数; 涉及 {', '.join(who[:3])}"))

    # --- 7) 跨图共用同一个框 ---
    for b, imgs in box_owner.items():
        if len(imgs) > 1:
            out.append(Finding("box_shared_across_images", "-",
                               f"{list(b)} 出现在 {len(imgs)} 张不同的图: "
                               f"{', '.join(sorted(imgs)[:4])}"))

    # --- 8) 多图共用同一句 ---
    for key, imgs in sent_owner.items():
        if len(imgs) >= dup_min_images:
            out.append(Finding("sentence_shared_across_images", sent_sample[key],
                               f"同一句式出现在 {len(imgs)} 张图: {key[:60]}…"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="VQA 语义一致性体检(跨条比对)")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--box-scale", type=int, default=1000)
    ap.add_argument("--dup-min-images", type=int, default=3,
                    help="同一句式跨多少张图算同质化")
    ap.add_argument("--dup-min-len", type=int, default=12,
                    help="短于此长度的句子不参与重复判定(「有异常。」本来就该重复)")
    ap.add_argument("--examples", type=int, default=5, help="每类问题打印几个例子")
    ap.add_argument("--out", help="把被标记的条目写成 jsonl, 供下游过滤")
    ap.add_argument("--fail-over", type=float, default=-1.0,
                    help="被标记比例超过该值时退出码置 1, 便于挂进流水线")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for fp in args.files:
        raw = Path(fp).read_text(encoding="utf-8").strip()
        rows += (json.loads(raw) if raw.startswith("[")
                 else [json.loads(x) for x in raw.splitlines() if x.strip()])
    print(f"共 {len(rows)} 条 / {len({meta_of(r).get('image_id') for r in rows})} 张图\n")

    findings = check(rows, args.box_scale, args.dup_min_images, args.dup_min_len)
    by_kind: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_kind[f.kind].append(f)

    flagged = {f.sample_id for f in findings if f.sample_id != "-"}
    if not findings:
        print("没有发现跨条矛盾。")
    for kind, fs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        print(f"[{kind}]  {len(fs)} 处")
        for f in fs[:args.examples]:
            print(f"    {f.sample_id}: {f.detail}")
        if len(fs) > args.examples:
            print(f"    … 还有 {len(fs) - args.examples} 处")
        print()

    ratio = len(flagged) / max(1, len(rows))
    print(f"被标记样本 {len(flagged)} / {len(rows)}  ({ratio:.2%})")
    if args.out:
        reason: dict[str, list[str]] = defaultdict(list)
        for f in findings:
            reason[f.sample_id].append(f"{f.kind}: {f.detail}")
        with Path(args.out).open("w", encoding="utf-8") as fh:
            for r in rows:
                if r.get("id") in flagged:
                    fh.write(json.dumps({**r, "qc": reason[r["id"]]},
                                        ensure_ascii=False) + "\n")
        print(f"已写出 {args.out}")
    if 0 <= args.fail_over < ratio:
        print(f"超过阈值 {args.fail_over:.2%}, 退出码 1")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

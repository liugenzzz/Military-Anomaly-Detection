"""qc_consistency.py 的对照样本 —— 每一类检查都必须在这里响一次。

抽检时人工挑出来的那几处矛盾(同图数字打架 / 方位词与框相反 / 两张图共用一个框 /
三个 image_id 共用一句 / 正常图断言异常 …)全部复刻在这里。

**一个从来不报警的检查器比没有更糟** —— 它会让人以为数据是干净的。所以改完
qc_consistency 之后跑这个: 十类检查少响一类就是检查器坏了。

    python tests/qc_fixture.py
"""
import json
import sys

def row(sid, img, turns, **meta):
    m = {"image_id": img, "task_type": "judge", "anomaly": ["massing"],
         "hard_negative": False, "bbox_scale": 1000, **meta}
    conv = []
    for i, (q, a) in enumerate(turns):
        conv.append({"from": "human", "value": ("<image>\n" + q) if i == 0 else q})
        conv.append({"from": "gpt", "value": a})
    return {"id": sid, "images": [f"{img}.jpg"], "conversations": conv, "metadata": m}

B = '{"bbox_2d":[700,120,760,190],"label":"人群"}'
rows = [
    # 1. 同图数字打架: 一条共26人, 一条二三十名
    row("t_count_a", "DroneCrowd_1", [("画面里有多少人？", "共26人。")], task_type="count"),
    row("t_count_b", "DroneCrowd_1", [("清点一下人员。", "二三十名人员聚在一起。")], task_type="count"),
    # 2. 方位词与框矛盾: 说右上, 框在左下
    row("t_dir", "MAR20_4", [("异常在哪？", f'位于画面右上。[{{"bbox_2d":[80,820,160,900],"label":"机群"}}]')],
        task_type="grounded"),
    # 3. 跨图共用同一个框
    row("t_box_a", "FASDD_0", [("定位烟雾。", f"[{B}]")], task_type="locate_box"),
    row("t_box_b", "FASDD_5", [("定位烟雾。", f"[{B}]")], task_type="locate_box"),
    # 4. 正常图却断言异常
    row("t_norm", "DOTA_4", [("有问题吗？", "有异常。类型为异常聚集。")],
        anomaly=[], task_type="judge"),
    # 5. 难负样本断言异常
    row("t_hard", "CITY_9", [("有问题吗？", "有异常，车队正在行进。")],
        anomaly=[], hard_negative=True),
    # 6. 框退化 / 越界
    row("t_badbox", "X_1", [("定位。", '[{"bbox_2d":[500,500,400,600],"label":"车"},'
                                      '{"bbox_2d":[10,10,1200,90],"label":"车"}]')],
        task_type="locate_box"),
    # 7. 占位符与媒体数对不上
    {"id": "t_ph", "images": ["a.jpg", "b.jpg"],
     "conversations": [{"from": "human", "value": "<image>\n比较两图。"},
                       {"from": "gpt", "value": "左图更密集。"},
                       {"from": "human", "value": "<image>\n再看一次。"},
                       {"from": "gpt", "value": "同上。"}],
     "metadata": {"image_id": "PH_1", "task_type": "compare", "anomaly": ["massing"]}},
]
# 8. 三个 image_id 共用同一句
for i in range(3):
    rows.append(row(f"t_dup_{i}", f"DUP_{i}",
                    [("描述一下。", "画面中部可见多辆车辆沿道路依次排列，间距均匀，队形完整。")],
                    task_type="describe"))
# 两条精确总数打架: 26 vs 31
rows.append(row("t_count_c", "DroneCrowd_1", [("总共多少人？", "共31人。")], task_type="count"))

# ---- 反例: 这些**必须不报** --------------------------------------------
# 三条全是检查器早期的误报, 每一条都曾让告警里三分之二是噪音。修好了就锁在这,
# 以后谁再把计数规则放宽, 这里会先炸。
CLEAN = [
    # 分区对比不是矛盾: 左右两半各自的数, 不是全图总数打架
    row("n_compare", "N_1", [("左半区和右半区哪侧更密集？",
                              "右半区更密集：左半区0个目标，右半区30个。")],
        task_type="compare"),
    # 纠错题里被否定掉的那个错数, 不算总数
    row("n_correct", "N_2", [("画面中有3辆车辆，对不对？", "不对。车辆的数量是2辆，不是3辆。")],
        task_type="correct"),
    row("n_correct2", "N_2", [("清点一下车辆。", "2辆。")], task_type="count"),
    # 量词相同但主语不同: 3辆车 vs 0辆坦克
    row("n_subj_a", "N_3", [("清点一下画面中的车辆。", "3辆。")], task_type="count"),
    row("n_subj_b", "N_3", [("这张图里有多少辆坦克？", "0辆。")], task_type="negation"),
    # 正常图说"未见异常", 措辞里出现"异常"二字不该被当成断言
    row("n_norm", "N_4", [("有问题吗？", "未见异常，目标分布稀疏。")],
        anomaly=[], task_type="judge"),
    # 否定词就贴在词前面 —— 规则侧标准否定答案就长这样, 按整句搜否定词抓不到
    row("n_neg1", "N_5", [("有车队吗？", "否，未观察到车队机动的迹象。")],
        anomaly=[], task_type="negation"),
    row("n_neg2", "N_6", [("有问题吗？",
                           "未见异常。逐项核查：无军事装备集结、无人员异常聚集、"
                           "无烟火、无车队机动。")], anomaly=[], task_type="judge"),
    row("n_neg3", "N_7", [("有问题吗？",
                           "未见异常。目标分布稀疏，无烟火迹象，也没有成队行进的车辆。")],
        anomaly=[], task_type="judge"),
]

# 这些**必须报** —— 都曾被否定词逻辑误放过去
DIRTY = [
    # 转折连词里的"不"不是否定词: 「不过…出现烟雾」是真的自相矛盾
    row("d_but", "D_1", [("有问题吗？", "未见异常。不过右上角出现烟雾。")],
        anomaly=[], task_type="judge"),
    row("d_plain", "D_2", [("有问题吗？", "画面中存在人员异常聚集，规模约三十人。")],
        anomaly=[], task_type="judge"),
]

EXPECTED = {
    "normal_but_asserts_anomaly", "direction_vs_box", "box_degenerate",
    "box_out_of_range", "placeholder_mismatch", "placeholder_not_first_turn",
    "count_conflict_same_image", "exact_vs_vague_same_image",
    "box_shared_across_images", "sentence_shared_across_images",
}

if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from qc_consistency import check
    got = {f.kind for f in check(rows, 1000, 3, 12)}
    noise = check(CLEAN, 1000, 3, 12)
    missed = [r["id"] for r in DIRTY
              if r["id"] not in {f.sample_id for f in check([r], 1000, 3, 12)
                                 if f.kind == "normal_but_asserts_anomaly"}]
    missing, extra = EXPECTED - got, got - EXPECTED
    for k in sorted(EXPECTED | got):
        print(f"  {'✓' if k in got else '✗'} {k}")
    if missing:
        print(f"\n检查器漏了 {sorted(missing)} —— 这几类现在抓不出来了")
        raise SystemExit(1)
    if extra:
        print(f"\n多报了 {sorted(extra)}(对照样本里本来没有这类问题, 大概率是误报)")
        raise SystemExit(1)
    if missed:
        print(f"\n这些自相矛盾没被抓出来: {missed}")
        raise SystemExit(1)
    if noise:
        print(f"\n反例里报出了 {len(noise)} 条告警, 全是误报:")
        for f in noise:
            print(f"    {f.kind} / {f.sample_id}: {f.detail}")
        raise SystemExit(1)
    print(f"\n{len(rows)} 条对照样本, {len(EXPECTED)} 类检查全部命中; "
          f"{len(CLEAN)} 条反例零告警; {len(DIRTY)} 条该报的都报了。")

# 异常本体与 VQA Schema 设计

## 1. 异常类别本体

> ⚠️ **已在 v0.2 收敛为 4 类异常 + 正常，定稿见 [08_task_types.md](08_task_types.md)。**
> 下表是最初的 10 类设计，保留作为背景；停用的类别仍在 `configs/ontology.yaml` 中标 `enabled: false`。

分三级：`大类 → 异常类 → 判定线索`。判定线索是给规则引擎和 VLM prompt 用的。

| ID | 异常类 | 中文 | 典型视觉线索 | 主要数据来源 |
|---|---|---|---|---|
| `normal` | Normal | 正常 | 无成规模目标、无烟火、无越界 | VisDrone / DOTA / ERA non-event |
| `massing` | Force Massing | 兵力集结 | ≥N 个同类目标密集成簇、规则排列、阵地化布设 | DOTA/FAIR1M/MAR20 派生 + 合成 |
| `crowd_gathering` | Crowd Gathering | 人员聚集 | 人头密度骤增、成团分布 | DroneCrowd / ERA parade |
| `convoy` | Convoy Movement | 车队机动 | 车辆沿道路线性排列、同向运动 | VisDrone MOT 派生 |
| `explosion` | Explosion | 爆炸/火光 | 强亮斑、火球、扬尘、碎片扩散 | XD-Violence / UCF-Crime / ERA fire |
| `smoke` | Smoke | 烟雾 | 烟柱、烟幕遮蔽、灰白扩散团 | FASDD_UAV / FLAME2 / Boreal |
| `border_crossing` | Border Crossing | 越界移动 | 目标轨迹跨越边界线/进入禁区 | VisDrone-MOT + 虚拟边界派生 |
| `fortification` | Fortification | 阵地/工事构筑 | 新增土方、掩体、壕沟、施工机械 | ERA constructing + 合成 |
| `air_activity` | Air Activity | 航空活动异常 | 停机坪飞机数量异常、直升机起降 | MAR20 / DOTA plane 派生 |
| `naval_massing` | Naval Massing | 舰船集结 | 港口/海域舰船密集停泊或编队 | FAIR1M / DOTA ship 派生 |

> 设计原则：**每个异常类都必须有可自动判定的客观线索**，否则无法规模化生成标注，只能靠人工。

## 2. 中间层 Scene JSON（统一表示）

所有异构数据源先归一到同一个 scene 结构，QA 生成器只认这一种输入：

```json
{
  "image_id": "visdrone_0000001_02999_d",
  "image_path": "data/raw/visdrone/images/0000001_02999_d.jpg",
  "width": 1920, "height": 1080,
  "source_dataset": "VisDrone2019-MOT",
  "license": "academic-research-only",
  "view": "uav",                       // uav | satellite | cctv | ground
  "objects": [
    {"id": 0, "cls": "vehicle", "bbox": [120, 340, 210, 400], "track_id": 7, "attrs": {"moving": true}}
  ],
  "regions": [
    {"name": "restricted_zone_A", "type": "polygon", "points": [[0,600],[1920,540]]}
  ],
  "tracks": {"7": [[0, 100, 350], [1, 140, 348]]},   // track_id -> [[frame, cx, cy], ...]
  "events": [
    {"type": "border_crossing", "conf": 1.0, "evidence": {"track_id": 7, "direction": "south->north"}}
  ],
  "caption": null,
  "meta": {"frame_idx": 12, "altitude_hint": "low"}
}
```

`events` 可以来自：① 原数据集自带标签（ERA/FASDD）；② `tools/derive_events.py` 规则派生（集结/越界/车队）；③ 人工标注。

## 3. VQA 题型设计

> ⚠️ **已在 v0.3 收敛为 5 类任务，定稿见 [10_task_definition.md](10_task_definition.md)。**
> 下表是最初的 9 类草稿，写于异常类别还是 10 类、还没有每类 25000 条要求之前，保留作为背景。

| # | 题型 | 目的 | 生成方式 | 建议占比 |
|---|---|---|---|---|
| 1 | **异常判定**（是/否） | 最核心能力，二分类 | 规则 | 20% |
| 2 | **异常分类**（单选/开放） | 区分 10 类异常 | 规则 | 15% |
| 3 | **计数** | 抑制幻觉，量化态势 | 规则（数 bbox） | 12% |
| 4 | **定位 grounding** | 输出 bbox，可解释 | 规则（转坐标） | 12% |
| 5 | **属性识别** | 目标类型/状态/朝向 | 规则 + VLM | 10% |
| 6 | **空间关系** | 边界哪一侧、相对方位 | 规则（几何计算） | 8% |
| 7 | **态势描述** | 生成情报式描述 | VLM（带标注约束） | 10% |
| 8 | **推理研判** | 为什么判为集结 / 威胁等级 | VLM（带标注约束） | 8% |
| 9 | **否定/拒答** | 抗幻觉，问不存在的东西 | 规则（采样缺席类别） | 5% |

**否定样本必须要有**。实测中 VLM 微调后最大的问题就是"你问什么它都说有"，第 9 类题是唯一的解药。

### Grounding 坐标约定
统一用 **归一化到 [0, 1000] 的整数** `[x1, y1, x2, y2]`，与 Qwen-VL 系列的习惯一致；如果基座换成 InternVL 或 LLaVA，改 `tools/build_vqa.py` 里的 `BOX_SCALE` 一处即可。

## 4. 输出格式：LLaMA-Factory ShareGPT 多模态

```json
{
  "messages": [
    {"role": "user", "content": "<image>画面中是否存在异常军事活动？"},
    {"role": "assistant", "content": "是。画面中检测到 14 辆车辆在开阔地密集聚集，符合兵力集结特征。"}
  ],
  "images": ["data/raw/dota/P0032.png"],
  "extra": {"anomaly": "massing", "qa_type": "judgement", "source_dataset": "DOTA-v2.0"}
}
```

在 `data/dataset_info.json` 中注册（本仓库已提供），训练时 `--dataset military_anomaly_vqa` 即可。

> `extra` 字段 LLaMA-Factory 会忽略，但对做数据分析、按类别切分评测集非常有用，建议保留；如果版本报错就用 `tools/convert_to_llamafactory.py --strip-extra` 剥掉。

## 5. 数据切分与评测

- **train / val / test = 8 : 1 : 1**，并且**按 `image_id` 的源视频/源大图切分**，不能按 QA 随机切 —— 同一段视频的相邻帧落在两侧会造成严重的指标虚高。
- 评测集建议再手工精修 500–1000 条作为 `test_gold`，用于报告最终指标。
- 指标：判定题算 Accuracy / F1（异常为正类，重点看 **Recall 与误报率**）；计数题算 MAE 与 ±1 容差准确率；grounding 算 IoU@0.5；描述/推理题用 GPT-4 级模型做 LLM-as-judge 或人工 5 分制。

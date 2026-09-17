# 军事异常检测多模态 VQA 数据集

> ✅ **2026-09-16：方向已重定，类别体系落定为 `configs/ontology.yaml` v1.0.0。**
>
> 上一轮六个异常类里四个在验证中死掉，死因相同：**数据里没有这个现象**。
> 活下来的 smoke / explosion 都是数据集自带的真标注，死掉的四个都是
> 靠几何规则从 bbox 推出来的高层语义。
>
> 重定后的体系是**层次化多标签**：领域(air/sea/land) × 族(行为/征候) × 二级类型(训练标签)。
> 设计过程见 `docs/00_RESTART_BRIEF.md`（为什么要重来）、`docs/01_taxonomy_draft.md`
> （体系怎么设计的）、`docs/02_data_strategy.md`（数据从哪来）、
> `docs/03_HANDOFF_IMPL.md`（**下一步怎么实现** —— 交接给实现方的细则）。
> **权威定义只有一处：`configs/ontology.yaml`。** 文档是过程记录，配置是结论。

---

> 这份文件记录需求、工具链、踩过的坑。每次开工先读它 + 本体文件，
> 不要靠翻聊天记录重建上下文。改了决策就改这里。

---

## 一、要做什么

给 Qwen 系 VLM 做微调语料（LLaMA-Factory / vLLM 本地推理），让模型能：

1. **识别航拍与监控画面里的异常活动**；
2. 产出**多模态描述**（文字 + 图像区域框）；
3. 给出**异常解释**。

硬性要求（按重要性排）：

| # | 要求 | 说明 |
|---|------|------|
| 1 | **质量优先于数量** | 「不要为了凑数据而破坏了训练的精度」。宁可少，不许编。 |
| 2 | 描述类任务占 **60%** | `DESC_SHARE`，在 `configs/generate.yaml` |
| 3 | 每个异常类目标 **~25000 条 QA** | 够不着就如实报，不许放宽规则凑数 |
| 4 | 各异常类**不得共用一套描述骨架** | 每类有自己的 facet（侧面） |
| 5 | **坐标要稀有**，优先用方位词 | 框的格式对齐参考项目（见下） |
| 6 | 反同质化 | 多轮/单轮、多图/单图混合；问法每类 15–30 种 |
| 7 | 图仍是图，视频仍是视频 | 不把视频拍平成图，也不把图伪装成视频 |
| 8 | golden set **零人工** | 只收自动可验证的题 |
| 9 | 说话要自然 | 「说话不要那么别扭」。流程：抽关键词 → 搭骨架 → 大模型润色 |

### 输出格式（不可改）

经典 ShareGPT，对齐参考项目 `liugenzzz/target_detection_vl_dataset/qwen3vl_sft_builder`：

```json
{"id": "...", "images": ["..."], "system": "...",
 "conversations": [{"from": "human", "value": "<image>\n问"},
                   {"from": "gpt",   "value": "答"}],
 "metadata": {"task_type": "...", "image_id": "...", ...}}
```

- `conversations` 不是 `messages`；`from`/`value` 不是 `role`/`content`；`human`/`gpt` 不是 `user`/`assistant`
- 媒体占位符带换行 `<image>\n`，且**只在首轮**
- `system` 单列一列，不塞进 `conversations`
- 元数据字段叫 `metadata`（早期叫 `extra`，读的时候用 `sharegpt.meta_of()` 兼容）

### 媒体归集

`/mnt/si003010kcx0/mmdata/data_process/corpus_media/<name>/{images,videos}/<数据源>/`
二级分 images/videos，三级按数据源分（DroneCrowd 和 VisDrone 都有 `img0001.jpg`，不分会互相覆盖）。

---

## 二、已定的决策（别再重新讨论）

### 异常类别体系（v1.0.0，2026-09-16 定）

**层次化多标签**，不是单标签分类。一张图可以同时挂多个类
（语料里 explosion+smoke 就共现 7819 次，硬做单标签本来就是错的）。

```
领域 domain    air 空中 / sea 水面 / land 地面      ← 电磁域不做
族   family    activity 行为族 / indicator 征候族
二级 class     **训练标签**，模型要学会判的就是这些
三级 event     只写进 metadata，**不作为独立训练类别**
```

- **行为族**：主体在**做什么**。需要运动、过程、上下文 —— **单帧大多判不了**。
- **征候族**：客观**存在什么**。单帧足够。现有数据几乎全落在这一族。

三级为什么不升成训练类别：瓶颈是**独立图数**，细分只会把同一批图切成更多
更小的类，每格都不够训。数据够了随时提升，信息不丢。

领域怎么判（必须有明确规则，否则标签噪声无穷）：行为族按**行为主体所属的
作战域**（停机坪上的军机仍属 air）；征候族按**现象发生的位置**（舰上火情属
sea）；`concentration` 例外，按**被聚集资产的类型**走。

| 类 | id | 域/族 | enabled | 家底 |
|---|---|---|---|---|
| 烟雾 | `smoke` | land/征候 | ✅ | FASDD_UAV 12899，ready |
| 火光爆炸 | `explosion` | land/征候 | ✅ | FASDD_UAV+ERA 8259，ready |
| 灾害现场 | `disaster` | land/征候 | ✅ **恢复** | ERA 1056，thin |
| 集结 | `massing` | land/行为 | ✅ | ERA+Mendeley，thin，**数字待重统** |
| 密集分布 | `concentration` | land/征候 | ⛔ 接线中 | DroneCrowd+VisDrone ~7000，ready |
| 机群密集停放 | `air_concentration` | air/征候 | ⛔ 接线中 | MAR20 3842，ready |
| 舰船密集停泊 | `sea_concentration` | sea/征候 | ⛔ 接线中 | DOTA，图数待统计 |
| 毁伤 | `damage` | land/征候 | ⛔ 无数据 | 候选 xBD |
| 设施变化 | `infra_change` | land/征候 | ⛔ 无数据 | 需前后配对影像 |
| 机动 | `maneuver` | land/行为 | ⛔ 无数据 | 三级含 `convoy` |
| 越界 | `incursion` | land/行为 | ⛔ 无数据 | 三级含 `border_crossing` |
| 侦察 | `recon` | air/行为 | ⛔ 无数据 | 判据是轨迹形态 |
| 伪装隐蔽 | `concealment` | land/行为 | ⛔ 无数据 | **最适合仿真补**（天然成对） |
| 工事构筑 | `fortification` | land/行为 | ⛔ 无数据 | 需多时相 |
| 协同 | `coordination` | land/行为 | ⛔ 无数据 | 需时空关联 |
| 补给保障 | `logistics` | land/行为 | ⛔ 无数据 | 需看往返 |

**`enabled` 和 `status` 是两件事**，本体里分开记：

- `status` 说**数据**够不够 —— `ready` / `thin`（硬凑就是同质化）/ `empty`
- `enabled` 说这一轮跑不跑
- `blocked_by` 说「有数据但开不了，缺的是**代码**」

所以三个 `concentration` 是 `status: ready` + `enabled: false` + `blocked_by` ——
素材现成，卡在 build_vqa 还没接线（见下一节）。这和 `damage` 那种真没素材的
`empty` 是两回事，别混为一谈。

**id 没有跟着草案改名。** 草案里 `explosion` 叫 fire、`fortification` 叫 fortify，
没改：这些 id 已经写进 `adapters.py` 的 LABEL_MAP、`ds/era.py` 的 ANOMALY、
`facets.py`、`build_vqa.py`，以及已落盘的 `all_ev.jsonl`。为一个内部字符串做
全库改名，收益是零、风险是全库。分类学信息由 `domain` / `family` / `parent`
三个字段承载，与 id 无关。

**`convoy` / `border_crossing` 是三级事件，却仍单列为 class。** 唯一理由是
规则要有挂载点且必须一直可测（`resolve_overlap` 和 `tests/convoy_fixture.py`
都按 `e.type == "convoy"` 找事件）。它们带 `rule_only: true`，统计口径归到
`parent`。停用是「数据上分不出来」，不是「规则写错了」。

### DroneCrowd / VisDrone 不产出 massing（2026-09-16 定）

定向复查（`docs/16_free_look_massing_recheck.md`，15 张逐条人工核对）：
**11/15 是误报。** 标注 99 人的图模型说「没有看到明显的人群聚集」，标注 169 人的
说「绝对没有"密密麻麻一片"的情况」。

**根因是类目错配，不是阈值。** DroneCrowd 是人群**计数**数据集，拍的是广场、
校园、路口、球场 —— 里面根本没有「异常聚集」这个现象。从「校园广场上 117 个
行人」提不出异常，再准的密度判据也提不出。和 convoy / border_crossing 同类。

配置在 `massing/personnel` 的 `rule.hard_negative_datasets`。
这些图没有浪费，反而各得其所：

- **计数题** —— DroneCrowd 本行就是人群计数，GT 极可靠
- **困难负样本** —— 人很多却不是异常，直接教模型别一看到人多就报警
- **v1.0 新增：`concentration` 的正样本** —— 「人群密集」本身就是一个可训的
  客观征候。等它接线完成，这批图从「否定句」升级成「正面标签」，
  `hard_negative_datasets` 这一行届时可以撤掉。

⚠️ 计数题差点也丢了：`_count_mode` 原来先判「目标太小」就 `return "none"`，
压根走不到「给量级」那一档，于是一图上百个人头点一道计数题都出不来。
**「太小」只该否掉准数，不该连量级一起否掉** —— 问「大概多少人」完全成立。

### 「集结」不包括停放的航空器（2026-09-16 定）

集结说的是**地面力量向某处汇聚这个动作**；停机坪上停着的飞机是机场常态、是静态
事实。把它训成异常聚集，等于教模型看见任何一个军用机场都报异常。同理，DroneCrowd
那些广场图、ERA 的演唱会和宗教活动，都是人群密集但不是军事异常。

**v1.0 给了它们一个正面标签**：`concentration` / `air_concentration` /
`sea_concentration`（密集分布，征候族）。这比原来的做法更值 ——
负样本只能说「不是什么」，正面标签能说「是什么」。

| | 说的是 | 族 | 单帧能判吗 |
|---|---|---|---|
| `concentration` 密集分布 | 客观事实：目标密集排布 | 征候 | ✅ |
| `massing` 集结 | 行为判断：力量正在汇聚 | 行为 | ❌ 需要过程 |

**构型上两者完全一样**，区别只在「是不是正在汇聚」。混为一谈就是教模型看见任何
停车场、任何广场都报警 —— 这是整个体系里最容易出错的一格。

⚠️ **当前是过渡状态。** 三个 `concentration` 还是 `enabled: false`，因为
`build_vqa` 的 `hard_negative` 分支会出「未见异常」，而本类是正面标签 ——
同一张图两种口径，正是 `qc_consistency` 会抓的自相矛盾。在它接线之前：

- MAR20 / 民用车簇仍走 massing/equipment 的 `on_require_fail: hard_negative`
- ERA 的 concert / party / religious_activity 已从 massing 映射移出，
  改指向 `concentration`，于是当前会被 `derive_events` 按 `enabled` 丢掉并打印。
  **那是有意的** —— 比顶着「集结」这个错标签进训练集强。

困难负样本的措辞按簇里的实际类别分情况（`meta.hard_negative_kind`）：

```
aircraft_parking → 未见异常。12架军用飞机排列整齐地停在停机坪上，属于日常停放，
                   不构成兵力或装备的集结。
civil_cluster    → 未见异常。画面中虽有12辆车辆密集成簇、达到了集结的规模条件，
                   但均为民用目标，未见坦克、装甲车等军事装备，属于正常场景。
```

⚠️ 写死一套文案会出事：对一张满是军机的图说「均为民用目标，未见军机」是睁眼说
瞎话，而且正好是 `qc_consistency` 会抓的那类自相矛盾。

接线做完之后，这两句要改成「达到密集规模但非集结」的口径，并把航空器/舰船从
massing/equipment 的 `target_classes` 里移除 —— 那时它们由正面标签承接，
不再需要靠否定句表达。

### 坐标制式 —— 改错了整批框都是废的，而且不报错

- Qwen2-VL / Qwen3-VL → `relative_1000`（归一化到 0~1000）
- **Qwen2.5-VL → `absolute_pixel`**（smart_resize 之后的绝对像素）

两代不兼容。当前配置是 `relative_1000`，在 `configs/generate.yaml` 的 `coordinate.mode`。

### 分工：规则出事实，大模型出语言

抄的 GeoChat / VRSBench 的做法，也和参考项目一致：

- **规则侧**（`build_vqa.py`）只产出：框、数量、方位、10 来个字的短事实
- **大模型侧**（`llm_qa.py`）拿 FACTS + 图，写自然语言
- 规则侧**不要写长篇散文** —— 那是同质化的根源

### 推理端点

12 路，同一个 `Qwen3.8-27B`，128k 上下文，配置在 `configs/generate.yaml`：

```
10.107.230.59:8001-8004    10.200.100.103:8001-8004    10.107.238.7:8001-8004
```

- `review` 留空。**这 12 路是同一个模型，分几路出来当审稿并不解决自审问题**，六维 review 仍是摆设。要真审稿得接一个不同的模型。
- `chat_template_kwargs: {enable_thinking: false}` 必须带 —— Qwen3 默认吐思维链，不关的话 `json_mode` 一条都解析不出来
- `inline_images: true` 必须 —— 否则要求 vLLM 起服务时带 `--allowed-local-media-path`

---

## 三、当前进度

### ✅ 已完成

| 模块 | 文件 | 状态 |
|---|---|---|
| 数据适配器 | `tools/ds/*.py` | 9 个数据源 |
| 事件派生 | `tools/derive_events.py` | 密度聚类 / 线性队列 / 标签转移；带否决记账 |
| 规则侧 QA | `tools/build_vqa.py` | 14 种题型 |
| 大模型侧 | `tools/llm_qa.py` | ping / screen / generate / verify |
| 场景合并 | `tools/merge_scenes.py` | 取代手敲 `cat` |
| 一致性 QC | `tools/qc_consistency.py` | 10 类跨条检查 |
| 数据体检 | `tools/inspect_data.py` | 标注统计 + VLM 自由看图 |
| 回归测试 | `tests/*.py` | qc_fixture / merge_fixture |

### 语料现状（2026-09-16，方案 A 落地后）

**59599 个 scene / 9 个数据源**（`data/all.jsonl`）：

```
FASDD_UAV          25097     VisDrone2019-MOT   8448    VisDrone2019-DET   6471
Mendeley-UAV-Mil    3982     MAR20              3842    DOTA-v2.0          3181
DroneCrowd          3176     ERA                2701    ERA-SingleFrames   2701
```

各类事件产量（`data/all_ev.jsonl`）—— ⚠️ **这组数字是 v0.5.6 跑出来的，v1.0
本体改了映射，必须重跑 `derive_events` 才作数**：

| 类 | v0.5.6 事件数 | 来源 | 实证 | v1.0 变化 |
|---|---|---|---|---|
| smoke | 12899 | FASDD_UAV | 自由看图 12/12 确认 | 不变 |
| explosion | 8259 | FASDD_UAV + ERA | 自由看图 12/12 确认 | 不变 |
| disaster | 0（停用） | ERA | 真标注 | **恢复启用**，预计 ~1056 |
| massing/personnel | 923 | ERA 游行/抗议/集会 | 真异常事件 | **会降** —— concert/party/<br>religious_activity 已移出 |
| massing/equipment | 84 | Mendeley 坦克簇 | 自由看图确认「纵队或楔形编队」 | 不变 |

**困难负样本 18878**（占正常样本 41.7%），其中 DroneCrowd 3176 + VisDrone 4045
是按 `hard_negative_datasets` 转过来的。这批在 `concentration` 接线后会改为
正面标签。

⚠️ **ERA / ERA-SingleFrames 是同一批 2701 段素材的两种形态**，排配额时不能算两次。
ERA-SF 的**目标框是 0**（ERA 本来没有 bbox），只能出描述题和判定题。

### ⛔ 未决问题

1. **类间比例严重失衡**（v0.5.6 实测 smoke 12899 : massing 1007 = 12.9:1，
   清理 ERA 民事活动后只会更悬殊）。`apply_quota` 会显眼报出来。
   补数据源和下采样**两者不冲突**，都要做 —— 但**不要靠在 massing 的同一批
   画面上反复出题来凑**，那是同质化不是数据量。
2. 各类都够不到 25000/类。**瓶颈是独立图数，不是 QA 条数**：
   `QA = 独立图数 × 每图出题数`，只有前者携带视觉多样性。
   `qa_per_image` 的健康区间是 3–8；massing 要凑 25000 需每图出题 27 次，
   disaster 需 23.7 次 —— 落不进区间就是该补数据源的信号。
3. **行为族 8 类目前全空**。单帧数据填不了「过程」。按 `docs/02_data_strategy.md`
   的结论，唯一能系统性填上的路子是**仿真引擎**。
4. **单帧几何分不出车队和密集车流**（`convoy` 已因此停用）。
5. **数据源层面的「类目错配」是最贵的一类错**，比阈值错贵得多：阈值能调，
   数据里没有的现象调不出来。新接一个数据源时先问「它到底拍的是什么」，
   再问「这个现象在里面存在吗」——`convoy` / `border_crossing` /
   `DroneCrowd 的 massing` / `ERA 的演唱会` 四次都栽在这里。

### 📋 待办（按优先级）

> **实现细则见 `docs/03_HANDOFF_IMPL.md`** —— 下面 1–4 条的逐文件逐行改法、
> 真值表、验收标准和红线都在那里，不要凭这份清单直接动手。


1. **重跑 `derive_events`**，把 v1.0 本体下各类的真实产量统计出来
   （massing 清理后还剩多少、disaster 恢复后有多少）
2. **给 `concentration` 接线**：build_vqa 支持「有 concentration 事件时，
   负样本答案改为『达到密集规模但非集结』」，然后把三个 concentration 的
   `enabled` 改回 true，并把航空器/舰船从 massing/equipment 的
   `target_classes` 移除
3. **统计 DOTA 的 ship/harbor 按图分布**，确定 `sea_concentration` 的 status
4. `concentration` / `disaster` 的 facet 与 prompt（`facets.EXPECTED` 要补两格）
5. **同图事实卡**：一张图的所有数字来自同一份计算，从生成端根除数字打架
6. describe prompt 里加 **GT 类别白名单**（防跨类污染）
7. 规则侧短答案补理由从句；问法扩到每类 15–30 种

---

## 四、怎么跑

```bash
# 0. 端点体检 —— 开跑前必做
python tools/llm_qa.py ping

# 1. 各数据源 -> interim
python tools/prepare.py dronecrowd --root .../military/DroneCrowd --out data/interim/dronecrowd.jsonl
#    （--ann-dir 一般不用传，缺省在 --root 下递归找）

# 2. 合并 -> all.jsonl（别用 cat，见「坑」）
python tools/merge_scenes.py --dry-run     # 先核对清单
python tools/merge_scenes.py

# 3. 派生事件
python tools/derive_events.py --scenes data/all.jsonl --out data/all_ev.jsonl --overwrite

# 4. 体检（标注统计 + 让模型自由看图）
python tools/inspect_data.py --scenes data/all_ev.jsonl --out-dir data/inspect --per-class 12

# 5. golden set（必须先做，否则泄漏）
python tools/make_golden.py --scenes data/all_ev.jsonl --out-dir data/golden

# 6. 规则侧 QA —— **--target-per-class 必须传**，否则负样本不受控
python tools/build_vqa.py --scenes data/all_ev.jsonl --out-dir data/vqa_rule \
    --exclude-ids data/golden/exclude_ids.txt --target-per-class 25000

# 7. 大模型侧描述
python tools/llm_qa.py generate --scenes data/all_ev.jsonl --out data/vqa_llm/all.json \
    --exclude-ids data/golden/exclude_ids.txt

# 8. 合并 + 体检 + 媒体归集
python tools/merge_dataset.py --in data/vqa_rule data/vqa_llm --out data/vqa
python tools/check_dataset.py data/vqa/*.json          # 结构、分布
python tools/qc_consistency.py data/vqa/*.json         # 跨条自相矛盾
python tools/export_media.py --vqa-dir data/vqa --media-root /mnt/.../corpus_media --name military
```

**改完代码跑这两个**（十几秒）：

```bash
python tests/qc_fixture.py       # QC 的 10 类检查各命中一次，反例零告警
python tests/merge_fixture.py    # 合并器对 4 种事故处理正确
python tests/convoy_fixture.py   # 车队判定的 8 个对照场景
python -m pyflakes tools/ tests/
```

输入参数的口径：**scene 级用 `--scenes`，VQA 目录级用 `--in`**
（`--vqa-dir` 作为别名保留，老命令不会失效）。

---

## 五、踩过的坑（别再踩）

这一节是血泪，每条都真实发生过。共同点：**静态检查抓不到，全在"接缝"上。**

### 静默降级 —— 最贵的一类

| 坑 | 症状 | 教训 |
|---|---|---|
| DOTA 标签在嵌套目录 | 1491 图 → 18963 切片 → **0 个框** | 读不到标注要硬失败，不能当"这批图没目标" |
| DroneCrowd 目录假设错 | 3120 帧**全部 0 目标**，只打了一句 warn | 同上。现在零目标直接 `raise` |
| DroneCrowd 帧号 | `int(尾部6位)` 把 `img001001` 读成 1001，标注里是 1 | 两个 bug 会叠加，修一个不够 |
| `load_endpoints` 读不了嵌套 yaml | 静默回落到 `api.openai.com`，整轮 SSL 失败 | **删掉兜底链**，读不到就报错 |
| `image_message` 找不到文件 | 悄悄改发路径 URL，服务端回 `RemoteDisconnected` | 缺文件要说人话 |
| 本体里没有的类别 | 落成「存在异常，为**异常**」和 `"label":"crowd_gathering区域"` | 叫不出名字就换措辞，绝不落英文 id |

### 配置与默认值

| 坑 | 症状 |
|---|---|
| `inline_images: false` | 发裸路径给 vLLM → `Cannot load local files without --allowed-local-media-path` |
| 没关思维链 | `content` 里全是「我们需要回答用户：…」，`json_mode` 全废 |
| ping 的探针图是 1×1 | Qwen-VL `min_pixels=200704`，小图让视觉塔在服务端崩 → **HTTP 500**（不是 400） |
| `--inline-images` 三个子命令各写各的默认值 | ping 全绿，正式跑全挂 |
| `review: []` 是空列表不是 None | 回落判的是 `is None`，落不进去 → 「没有可用的推理端点」 |

### 数据处理

| 坑 | 症状 |
|---|---|
| `cat data/interim/*.jsonl` | ①卷进 `gen_req.jsonl`（它也有 `image_id`/`image_path`）②缺行尾换行把两条粘成一条**静默丢失** |
| 把视频当图发 | ERA 的 `.mp4` 被标成 `image/jpeg` base64 → HTTP 400 超 max_model_len |
| 画框外的人头点 | 硬裁后 `x2 < x1`，46.7 万个里混进一个退化框 |
| 关键词子串匹配 | `vd_mot_testdev` 里含 `test` → 整个数据划分被当演示数据跳掉 |

### 自由看图（必看）

`docs/15_free_look_findings_raw.md` 是让模型对 57 张真实图**无任何约束**地描述
「你看到了什么」。它推翻了两个我靠合成数据得出的结论，**改判定规则之前先读它**。

分析时注意：**不要用正则统计**。第一次分析我用关键词匹配，得出「massing 只有
1/10 确认」的结论，逐条读原文才发现 Mendeley 那张模型明说「一队排列成纵队或
楔形编队的装甲车辆」是真阳性，只是"纵队编队"不在我的词表里。断章取义比不分析更糟。

### 算法上的硬限制（不是 bug，别再试图调参解决）

- **单帧几何分不出车队和密集车流。** 见上「未决问题 3」。`border_crossing` 当初
  被停用是同一个道理：单帧看不到运动。遇到这类，先量分布再决定，别硬调阈值。
- **收紧"整齐度"类阈值往往适得其反。** 随机点凑出的线比真实物体还规整 ——
  真实世界有抖动，随机采样没有。
- **单链接聚类（纯 union-find）在密集场景里会链式串联。** 等价于 DBSCAN 的
  minPts=1。1920×1080 里 158 人均匀散布就能串出 20 人的"簇"。凡是用连通性
  聚类的地方都要配核心点要求（`min_core_neighbors`）**和**密度检查
  （`max_area_per_object`）——「够大」不等于「够密」。

### 我自己犯的错

- **把幻觉写进了 prompt**：`smoke/color.txt` 写着「可以补一句该颜色通常对应什么燃烧物」，
  而 `system.txt` 第 4 条明写「不推测事件起因」。现在有 `INFERENCE_BAN` 全局词表。
- **QC 误报比漏报更贵**：第一版三分之二告警是假的（分区对比、被否定的错数、
  量词相同主语不同）。一个天天喊狼来了的检查器最后没人看。
- **`compileall` 过了不等于跑得对**：`_differs` 被我插进 `main()` 中间，
  后面整个循环成了 `return` 之后的死代码，语法完全合法。
- **顺手的批量正则改**：去 f-string 前缀的正则啃坏了 `"鿿"`。装饰性改动不值得冒险。
- **结论下太早**：说过「DOTA 规则没跑通」（实际是按设计转困难负样本）、
  「车队能有几千」（实际几十条量级）。**先造对照数据自证，再下结论。**

---

## 六、写代码的约定

1. **失败要响。** 读不到标注、找不到文件、零产出 —— 硬失败，不要 warn 了继续。
   静默降级的代价是废数据一路混到最后才被发现。
2. **报错要带上游的原话。** `urllib` 的 `HTTPError` 字符串只有 "HTTP Error 500"，
   真正的 traceback 在 body 里（见 `llm_qa.http_detail`）。
3. **单一出口。** 同一件事只在一个地方做：取图走 `Scene.still`，
   数量措辞走 `count_phrase()`，字段读取走 `sharegpt.meta_of()`。
   靠"记得在每处都判一下"是防不住的。
4. **改了判定逻辑，先造对照数据验证**，正反例都要（`tests/qc_fixture.py` 的写法）。
5. **注释写「为什么」，尤其是为什么不用那个显而易见的写法。**
   代码本身已经说清「做了什么」。
6. 报数字要说清口径（ERA 那 2173×2 不是 4346 份独立素材）。

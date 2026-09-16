# 军事异常检测多模态 VQA 数据集

> 这份文件是**项目的唯一事实来源**。需求、已定的决策、当前进度、已知坑都在这里。
> 每次开工先读它，不要靠翻聊天记录重建上下文。改了决策就改这里。

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

### 异常类别：四类

| 类 | id | 状态 | 数据来源 |
|---|---|---|---|
| 异常聚集 | `massing` | ✅ 启用 | **仅 ERA 的游行/抗议/集会 + Mendeley 坦克簇**，见下 |
| 爆炸火光 | `explosion` | ✅ 启用 | FASDD_UAV / ERA |
| 烟雾 | `smoke` | ✅ 启用 | FASDD_UAV |
| ~~车队机动~~ | `convoy` | ❌ 停用 | v0.5.5。自由看图 12 张里 7 张模型说是「静止停放」 |
| ~~越界移动~~ | `border_crossing` | ❌ 停用 | 单帧既看不到运动也看不到虚拟线，这类会教模型编造 |
| ~~灾害现场~~ | `disaster` | ❌ 停用 | 规则成立，但没有数据源（2026-09 决定不再补数据集） |

停用的类**规则和 prompt 都原样留着**，补上数据源把 `enabled` 改回 `true` 即可。
`tests/convoy_fixture.py` 会在自己的本体副本里强制打开 convoy —— 停用是「数据上
分不出来」，不是「规则写错了」，规则逻辑必须一直可测。

### DroneCrowd / VisDrone 只做困难负样本 + 计数题（2026-09-16 定）

定向复查（`docs/16_free_look_massing_recheck.md`，15 张逐条人工核对）：
**11/15 是误报。** 标注 99 人的图模型说「没有看到明显的人群聚集」，标注 169 人的
说「绝对没有"密密麻麻一片"的情况」。

**根因是类目错配，不是阈值。** DroneCrowd 是人群**计数**数据集，拍的是广场、
校园、路口、球场 —— 里面根本没有「异常聚集」这个现象。从「校园广场上 117 个
行人」提不出异常，再准的密度判据也提不出。和 convoy / border_crossing 同类。

配置在 `rule.hard_negative_datasets`。这些图没有浪费，反而各得其所：

- **困难负样本** —— 人很多却不是异常，直接教模型别一看到人多就报警
- **计数题** —— DroneCrowd 本行就是人群计数，GT 极可靠

⚠️ 计数题差点也丢了：`_count_mode` 原来先判「目标太小」就 `return "none"`，
压根走不到「给量级」那一档，于是一图上百个人头点一道计数题都出不来。
**「太小」只该否掉准数，不该连量级一起否掉** —— 问「大概多少人」完全成立。

### 「集结」不包括停放的航空器（2026-09-16 定）

集结说的是**地面力量向某处汇聚这个动作**；停机坪上停着的飞机是机场常态、是静态
事实。把它训成异常聚集，等于教模型看见任何一个军用机场都报异常。

**但 MAR20 那 3842 张没有浪费** —— 航空器仍留在 `target_classes` 里，只是从
`require_any` 移出，于是停机坪照样聚成簇、然后落进 `on_require_fail: hard_negative`。
「看着像集结（一堆军用装备聚在一起）但不是」恰恰是最有价值的困难负样本，直接教
模型分清「密集停放」与「集结」。移出 `target_classes` 的话这批图会变成平平无奇
的正常样本，白白浪费。

困难负样本的措辞按簇里的实际类别分情况（`meta.hard_negative_kind`）：

```
aircraft_parking → 未见异常。12架军用飞机排列整齐地停在停机坪上，属于日常停放，
                   不构成兵力或装备的集结。
civil_cluster    → 未见异常。画面中虽有12辆车辆密集成簇、达到了集结的规模条件，
                   但均为民用目标，未见坦克、装甲车等军事装备，属于正常场景。
```

⚠️ 写死一套文案会出事：对一张满是军机的图说「均为民用目标，未见军机」是睁眼说
瞎话，而且正好是 `qc_consistency` 会抓的那类自相矛盾。

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

各类事件产量（`data/all_ev.jsonl`）—— **三个类都有自由看图的实证背书**：

| 类 | 事件数 | 来源 | 实证 |
|---|---|---|---|
| smoke | 12899 | FASDD_UAV | 自由看图 12/12 确认 |
| explosion | 8259 | FASDD_UAV + ERA | 自由看图 12/12 确认 |
| massing/personnel | 923 | **仅 ERA 游行/抗议/集会** | 真异常事件 |
| massing/equipment | 84 | Mendeley 坦克簇 | 自由看图确认「纵队或楔形编队」 |

**困难负样本 18878**（占正常样本 41.7%），其中 DroneCrowd 3176 + VisDrone 4045
是按 `hard_negative_datasets` 转过来的。

⚠️ **类间比例 12.9:1（smoke : massing），严重失衡。** `apply_quota` 现在会显眼
报出来。要么给 massing 补数据源，要么对 smoke 主动下采样 —— **不要靠在 massing
的同一批画面上反复出题来凑**，那是同质化不是数据量。

⚠️ **ERA / ERA-SingleFrames 是同一批 2701 段素材的两种形态**，排配额时不能算两次。
ERA-SF 的**目标框是 0**（ERA 本来没有 bbox），只能出描述题和判定题。

### ⛔ 未决问题

1. **类间比例 12.9:1**（smoke 12899 : massing 1007）。补数据源还是下采样，待定。
2. 各类都够不到 25000/类。**要接受还是补数据源的决策题。**
   注意事件数 ≠ QA 条数：一个 scene 能出多道题，所以 QA 总量会高于事件数，
   但类间比例基本由事件数决定。
3. **单帧几何分不出车队和密集车流**（convoy 已因此停用）。
4. **数据源层面的「类目错配」是最贵的一类错**，比阈值错贵得多：阈值能调，
   数据里没有的现象调不出来。新接一个数据源时先问「它到底拍的是什么」，
   再问「这个现象在里面存在吗」——`convoy` / `border_crossing` /
   `DroneCrowd 的 massing` 三次都栽在这里。

### 📋 待办（按优先级）

1. 跑通自由看图，看真实数据长什么样
2. **同图事实卡**：一张图的所有数字来自同一份计算，从生成端根除数字打架
3. describe prompt 里加 **GT 类别白名单**（防跨类污染）
4. facet 按 MMAD 七类重排（会改写整个 `task_type`，要整体重跑）
5. 规则侧短答案补理由从句；问法扩到每类 15–30 种

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

# Military Anomaly Detection — 多模态 VQA 数据集构建

用公开数据构建**军事异常检测**的多模态 VQA 数据集，训练一个能在**航拍/监控画面**中识别异常军事活动（集结、爆炸、烟雾、越界移动等）的视觉语言模型。训练框架为 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。

## 核心结论

**没有现成可用的「军事异常 VQA」数据集**，但完全可以构建。路线是四层拼装 + 规则化 QA 生成：

```
军事目标外观层   MAR20 / Mendeley-UAV军事目标 / MOCO(申请制)     → 认得出"是什么"
异常事件层       ERA / XD-Violence / FASDD / Drone-Anomaly       → 认得出"出事了"
语言标注层       CapERA / HIVAU-70k / UCA / CUVA                 → 说得出"为什么"
航拍底座层       VisDrone / DroneCrowd / DOTA / FAIR1M / xView   → 视角、尺度、负样本
        ↓  adapters.py
  统一 Scene 中间表示
        ↓  derive_events.py      集结=密度聚类，越界=轨迹跨边界
  规则派生事件标签
        ↓  screen.py + llm_qa.py review     五道闸质量筛选，预期保留 65-75%
  干净的 Scene
        ↓
  ┌── build_vqa.py   规则侧  判定/方位/坐标框/计数/构成/对比/纠错/时序 —— 答案由标注唯一决定
  └── llm_qa.py      LLM 侧  描述(21 个侧面) / 推理
        ↓  generate → must-not 硬过滤(零成本, 先滤跑题的) → 六维 review
        ↓  merge_dataset.py   合并两侧 + 生成 dataset_info.json
        ↓  export_media.py    图片/视频归集到语料库目录并改写路径
  LLaMA-Factory ShareGPT 指令数据
```

**四类异常各用各的描述骨架**：聚集讲构型、爆炸讲亮度、烟雾讲形态走向、越界讲时序，
每个侧面配 `must-not` 禁用词，答案里出现就整条丢——没有这道闸，跑几轮就会退化成同一种三段式。

**LLM 只做语言表达，判断权在规则手里**：先用标注算出 FACTS 事实包（数量、坐标、事件），LLM 在事实约束下改写成多样化指令问答，再由另一个模型逐条校验。详见 [docs/05_instruction_design.md](docs/05_instruction_design.md)。

完整数据源清单、许可说明与推荐取样配比见 **[docs/01_data_sources.md](docs/01_data_sources.md)**；
异常本体与 VQA schema 设计见 **[docs/02_ontology_and_schema.md](docs/02_ontology_and_schema.md)**；
**按优先级排好的下载清单、以及视频要不要处理，见 [docs/03_acquisition_checklist.md](docs/03_acquisition_checklist.md)**；
质量筛选的五道闸与阈值见 [docs/04_quality_screening.md](docs/04_quality_screening.md)；
指令型 QA 的生成与防幻觉设计见 [docs/05_instruction_design.md](docs/05_instruction_design.md)；
**最终选定哪 8 个源数据集、不选哪些、以及框架选型，见 [docs/06_final_selection.md](docs/06_final_selection.md)**；
**这 8 个的下载地址与下载顺序见 [docs/07_download_list.md](docs/07_download_list.md)**；
**异常类型定稿（4 类 + 正常，每类 25000 条）与各类数据缺口见 [docs/08_task_types.md](docs/08_task_types.md)**；
**9 个数据集各自的处理方式、命令与坑见 [docs/09_dataset_processing.md](docs/09_dataset_processing.md)**；
**任务类型定稿（5 类）、数量分配与数据可行性核对见 [docs/10_task_definition.md](docs/10_task_definition.md)**；
**20 个生成任务的逐条规格（问什么、答什么、谁生成）见 [docs/11_generation_tasks.md](docs/11_generation_tasks.md)**；
**描述任务的分侧面设计（四类异常各用各的骨架、must-not 防模板化）见 [docs/12_description_design.md](docs/12_description_design.md)**；
**每个侧面的实际 Q/A 样例（供审阅修改）见 [docs/13_sample_qa.md](docs/13_sample_qa.md)**；
**方位指代与坐标的问法拆分、轮数/图数的反同质化配比见 [docs/14_qa_form_design.md](docs/14_qa_form_design.md)**。

## 一条命令跑完

```bash
# 不配端点: 处理 + 规则生成走完, LLM 那步只导出请求(可喂 vLLM 离线批推理)
bash run_all.sh /path/to/数据根目录

# 配了端点: 一路跑到 LLM 描述/推理生成 + must-not 硬过滤 + 六维 review
VLM_BASE_URL=http://127.0.0.1:8000/v1 \
VLM_MODEL=qwen2.5-vl-72b-instruct \
REVIEW_MODEL=internvl2_5-78b \
bash run_all.sh /path/to/数据根目录

python tools/doctor.py --root /path/to/数据根目录   # 只体检: 看手上的数据能产出哪些异常类
```

没下到的数据集会自动跳过并在末尾列出，不会中断流程。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `VLM_BASE_URL` | 空 | OpenAI 兼容端点。**留空就只导出请求，不调用任何 API** |
| `VLM_MODEL` | `qwen2.5-vl-72b-instruct` | 写描述/推理的模型 |
| `REVIEW_MODEL` | 同 `VLM_MODEL` | 审稿模型。**务必换一个**——同一个模型审自己写的答案基本全过 |
| `TARGET_PER_CLASS` | `25000` | 每个异常类的目标条数。富余的按图下采样，稀缺的自动提高每图侧面数，仍不足则如实报缺口 |
| `FACETS_PER_IMAGE` | `4` | 每张图抽几个描述侧面（会被配额上调/下调） |
| `INLINE_IMAGES` | `0` | 置 1 把图转 base64 内联。远端 API 需要；本地 vLLM 挂同一块盘就不用 |
| `WORKERS` | `8` | 并发数 |

配额两个方向都调：`explosion`/`smoke` 源数据富余，按 **image_id 分组**下采样（同图的几条
QA 要么一起留要么一起丢，否则后面按组切 train/test 会串），`border_crossing` 这类源数据本来
就少的，把每图侧面数顶到侧面池抽干为止——**不靠复制样本凑数，凑不够就在报告里点名**。

LLM 调用单条失败不中断整批：失败的记 `error` 字段、答案留空，`verify` 那步当空答案滤掉；
重跑时命中 `.llm_cache`，已经成功的不重复计费。

## 配额: 多的筛精，少的放宽

数据量在四类异常之间差了一个数量级，直接按同一套阈值跑，产出必然一头沉。两个方向分开处理：

**多的筛精**（explosion / smoke）。超出配额的部分不是随机丢，而是按质量分排序后
**在各数据源之间轮着取**：

- 质量分 = 清晰度分位 × 0.35 + 目标数分位 × 0.25 + 事件置信度 × 0.30 + 困难负样本加分 0.10，
  放宽档样本 −0.20、闸5 判「不确定」的 −0.10；
- 清晰度与目标数都先在**本数据源内部**换算成分位再比——卫星图天然锐利、夜间监控天然发糊，
  blur 的绝对值跨数据源没有可比性；
- 轮取是为了防止某个又大又清晰的数据源把一类的名额全占了。全来自一个源的 25000 条，
  训出来的是那个源的模型。

**少的放宽**（massing/equipment、border_crossing）。`configs/ontology.yaml` 里每条规则带一个
`relax` 档，由 `derive_events.py --relax-below N` 触发，只对产量不够的类生效：

| 类别 | 严格档 | 放宽档 |
|---|---|---|
| 人员聚集 | ≥20 人成簇 | ≥12 人 |
| 装备集结 | ≥5 件军事装备 | ≥3 件 |
| 越界移动 | 位移 ≥2% 对角线 | ≥1%，且放宽目标类别 |

放宽的是**召回，不是结论的确定性**。三条约束保证这一点：补出来的事件带 `relaxed` 标记；
对应答案的措辞随之变软（「判定为集结」→「存在苗头，可能属于集结…规模有限，建议继续观察确认」）；
放宽样本最多占该类的一半，否则这一类的分布会被边缘样本主导，模型学到的就是「稍微聚一下就算集结」。

装备集结的 `require_any`（必须含军事目标）**不在放宽之列**——放宽它等于把民用停车场
标成装甲集群，那不是补量，是造错标。

## 题型

规则侧 11 种（答案全部由标注唯一决定，零幻觉、零成本）：

| 题型 | 问什么 | 谁出得了 |
|---|---|---|
| `judge` | 有没有异常、是哪类 | 所有图，每张必出 |
| `locate_verbal` | 异常在画面什么方位（**不给坐标**） | 有区域信息的图，占定位题七成 |
| `locate_box` | 框出异常区域 | 同上，占三成 |
| `count` | 某类目标有几个 | 有检测框的图 |
| `count_box` | 计数**并逐个框出** | 目标数 ≤12 的图 |
| `compose` | 画面里有哪几类目标、各多少 | 含 ≥2 类目标的图 |
| `compare` | 左右/上下哪边更密集 | 目标数 ≥4 的图 |
| `correct` | 给一句**错误陈述**让模型推翻 | 所有图。**三成给的是真陈述** |
| `temporal` | 异常出现在序列的哪个阶段 | 带轨迹的视频/多帧 |
| `negation` | 问画面里**没有**的东西 | 所有图 |
| `judge+locate(+why)` | 两轮 / 三轮追问 | 多轮占三成，其中三成半追到第三轮 |

`correct` 那三成真陈述不能省：全给假的，模型会学成「凡是被问就否定」，换个正确说法它照样推翻，
这比一味附和还糟。`temporal` 只答轨迹能证明的事——静态图问「什么时候开始的」，只能靠编。

描述与推理走 LLM 侧，见 [docs/12_description_design.md](docs/12_description_design.md)。

## 三种输入形态

| 形态 | 用在哪 | 输出字段 |
|---|---|---|
| `image` 单图 | MAR20 / Mendeley / FASDD / DOTA / VisDrone-DET 等静态数据集 | `images` |
| `video` 整段视频 | **ERA 的 5 秒片段**（人群运动、火焰跳动只有视频看得到，抽成静帧就丢了） | `videos` |
| `multi_image` 多帧序列 | 越界移动（需要逐帧对位判断跨越时机） | `images` |

图像样本与视频样本**分文件落盘**（`train.json` / `train_video.json`），因为在 LLaMA-Factory
里它们是两个数据集条目，`columns` 分别映射 `images` 与 `videos`，混在一个文件里会加载失败。
`merge_dataset.py` 会按这个分法生成 `dataset_info.json`，四个条目：
`military_anomaly` / `military_anomaly_val` / `military_anomaly_video` / `military_anomaly_video_val`。

图片就是图片、视频就是视频，中间不互转：ERA 的 5 秒片段整段进 `videos`，
静态数据集整张进 `images`，越界的多帧序列进 `images`（一条样本挂多个路径）。

## 输出目录与媒体归集

标注产物（json）和媒体文件（jpg/mp4）分开放：

```bash
# 标注产物落在哪: 第二个参数, 或环境变量 OUT_DIR
bash run_all.sh /path/to/数据根 /path/to/输出目录

# 媒体归集到语料库: 填 MEDIA_ROOT 就自动建目录、硬链文件、改写 json 里的路径
MEDIA_ROOT=/mnt/si003010kcx0/mmdata/data_process/corpus_media \
MEDIA_NAME=military_anomaly \
bash run_all.sh /path/to/数据根
```

归集后的结构，和 `corpus_media` 下已有的 `book/ journal/ video/` 一个分法：

```
corpus_media/military_anomaly/
  images/<数据源>/xxx.jpg      MAR20/ FASDD_UAV/ DOTA/ VisDrone/ DroneCrowd/ ...
  videos/<数据源>/xxx.mp4      ERA/ Drone-Anomaly/ ...
```

图片和视频分成两个二级目录，其下**再按数据源分目录**——不同数据集重名文件太多
（DroneCrowd 和 VisDrone 都有 `img0001.jpg`），不按源分会互相覆盖。

默认建**硬链接**：同一块盘上不占额外空间，删原文件也不影响；跨盘时自动退回复制
（`MEDIA_MODE=symlink|copy` 可改）。归集完 json 里的路径直接指向语料库位置，
训练时不用再拼相对路径。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 跑通全链路(合成 demo 数据，不需要下载任何图片)
python tools/adapters.py demo --n 40 --out data/interim/demo_scenes.jsonl
python tools/derive_events.py --scenes data/interim/demo_scenes.jsonl \
                              --out data/interim/demo_scenes_ev.jsonl
python tools/build_vqa.py --scenes data/interim/demo_scenes_ev.jsonl --out-dir data/vqa
python tools/check_dataset.py data/vqa/train.json
```

## 接入真实数据

```bash
# DOTA / FAIR1M / MAR20 (COCO 格式) → 派生"集结/机群/舰船集结"
python tools/adapters.py coco --ann DOTA_train.json --img-root data/raw/dota/images \
  --dataset DOTA-v2.0 --license academic-only --view satellite --out data/interim/dota.jsonl

# ERA / FASDD / UCF-Crime 抽帧 (按类别分目录) → 火光/烟雾/聚集/正常
python tools/adapters.py folder --root data/raw/era_frames \
  --dataset ERA --license research-only --view uav --out data/interim/era.jsonl

# VisDrone-MOT + 一条人工画的虚拟边界 → 派生"越界移动"
python tools/adapters.py visdrone-mot --seq-dir data/raw/visdrone/sequences/uav0000013_00000_v \
  --ann data/raw/visdrone/annotations/uav0000013_00000_v.txt \
  --boundary "0,700;960,660;1920,620" \
  --dataset VisDrone2019-MOT --license academic-only --out data/interim/visdrone.jsonl

# 合并 → 派生事件 → 生成 QA
cat data/interim/*.jsonl > data/interim/all_scenes.jsonl
python tools/derive_events.py --scenes data/interim/all_scenes.jsonl --out data/interim/all_ev.jsonl
python tools/build_vqa.py --scenes data/interim/all_ev.jsonl --out-dir data/vqa
python tools/check_dataset.py data/vqa/*.json --check-images
```

新数据源只需在 `tools/adapters.py` 里加一个函数，输出 `Scene` 即可，下游不用改。

## 训练

`merge_dataset.py` 会在输出目录下生成 `vqa/dataset_info.json`（内容随实际产出的文件变化），
把 LLaMA-Factory 的 `--dataset_dir` 指到那个目录即可；仓库里的 `data/dataset_info.json`
是一份可直接照抄的样例。

```bash
llamafactory-cli train configs/qwen2_5vl_lora_sft.yaml
```

## 目录结构

```
configs/ontology.yaml              异常本体(4 类 + 正常) + 判定规则的严格档与放宽档
configs/qwen2_5vl_lora_sft.yaml    LLaMA-Factory 训练配置示例
configs/prompts/                   33 份纯文本 prompt，与代码分离，服务器上可直接改
  system.txt                       训练数据里的 system prompt
  describe/<异常类>/<侧面>.txt      21 个描述侧面，各带 must-not 硬隔离与问法池
  ask/*.txt                        11 种题型各自的问法池
  _tools/*.txt                     描述生成、推理生成、六维 review、图像质检
tools/scene.py                     统一中间表示
tools/prepare.py                   数据集预处理统一入口(9 个数据集各一个子命令)
tools/ds/                          每个数据集的专用适配器 + 共享工具
tools/adapters.py                  通用适配器(COCO/YOLO/分类目录) + demo 生成器
tools/derive_events.py             规则派生异常事件标签
tools/screen.py                    质量筛选闸1-4 + 合并闸5 VLM 复核结果
tools/facets.py                    侧面与问法池的加载 + 自检(--check)
tools/build_vqa.py                 规则生成: 11 种题型 + 按质量分的配额筛选
tools/llm_qa.py                    按侧面生成描述与推理 → must-not 硬过滤 → 六维 review
tools/make_golden.py               自动构建 golden set(零人工) + 训练集排除清单
tools/merge_dataset.py             合并规则侧/LLM 侧 + 生成 dataset_info.json
tools/export_media.py              图片/视频归集到语料库目录并改写 json 路径
tools/check_dataset.py             数据体检(路径/坐标/分布/风格/轮数/正负比)
data/dataset_info.json             LLaMA-Factory 数据集注册
docs/                              数据源调研 + schema 设计
```

## 五个容易翻车的点

1. **负样本比例 ≥30%**。只喂异常样本，模型会对任何航拍图都报"发现集结"，误报比漏报更致命。`check_dataset.py` 会强制检查这一项。
2. **按源图/源视频切分 train/test**，不能按 QA 随机切。同段视频的相邻帧分落两侧会让指标严重虚高。
3. **必须有否定题**（问画面里没有的东西）。这是抑制 VLM 微调后"你问啥都说有"的唯一有效手段。
4. **校验要换一个模型**。同模型自查会系统性放过自己的错误——它按同样的方式理解图像，自然觉得自己是对的。
5. **盯住 verify 的 drop 率**。超过 15% 说明生成端在编造事实，回去收紧 prompt，别硬着头皮往下跑。

## 合规

本项目仅用于**监控画面异常事件识别**的研究与训练数据构建。所有数据源的许可条件见 [docs/01_data_sources.md 第 7 节](docs/01_data_sources.md#7-合规与许可注意事项)；每条 QA 都带 `source_dataset` 与 `license` 字段，发布时请按许可分层裁剪。MOCO 等申请制数据集须走完申请流程后方可使用。

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
  ┌── build_vqa.py   规则 QA (40%)  计数、grounding —— 答案是精确真值
  └── llm_qa.py      LLM 指令 QA (60%)  描述、推理、多轮、JSON 输出
        ↓  generate(t=0.8) → verify(t=0, 换模型) → pass/fix/drop
  LLaMA-Factory ShareGPT 指令数据
```

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
**任务类型定稿（5 类）、数量分配与数据可行性核对见 [docs/10_task_definition.md](docs/10_task_definition.md)**。

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

`data/dataset_info.json` 已注册好数据集，复制到 LLaMA-Factory 的 `data/` 目录（或用 `--dataset_dir` 指向本仓库 `data/`）：

```bash
llamafactory-cli train configs/qwen2_5vl_lora_sft.yaml
```

## 目录结构

```
configs/ontology.yaml              异常本体(10 类) + 自动判定规则参数
configs/qwen2_5vl_lora_sft.yaml    LLaMA-Factory 训练配置示例
configs/prompts/                   system / 质检 / 生成 / 校验 四份 prompt 模板
tools/scene.py                     统一中间表示
tools/prepare.py                   数据集预处理统一入口(9 个数据集各一个子命令)
tools/ds/                          每个数据集的专用适配器 + 共享工具
tools/adapters.py                  通用适配器(COCO/YOLO/分类目录) + demo 生成器
tools/derive_events.py             规则派生异常事件标签
tools/screen.py                    质量筛选闸1-4 + 合并闸5 VLM 复核结果
tools/build_vqa.py                 Scene → 9 类题型规则 VQA
tools/llm_qa.py                    FACTS 事实包 → LLM 指令型 VQA → 校验
tools/make_golden.py               自动构建 golden set(零人工) + 训练集排除清单
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

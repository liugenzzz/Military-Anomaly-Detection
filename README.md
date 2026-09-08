# Military Anomaly Detection — 多模态 VQA 数据集构建

用公开数据构建**军事异常检测**的多模态 VQA 数据集，训练一个能在**航拍/监控画面**中识别异常军事活动（集结、爆炸、烟雾、越界移动等）的视觉语言模型。训练框架为 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。

## 核心结论

**没有现成可用的「军事异常 VQA」数据集**，但完全可以构建。路线是四层拼装 + 规则化 QA 生成：

```
军事目标外观层   MAR20 / Mendeley-UAV军事目标 / MOCO(申请制)     → 认得出"是什么"
异常事件层       ERA / XD-Violence / FASDD / Drone-Anomaly       → 认得出"出事了"
航拍底座层       VisDrone / DroneCrowd / DOTA / FAIR1M / xView   → 视角、尺度、负样本
合成补充层       ARMA3 / Unreal 渲染                             → 补集结、越界等稀缺类
        ↓
  统一 Scene 中间表示 (tools/scene.py)
        ↓
  规则派生事件标签 (tools/derive_events.py)  ← 集结=密度聚类，越界=轨迹跨边界
        ↓
  9 类题型 QA 生成 (tools/build_vqa.py)      → LLaMA-Factory ShareGPT 格式
```

完整数据源清单、许可说明与推荐取样配比见 **[docs/01_data_sources.md](docs/01_data_sources.md)**；
异常本体与 VQA schema 设计见 **[docs/02_ontology_and_schema.md](docs/02_ontology_and_schema.md)**；
**按优先级排好的下载清单、以及视频要不要处理，见 [docs/03_acquisition_checklist.md](docs/03_acquisition_checklist.md)**。

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
tools/scene.py                     统一中间表示
tools/adapters.py                  各公开数据集 → Scene
tools/derive_events.py             规则派生异常事件标签
tools/build_vqa.py                 Scene → 9 类题型 VQA
tools/check_dataset.py             数据体检(路径/坐标/分布/正负比)
data/dataset_info.json             LLaMA-Factory 数据集注册
docs/                              数据源调研 + schema 设计
```

## 三个容易翻车的点

1. **负样本比例 ≥30%**。只喂异常样本，模型会对任何航拍图都报"发现集结"，误报比漏报更致命。`check_dataset.py` 会强制检查这一项。
2. **按源图/源视频切分 train/test**，不能按 QA 随机切。同段视频的相邻帧分落两侧会让指标严重虚高。
3. **必须有否定题**（问画面里没有的东西）。这是抑制 VLM 微调后"你问啥都说有"的唯一有效手段。

## 合规

本项目仅用于**监控画面异常事件识别**的研究与训练数据构建。所有数据源的许可条件见 [docs/01_data_sources.md 第 7 节](docs/01_data_sources.md#7-合规与许可注意事项)；每条 QA 都带 `source_dataset` 与 `license` 字段，发布时请按许可分层裁剪。MOCO 等申请制数据集须走完申请流程后方可使用。

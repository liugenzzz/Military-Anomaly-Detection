# 下载清单：8 个源数据集 + 3 个工具性数据集

链接均已核实。标注「⚠️」的是需要注意的坑。

---

## A 组 · 骨架

### 1. ERA（航拍事件视频）
- **主页**：<https://lcmou.github.io/ERA_Dataset/>
- **备用**：<https://ieee-dataport.org/open-access/era-dataset>
- **内容**：2,864 段视频，每段 5 秒 / 30fps / 640×640，25 类事件
- **要什么**：全部下载。`fire` `conflict` `parade` `constructing` 是异常类，`non-event` 是负样本
- **处理**：每段抽 3 帧（首/中/尾）→ 约 8.5k 图
- ⚠️ 视频源自 YouTube，仅可按原协议用于研究

### 2. CapERA（ERA 的人工 caption）
- **仓库**：<https://github.com/yakoubbazi/CapEra>
- **内容**：给 ERA 全部 2,864 段视频每段 5 条人工 caption
- ⚠️ **只有标注文件，视频要用 ERA 的**。所以 ERA 必须先下
- **价值**：零成本拿到人工描述标注，也是 golden set 的 L2 层主力来源

### 3. HIVAU-70k（层级化异常理解指令）
- **官方仓库**：<https://github.com/pipixin321/HolmesVAU>（CVPR 2025 Highlight）
- **HuggingFace 预处理版**：`backseollgi/HIVAU-70k_UCF-Crime`、`backseollgi/HIVAU-70k_XD-Violence`
- **内容**：7 万+ 条指令标注，clip / event / video 三粒度，含判断、描述、因果分析
- ⚠️ **仓库里是标注 JSON，视频本体要另外下**：
  - UCF-Crime：<https://www.crcv.ucf.edu/projects/real-world/>（1,900 段 / 128 小时）
  - XD-Violence：<https://roc-ng.github.io/XD-Violence/>（4,754 段 / 217 小时）
- 💡 **先试 HuggingFace 的预处理版**，它已经切好片段。够用就不必下 345 小时的原始视频，能省几百 GB 磁盘和大量下载时间

### 4. VisDrone（航拍底座 + 轨迹）
- **仓库**：<https://github.com/VisDrone/VisDrone-Dataset>
- **要两个子集**：
  - `VisDrone2019-DET`（检测，train/val/test-dev）—— 提供航拍 bbox 底座与负样本
  - `VisDrone2019-MOT`（多目标跟踪）—— **提供 track_id，越界移动只能从这里派生**
- **处理**：MOT 按 `--stride 30` 抽关键帧，配合人工画的虚拟边界线

---

## B 组 · 军事属性

### 5. Mendeley UAV 多类军事目标
- **地址**：<https://data.mendeley.com/datasets/9z7yrcrpjk/1>
- **内容**：7,985 张标注图 / 14,018 实例，4 类：tank、drone、people、soldier，航拍视角
- **价值**：**唯一「军事目标 + 航拍视角 + bbox」三者齐全**的公开集
- ⚠️ 含合成增强数据，筛选时留意是否需要区分对待

### 6. MAR20（军机遥感识别）
- **官方**：<https://gcheng-nwpu.github.io/>（西北工业大学）
- **YOLO 镜像**：<https://universe.roboflow.com/mar20/mar20-s3e1w>
- **内容**：3,842 张高分辨率遥感图 / 22,341 实例 / 20 类军机，取自全球 60 个军用机场（Google Earth），HBB + OBB 双标注
- **用途**：机群集结、机场活动异常

---

## C 组 · 专项

### 7. FASDD_UAV（烟雾火焰，无人机视角）
- **地址**：<https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda>
- **只下 `FASDD_UAV.zip`**，另两个（`FASDD_CV` 通用视角、`FASDD_RS` 遥感）暂不需要
- **内容**：36,308 个火焰实例 + 17,222 个烟雾实例
- 💡 标注同时提供 **YOLO / VOC / COCO / TDML** 四种格式，直接取 COCO 版喂 `adapters.py coco`

### 8. DOTA v2.0（遥感俯视，集结派生）
- **地址**：<https://captain-whu.github.io/DOTA/dataset.html>
- **内容**：188,282 实例 / 15 类 / OBB 标注
- **要什么**：`plane`、`ship`、`large-vehicle`、`small-vehicle`、`helicopter` 这几类，其余（球场、泳池等）可在归一化时过滤
- ⚠️ 原图尺寸极大（可达 20000×20000），**必须先切片**（官方提供 `DOTA_devkit` 的 `ImgSplit`），建议 1024×1024 / overlap 200

### 9. Drone-Anomaly（同场景正常/异常配对）
- **仓库**：<https://github.com/Jin-Pu/Drone-Anomaly>
- **内容**：7 个场景，37 训练 / 22 测试视频序列，51,635 + 35,853 帧，640×640
- **价值**：**负样本金矿**——同一场景的正常段与异常段成对，训练"不乱报警"全靠它

---

## 工具性数据集（不当训练源，但要用）

| 数据集 | 地址 | 用途 |
|---|---|---|
| **UCA** | <https://xuange923.github.io/Surveillance-Video-Understanding> | 用它 0.1 秒精度的事件边界，从异常视频切出干净正常段做困难负样本 |
| **VRSBench** | <https://huggingface.co/datasets/xiang709/VRSBench> | 按 10% 混入训练防遗忘，顺带提供 grounding 范式 |
| **MOCO** | <https://github.com/Panlizhi/MOCO> | **申请制**，现在就发申请，批下来直接加入 B 组 |

---

## 下载顺序与磁盘预算

按这个顺序下，任何一步卡住都不影响前面的成果：

| 顺序 | 数据集 | 体量估计 | 说明 |
|---|---|---|---|
| 1 | Mendeley 军事目标 | 小（< 5 GB） | 最快见效，立刻能跑通 adapters → QA 全链路 |
| 2 | MAR20 | 小 | 同上 |
| 3 | ERA + CapERA | 中 | 骨架第一块，抽完帧就能出第一批带描述的数据 |
| 4 | FASDD_UAV | 中 | 烟雾类一次到位 |
| 5 | VisDrone DET + MOT | 中 | 越界派生依赖它 |
| 6 | Drone-Anomaly | 中 | 负样本 |
| 7 | DOTA v2.0 | 大（切片后更大） | 留到最后，切片耗时 |
| 8 | HIVAU-70k（先 HF 预处理版） | 视是否下原始视频而定 | **先别下 UCF-Crime/XD-Violence 原始视频**，345 小时，几百 GB |

**磁盘预算**：不下 UCF-Crime/XD-Violence 原始视频的话，全部约 **150–250 GB**；下了则要预留 **500 GB 以上**。远程环境磁盘有限，建议 1–7 先跑通，HIVAU 的视频本体放到最后按需补。

下载建议用 `aria2c -x 16 -s 16` 或 `wget -c`（支持断点续传），大文件被中断重来很浪费时间。

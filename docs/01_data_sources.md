# 数据源调研：军事异常检测多模态 VQA

> 结论先说：**没有现成的「军事异常 VQA」数据集**，但可以用「军事目标外观数据 + 通用异常/事件数据 + 航拍监控底座数据 + 少量合成数据」四层拼出来，再用规则 + VLM 自动生成 QA 对。下面是逐层的候选清单。

---

## 0. 目标场景 → 数据缺口对照

| 目标异常场景 | 现成数据情况 | 主要来源策略 |
|---|---|---|
| 兵力/装备集结 | ❌ 无直接标注 | 遥感检测数据（DOTA/FAIR1M/xView）车辆框 **密度聚类** 自动派生 + 合成 |
| 爆炸 / 火光 | ✅ 有（非军事语境为主） | ERA、XD-Violence、UCF-Crime、FASDD |
| 烟雾 | ✅ 较充分 | FASDD_UAV、FLAME/FLAME2、Boreal Forest Fire |
| 越界移动 / 闯入禁区 | ❌ 无直接标注 | VisDrone/UAVDT/DroneCrowd **跟踪轨迹 + 人工虚拟边界** 自动派生 |
| 人群聚集 | ✅ 有 | DroneCrowd、VisDrone-CC、ERA(parade/protest) |
| 军事目标外观（坦克/军机/士兵） | ✅ 有 | MAR20、Kaggle 军机、Mendeley UAV 军事目标、MOCO(需申请) |
| 车队机动 / 阵地构筑 | ⚠️ 弱 | ERA(constructing)、遥感时序 + 合成 |
| 正常样本（负样本） | ✅ 充分 | VisDrone、UAVDT、DOTA 普通场景 |

---

## 1. 军事目标外观层（提供"军味"）

### 1.1 MAR20 —— 军用飞机遥感识别 ⭐推荐
- 20 类军机、3,842 图、22,341 实例，HBB + OBB 双标注，目前最大的公开军机遥感识别集。
- 论文：<https://www.ygxb.ac.cn/en/article/doi/10.11834/jrs.20222139/>
- 镜像/YOLO 版：<https://universe.roboflow.com/mar20/mar20-s3e1w>
- 用途：机场停机坪目标识别、**机群集结**判定、计数题、grounding 题。

### 1.2 Kaggle Military Aircraft Detection Dataset ⭐推荐
- ~100 类军机、带 bbox，社区维护活跃，下载无门槛。
- <https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset>
- HF 镜像：<https://huggingface.co/datasets/a2015003713/military-aircraft-detection-dataset>
- 注意：以侧视/斜视照片为主，需筛选出俯视/航拍子集，或仅用于属性识别题。

### 1.3 Mendeley 多类 UAV 军事目标数据集 ⭐强推荐（最贴合航拍视角）
- 7,985 张标注图、14,018 实例，4 类：**tank / drone / people / soldier**，航拍视角，含合成增强数据。
- <https://data.mendeley.com/datasets/9z7yrcrpjk/1>
- 用途：低空 UAV 视角的军事目标检测 → 直接派生「装甲集群/人员集结」QA。

### 1.4 MOCO —— 军事图像描述（唯一的军事图文数据）⚠️需申请
- 7,449 图 / 37,245 条 caption，真实战场 UAV/UGV 低空视角，专为 Military Image Captioning 设计。
- 仓库：<https://github.com/Panlizhi/MOCO> · 论文：<https://www.mdpi.com/2504-446X/8/9/421>
- **出于伦理审查要求需申请获取**，不能直接下载。建议尽早提交申请——这是最接近你目标的图文对，可直接改写为描述题/推理题。

### 1.5 Roboflow Universe 军事类数据集（补充，质量参差）
- military-vehicle：<https://universe.roboflow.com/vgtu-n8zmy/military-vehicle>
- military object detection（tank/soldier/militarycar）：<https://universe.roboflow.com/yolo-datasets-ymdve/military-object-detection-uxkcn>
- 建议：只作补充，务必人工抽检；很多是网图爬取，分辨率与标注质量不稳定。

---

## 2. 异常事件层（提供"异常"语义）

### 2.1 ERA Dataset —— 航拍事件识别 ⭐⭐最推荐
- 2,864 段 5 秒航拍视频，25 类事件，含 **fire、conflict、police chase、parade/protest、constructing、post-earthquake、非事件类**。
- 主页：<https://lcmou.github.io/ERA_Dataset/> · <https://ieee-dataport.org/open-access/era-dataset> · 论文：<https://arxiv.org/abs/2001.11394>
- 为什么最推荐：**唯一同时满足「航拍视角 + 事件级语义 + 含 non-event 负样本」的公开集**。抽帧后可直接映射到你的异常本体：fire→火光/烟雾、conflict→冲突、parade→集结、constructing→阵地构筑、non-event→正常。

### 2.2 XD-Violence / UCF-Crime —— 爆炸与暴力事件
- XD-Violence：4,754 段、217 小时，6 类（含 **Explosion / Riot / Shooting / Car accident**），有视频级 + 帧级标签。
- UCF-Crime：1,900 段、128 小时监控视频，13 类异常（含 **Explosion / Arson / Shooting**）。Kaggle 镜像：<https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf/data>
- 用途：**监控视角**的爆炸/枪击帧，补足 ERA 的地面监控缺口。注意这两个是民用治安场景，用于「爆炸/烟雾」的视觉特征迁移，不要直接标成军事事件。

### 2.3 火焰烟雾专项
- **FASDD**（Flame And Smoke Detection Dataset）：12 万+ 图，其中 **FASDD_UAV 子集 25,097 张为无人机视角** ⭐。<https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda> · 论文：<https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2347922>
- **FLAME 2**：航拍多光谱（RGB+IR）火灾数据，可用于红外/夜视场景泛化。<https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset>
- **Boreal Forest Fire**：UAV 采集，含 bbox + **烟雾分割掩码**，Scientific Data 2025。<https://www.nature.com/articles/s41597-025-05634-0>
- 用途：烟雾/火光的形态学特征是通用的，训练出「识别烟柱/爆燃」的能力后，在军事语境里靠 prompt 与少量真实样本对齐即可。

### 2.4 航拍异常检测专项
- **Drone-Anomaly**：7 个场景、37 训练/22 测试视频序列，51,635 训练帧 + 35,853 测试帧，10 类异常事件。<https://github.com/Jin-Pu/Drone-Anomaly>
- **UIT-ADrone**：51 段视频、206K 帧、1080p，环岛交通异常，10 类异常。<https://ieeexplore.ieee.org/document/10158513/>
- 用途：提供「**什么叫异常**」的范式样本 —— 正常/异常成对的同场景数据，对训练模型不乱报警特别有价值。

---

## 3. 航拍/监控底座层（提供视角、尺度、负样本）

| 数据集 | 规模 | 关键价值 | 链接 |
|---|---|---|---|
| **VisDrone** | 10 城、检测/跟踪/计数多任务 | 小目标航拍标准底座，**含 MOT 轨迹** → 越界判定 | <https://github.com/VisDrone/VisDrone-Dataset> |
| **DroneCrowd** | 112 段、33,600 帧 1080p、480 万头部标注、20,800 条轨迹 | **人群聚集/集结**的规模与轨迹依据 | <https://github.com/VisDrone/DroneCrowd> |
| **DOTA** | 188,282 实例、15 类、OBB | 遥感俯视，plane/ship/large-vehicle → **装备列阵集结** | <https://captain-whu.github.io/DOTA/> |
| **FAIR1M** | ~15,000 图、100 万+ 实例、5 大类 37 子类、0.3–0.8m | 细粒度飞机/舰船/车辆，**集结判定最佳底座** | <https://www.gaofen-challenge.com/benchmark> |
| **xView** | 100 万+ 实例、60 类、0.3m WorldView-3（NGA 发布） | 含 aircraft hangar / cargo plane / 各类车辆 | <https://docs.ultralytics.com/datasets/detect/xview> |

> 这一层本身没有"异常"标签，但它是**自动派生异常标注**的原料：车辆框做密度聚类 → 集结；轨迹跨越虚拟边界 → 越界。本仓库 `tools/derive_events.py` 实现了这两条规则。

---

## 4. 合成数据层（补稀缺类别）

真实的「集结 / 越界 / 阵地构筑」几乎没有公开标注，合成是现实可行的补充路径：

- **ARMA 3 合成遥感军事目标**（2025）：用 ARMA3 的 Real Virtuality 4 引擎批量随机生成人员/车辆并多角度出图，专门针对遥感军事目标检测。<https://www.researchgate.net/publication/388747706_Exploiting_Arma_3_to_Construct_Synthetic_Data_for_Military_Target_Detection_on_Remote_Sensing_Imagery>
- **G-MAD**：基于游戏引擎的多视角 RGB-T 航拍目标检测数据生成框架。<https://arxiv.org/html/2607.19942>
- **Unreal Engine / AirSim**：可编程相机高度与俯仰角，天然带 pixel-perfect 标注。<https://www.sciencedirect.com/science/article/abs/pii/S0921889023001033>
- 综述参考：<https://arxiv.org/pdf/2112.12252>（合成数据在 UAV 目标检测中的作用）

**建议配比**：合成数据 ≤ 30%，且必须与真实数据混训，否则模型会学到渲染风格而非语义。

---

## 5. VQA 数据构建方法学参考（照着做）

| 参考工作 | 可借鉴的点 | 链接 |
|---|---|---|
| **VRSBench** | 遥感 VQA 基准的题型设计（caption / VQA / grounding 三合一），GPT-4V 生成 + 人工校验流程 | <https://arxiv.org/html/2406.12384v1> |
| **RSVQA / RSIVQA** | 从已有检测/分割标注**规则化**生成 QA（存在、计数、比较），零人工成本 | RSIVQA ~37k 图 / 110k QA |
| **EarthVQA** | 20 万+ 问题的大规模城市遥感 VQA，含推理型问题 | — |
| **RSVLM-QA** | 用 GPT-4.1 从 WHU/LoveDA/iSAID 的分割标注自动生成 caption + 空间关系 + VQA | <https://arxiv.org/abs/2508.07918> |

**核心方法论**：`已有 bbox/mask 标注` → `规则模板生成事实型 QA（存在/计数/位置/属性）` → `VLM 基于标注 + 图像生成描述型与推理型 QA` → `人工抽检 5–10%`。这条路线成本最低、事实一致性最好，本仓库就是按它实现的。

---

## 6. 推荐的最小可行组合（MVP，约 3–5 万 QA）

| 层 | 数据集 | 取样量 | 覆盖的异常类 |
|---|---|---|---|
| 事件 | **ERA**（抽帧，每段 3 帧） | ~8,500 图 | 火光、冲突、集结/游行、构筑、正常 |
| 烟雾 | **FASDD_UAV** | ~6,000 图 | 烟雾、火光 |
| 军事目标 | **Mendeley UAV 军事目标** | ~8,000 图 | 装甲/人员目标识别 |
| 军机 | **MAR20** | ~3,800 图 | 机群集结、机场活动 |
| 底座+派生 | **VisDrone(MOT) + DroneCrowd** | ~6,000 帧 | 越界移动、人群聚集 |
| 底座+派生 | **DOTA / FAIR1M** | ~4,000 图 | 装备列阵集结 |
| 爆炸 | **XD-Violence / UCF-Crime** 抽帧 | ~2,000 帧 | 爆炸 |
| 负样本 | VisDrone / DOTA 普通场景 + ERA non-event | **≥30% 总量** | 正常 |

> **负样本比例是这个项目的生死线**。只喂异常样本，模型会对任何航拍图都喊"发现异常集结"，实测上比漏报更致命。

---

## 7. 合规与许可注意事项

| 数据集 | 许可 | 注意 |
|---|---|---|
| MOCO | 申请制 + 伦理审查 | 必须走申请流程，不可转授 |
| xView | 非商业研究许可（NGA） | 商用需另行确认 |
| DOTA / FAIR1M / MAR20 | 学术研究用途 | 论文引用义务 |
| UCF-Crime / XD-Violence | 学术研究 | 含真实暴力内容，注意数据脱敏与人员隐私 |
| Kaggle / Roboflow 社区集 | 逐个查看，多为 CC BY 4.0 / 未声明 | 未声明许可的**不要进最终发布集**，只用于内部实验 |
| ERA | 视频源自 YouTube | 仅可按原协议用于研究 |

统一做法：把每条 QA 的 `source_dataset` 与 `license` 字段带到最终 JSON 里（本仓库 schema 已包含），发布时按许可分层裁剪。

# 数据获取清单（按优先级排序）

> 目标能力拆成三项，**没有任何单一数据集能同时满足三项**，必须组合：
>
> | 能力要求 | 需要的标注类型 | 谁能提供 |
> |---|---|---|
> | R1 识别异常（集结/爆炸/烟雾/越界） | 帧级或视频级**异常标签** | ERA、UCF-Crime、XD-Violence、FASDD、Drone-Anomaly |
> | R2 多模态描述（文字 + **图像区域**） | **bbox / mask** | DOTA、FAIR1M、VisDrone、MAR20、Mendeley-UAV、VRSBench |
> | R3 生成异常说明 | **自然语言描述 / 因果标注** | CapERA、UCA、HIVAU-70k、CUVA、MOCO |
>
> R2 是最容易被忽略的一项：**只有 bbox 才能训出"异常在 [x1,y1,x2,y2] 这个区域"**。
> ERA/UCF-Crime 这类只有视频级标签的数据，无论多大都训不出区域定位能力，只能贡献判定与描述。

---

## P0 —— 立刻就下，构成骨架（无门槛、质量可靠）

| # | 数据集 | 规模 | 提供 | 视角 | 链接 |
|---|---|---|---|---|---|
| 1 | **ERA** | 2,864 段 5s 航拍视频、25 类事件（含 fire/conflict/parade/constructing/**non-event**） | R1 | UAV | <https://lcmou.github.io/ERA_Dataset/> |
| 2 | **CapERA** ⭐新发现 | 给 ERA 全部 2,864 段视频补了**每段 5 条人工 caption**（含事件、目标、地点、动作、数量、时间） | **R3** | UAV | <https://github.com/yakoubbazi/CapEra> |
| 3 | **HIVAU-70k** ⭐新发现 | **7 万+ 条层级化指令标注**（clip/event/video 三粒度），覆盖 UCF-Crime + XD-Violence，含判断题/描述题/**因果分析题** | **R1+R3** | CCTV | <https://github.com/pipixin321/HolmesVAU> · [镜像](https://github.com/jungseoik/HIVAU-70k) |
| 4 | **UCA** ⭐新发现 | UCF-Crime 的 1,854 段视频、23,542 句描述、110.7 小时，**时间戳精确到 0.1 秒** | **R3** | CCTV | <https://xuange923.github.io/Surveillance-Video-Understanding> |
| 5 | **VisDrone**（DET + MOT） | 10 城航拍，bbox + **轨迹 ID** | **R2** + 越界派生 | UAV | <https://github.com/VisDrone/VisDrone-Dataset> |
| 6 | **DOTA v2.0** | 188,282 实例、15 类、OBB | **R2** + 集结派生 | 卫星 | <https://captain-whu.github.io/DOTA/> |
| 7 | **FASDD** | 12 万+ 图，**FASDD_UAV 子集 25,097 张为无人机视角**，带 bbox | R1+**R2** | UAV | <https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda> |
| 8 | **Mendeley UAV 军事目标集** | 7,985 图 / 14,018 实例，tank / drone / people / **soldier**，带 bbox | **R2**（唯一的军事目标 bbox 航拍源） | UAV | <https://data.mendeley.com/datasets/9z7yrcrpjk/1> |

**P0 的组合逻辑**：ERA+CapERA 给「航拍异常事件 + 人话描述」；HIVAU-70k+UCA 给「监控异常 + 带时间戳的因果说明」；VisDrone/DOTA/FASDD/Mendeley 给「可定位的区域」。
这 8 个下完，R1/R2/R3 三项就都有底了。

---

## P1 —— 补场景与军事属性

| # | 数据集 | 规模 | 提供 | 链接 |
|---|---|---|---|---|
| 9 | **XD-Violence** | 4,754 段 / 217h，含 Explosion / Riot / Shooting，视频级+帧级标签 | R1 爆炸 | <https://roc-ng.github.io/XD-Violence/> |
| 10 | **UCF-Crime** | 1,900 段 / 128h，13 类异常（Explosion/Arson/Shooting） | R1 爆炸；且是 UCA/HIVAU 的**视频本体，必须下** | [Kaggle 镜像](https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf/data) |
| 11 | **MAR20** | 20 类军机、3,842 图、22,341 实例，HBB+OBB | R2 机群集结 | [论文](https://www.ygxb.ac.cn/en/article/doi/10.11834/jrs.20222139/) · [YOLO 版](https://universe.roboflow.com/mar20/mar20-s3e1w) |
| 12 | **FAIR1M** | ~15,000 图、100 万+ 实例、37 子类、0.3–0.8m | R2 细粒度集结 | <https://www.gaofen-challenge.com/benchmark> |
| 13 | **Drone-Anomaly** | 7 场景、51,635 训练帧 + 35,853 测试帧，10 类异常 | R1 + **同场景正常/异常配对** | <https://github.com/Jin-Pu/Drone-Anomaly> |
| 14 | **DroneCrowd** | 112 段、33,600 帧 1080p、480 万头部标注、20,800 条轨迹 | R2 人员聚集 | <https://github.com/VisDrone/DroneCrowd> |
| 15 | **CUVA** | 42 个异常子类，每条含 **what / why / how 三段人工标注** | R3 推理题范式 | <https://github.com/fesvhtr/CUVA> |
| 16 | **VRSBench** | 29,614 遥感图、**52,472 条 object refer**、310 万 QA | **R2 区域指代**的现成范式，可直接混入训练 | <https://huggingface.co/datasets/xiang709/VRSBench> |

---

## P2 —— 申请制 / 选配

| # | 数据集 | 说明 |
|---|---|---|
| 17 | **MOCO** | 7,449 图 / 37,245 条军事 caption，真实战场 UAV/UGV 视角。**申请制（伦理审查）**，最贴合目标，建议现在就发申请。<https://github.com/Panlizhi/MOCO> |
| 18 | **xView** | 100 万+ 实例、60 类、0.3m WorldView-3（NGA）。非商业研究许可。<https://docs.ultralytics.com/datasets/detect/xview> |
| 19 | **GeoChat_Instruct** | 318k 遥感多模态指令，含区域描述与 visual grounding。可作为**通用遥感能力的保底混入数据**，防灾难性遗忘。<https://github.com/mbzuai-oryx/geochat> |
| 20 | **FLAME 2 / Boreal Forest Fire** | 航拍多光谱火灾 + 烟雾分割掩码，用于红外/夜视泛化。 |

**MilData（MilChat 论文，2025）**：军事遥感 MLLM，专注隐蔽军事设施（导弹发射场等），用 CoT 标注 + GRPO，宣称 80%+ recall / 98% precision。**数据集未公开释出**，但方法论很值得抄——它明确把「抑制民用场景假阳性」作为核心优化目标，和我们对负样本的判断一致。<https://www.alphaxiv.org/abs/2505.07984v1>

---

## 关于视频：要处理，但只处理两件事

**结论：绝大部分工作是抽帧，不是训视频模型。**

### 需要做的

**① 抽帧（80% 的工作量）**
ERA / UCF-Crime / XD-Violence / FASDD 视频源都要抽成关键帧。策略：

- **异常段密采**（1–2 fps），**正常段稀采**（0.2 fps）
- 每段视频至少保留 1 帧正常段 —— 见下方"最优质负样本"
- 抽完做**近重复去除**（相邻帧 pHash 汉明距离 < 5 的丢弃），否则训练集里全是几乎一样的图，等于白刷数量

**② 时序类异常保留多帧序列（20% 的工作量）**
**越界移动、车队机动这两类，单帧根本判不出来** —— 一张静止画面里你看不出车是在越界还是在停车。两个方案：

- **方案 A（推荐）：多图输入**。取 3–4 帧组成序列，Qwen2.5-VL 原生支持多图，LLaMA-Factory 的 `images` 字段可以放多张，prompt 写成 `<image><image><image>这三帧按时间顺序拍摄，目标是否发生越界移动？`
- **方案 B：单帧 + 边界可视化**。把禁区边界线画进图里，配合轨迹箭头叠加，退化成单帧任务。工程更简单，但模型学到的是"看线"而不是"看运动"，泛化差。

爆炸/烟雾/集结这三类**单帧就够**，不用碰视频时序。

### 不需要做的

- ❌ 不需要训练视频 encoder、不需要光流、不需要 3D CNN
- ❌ 不建议用 LLaMA-Factory 的 `videos` 视频输入字段：显存开销比多图大得多，而你的异常类型里只有两类真正需要时序，性价比不划算

### 抽帧带来的最大红利：最优质的负样本来源

**从异常视频的正常段里抽负样本。** UCF-Crime 测试集有帧级标注，UCA / HIVAU-70k 给出了精确到 0.1 秒的事件时间边界 —— 这意味着你能精确切出「同一段监控视频里，爆炸发生之前的那些帧」。

这种负样本的价值远高于随便找的正常图：**背景、光照、摄像机、场景全部相同，唯一的差别就是异常本身**。模型没法靠"这个场景看起来就危险"蒙对，只能真的去看有没有火光。这是把误报率压下来最有效的一招，比单纯堆负样本数量管用得多。

建议负样本构成：

| 来源 | 占比 | 作用 |
|---|---|---|
| 异常视频的正常段（同场景困难负样本） | 40% | 压误报，最关键 |
| ERA non-event + Drone-Anomaly 正常序列 | 30% | 航拍常态分布 |
| VisDrone / DOTA 普通城市与郊野场景 | 30% | 视角与尺度多样性 |

总量目标：**正常样本 ≥ 全体的 30%**。

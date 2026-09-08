# 源数据集最终选型（决策记录）

前面调研列了 20 个候选，**实际只用 8 个**。清单越长越容易陷入"什么都想要"，最后哪个都没做透。以下是定稿，附明确的不选理由。

## 选定：8 个源数据集

### A 组 · 骨架（三个，构成主体）

| # | 数据集 | 取样量估计 | 不可替代的原因 |
|---|---|---|---|
| 1 | **ERA + CapERA** | 2,864 段 × 3 帧 ≈ 8.5k 图 | 唯一「航拍视角 + 事件级语义 + 自带 non-event 负样本 + 人工 caption」四合一。CapERA 与 ERA 同批视频，等于白送描述标注 |
| 2 | **HIVAU-70k**（含 UCF-Crime / XD-Violence 视频本体） | 7 万条指令 ≈ 2 万图 | 已是指令格式的异常理解数据，含判断/描述/**因果分析**三类。省掉最贵的一段人工成本 |
| 3 | **VisDrone**（DET + MOT） | ≈ 16k 图 | 航拍底座 + **轨迹标注**。越界移动只能从这里派生，没有替代品 |

### B 组 · 军事属性（两个）

| # | 数据集 | 取样量估计 | 原因 |
|---|---|---|---|
| 4 | **Mendeley UAV 军事目标** | ≈ 8k 图 | **唯一的「军事目标 + 航拍视角 + bbox」三者齐全**的公开集（tank/drone/soldier/people） |
| 5 | **MAR20** | 3.8k 图 | 军机遥感，HBB+OBB 双标注，机群集结与机场活动的主力 |

### C 组 · 专项补充（三个）

| # | 数据集 | 取样量估计 | 原因 |
|---|---|---|---|
| 6 | **FASDD_UAV** | 25k 中抽 8k | 烟雾/火焰，无人机视角，带 bbox |
| 7 | **DOTA v2.0** | 切片后取 5k | 遥感俯视，车辆/飞机/舰船密集，集结派生主力 |
| 8 | **Drone-Anomaly** | ≈ 6k 帧 | **同场景正常/异常成对**——负样本金矿，训练"不乱报警"全靠它 |

**合计约 5 万张图 → 筛选后约 3.5 万 → QA 约 30–40 万条。**

### 另外两类不算"源数据集"，但要用

- **UCA**：不当训练数据，只当**工具**——用它精确到 0.1 秒的事件边界，从异常视频里切出干净的正常段做困难负样本
- **VRSBench / GeoChat_Instruct**：不当源数据，按 **10% 比例混入训练**防灾难性遗忘，顺便提供 grounding 范式
- **MOCO**：现在就发申请，批下来直接加入 B 组，不阻塞当前进度

---

## 明确不选，以及为什么

| 数据集 | 判决 | 理由 |
|---|---|---|
| **Kaggle 军机（103 类）** | ❌ 不选 | 预期保留率仅 30–40%，绝大部分是地面平视的展会/机场照，与航拍视角不符。清洗成本高于收益 |
| **Roboflow 社区军事集** | ❌ 不选 | 网图爬取，标注质量不可控，多数无许可声明，不能进最终发布集 |
| **FAIR1M** | ⏸ 暂缓 | 与 DOTA 同为高分辨率遥感、**同源影像重叠风险高**，二选一即可。DOTA 工具生态更成熟。除非确实需要 37 个细粒度子类 |
| **xView** | ⏸ 暂缓 | 非商业研究许可，且 0.3m 卫星视角与你的「航拍/监控」目标场景差异大。要做卫星方向再加 |
| **CUVA** | 📖 只借鉴 | 42 个子类全是民用事故，军事相关性低。**只抄它 what/why/how 的三段式标注范式**，不用它的数据 |
| **FLAME 2 / Boreal Forest Fire** | ⏸ 暂缓 | 烟雾已有 FASDD_UAV 覆盖。等要做红外/夜视泛化时再加 |
| **ARMA3 / Unreal 合成数据** | ⏸ 二期 | 真实数据还没跑通就上合成，会分不清问题出在数据还是渲染域差。**第一版发布后再说** |

---

## 质量保证：在五道闸之上再加三条

前面 [04 号文档](04_quality_screening.md) 的五道闸是流程，这里是**验收标准**——没有验收标准的筛选等于没筛。

### 1. 用 embedding 去重替换 dHash 去重

`screen.py` 目前用 dHash（像素级），能抓抽帧产生的近重复，但**抓不到语义重复**——同一片区域的卫星图经过不同裁剪、不同年份采集，dHash 完全看不出来，但对模型来说就是同一个场景。

改用 CLIP embedding + 余弦相似度聚类。这件事 FiftyOne 已经做好了（见下节），不必自己写。

### 2. 人工抽检要分层，且要量化

- **分层抽检**，不要随机抽：按 `(数据源 × 异常类 × 题型)` 分层，每格至少 20 条。随机抽会让占比小的类别（如 fortification）完全抽不到
- **两人独立标注同一批 200 条**，算 Cohen's kappa。**kappa < 0.7 说明标注标准本身有歧义**，此时应该先回去改本体定义，而不是继续标
- **错误率 > 5% 的数据源整批打回重筛**，不要逐条修

### 3. 建 golden set，且永不进训练

从筛选后的数据里挑 500–1000 条，人工逐条精修，**单独存放，永远不进训练集**。用途：

- 每次改筛选阈值、改 prompt、换生成模型后，在 golden set 上跑一遍对比
- 最终指标只报 golden set 上的结果

没有这个集合，你后面所有"这次改进有没有效果"的判断都是靠感觉。

---

## 框架选型：三段各用各的，不要指望一个框架通吃

调研结论是**没有任何一个开源框架能覆盖「异构数据集归一 → 质量筛选 → 事实约束下的 LLM 生成 → LLaMA-Factory 格式」全流程**，也不该指望有。分段选型：

### 质量筛选段：FiftyOne ⭐ 建议引入

<https://github.com/voxel51/fiftyone> · [Brain 文档](https://docs.voxel51.com/brain/index.html)

能直接替掉 `screen.py` 一半的工作，而且做得更好：

| 能力 | 对应我们的闸 | 优势 |
|---|---|---|
| `compute_similarity` + 近重复检测 | 闸 3 | **embedding 级**，能抓语义重复，远强于 dHash |
| `compute_uniqueness` | 闸 3 | 按独特性排序，可直接做代表性子集采样 |
| `compute_mistakenness` | 闸 2 | 标签错误检测，需要模型 logits |
| exact duplicate detection | 闸 3 | 跨数据集精确去重 |
| **可视化 App** | 全部 | **这条价值最大**——纯脚本筛完你根本不知道筛掉的对不对，FiftyOne 能让你把 3000 张被剔除的图铺在屏幕上肉眼扫一遍 |

原生支持 COCO / YOLO / VOC 导入，DOTA、VisDrone 都能直接进。
**局限**：它不懂 VQA，QA 生成那段帮不上忙。

### 数据处理流水线段：Data-Juicer 值得评估

<https://github.com/modelscope/data-juicer> · [Sandbox 论文](https://arxiv.org/html/2407.11784v1)

阿里/ModelScope 出品，200+ 算子覆盖文本/图像/视频，NeurIPS 2025 Spotlight，中文文档好。**魔改友好度最高**——算子是插件式的，我们的密度聚类、越界判定可以直接封装成自定义 OP；自带多模态格式转换工具，能转 LLaVA/ShareGPT。Sandbox 提供 probe-analyze-refine 的数据-模型协同迭代循环。

**局限**：学习曲线陡，YAML 驱动配置调试麻烦；多模态算子偏向图文对（caption）场景，**对 bbox 语义没有原生理解**——我们的集结/越界派生还是得自己写。

**建议**：先按现在的脚本跑通第一版，验证数据配方有效后，再把流水线迁到 Data-Juicer 上换取可扩展性。上来就用它，会在调框架上浪费掉本该用于调数据的时间。

### LLM 合成段：distilabel / Curator ❌ 不建议

- distilabel：<https://github.com/argilla-io/distilabel>
- Curator：<https://github.com/bespokelabsai/curator>

两者都是成熟的合成数据框架，但**对我们的场景性价比不高**：主要面向纯文本合成，多模态支持薄；而我们的核心逻辑是「FACTS 约束下的改写 + 换模型校验」，本身只有两百行，套进它们的抽象层反而更难调试和排错。

### 人工审核段：Label Studio 或 Argilla

- Label Studio：<https://github.com/HumanSignal/label-studio> —— 支持自定义 VQA 标注模板，抽检和 golden set 精修用它
- Argilla：<https://github.com/argilla-io/argilla> —— 更偏 LLM 输出审核，和 distilabel 同一家

分层抽检那一步需要界面，纯 JSON 人工看效率太低。二选一即可。

### 评测段：VLMEvalKit 或 lmms-eval

训练完的评测环节，不要自己写评测脚本。
- VLMEvalKit：<https://github.com/open-compass/VLMEvalKit>
- lmms-eval：<https://github.com/EvolvingLMMs-Lab/lmms-eval>

---

## 一句话总结

**数据集选 8 个不选 20 个；质量靠五道闸 + 分层抽检 + golden set；框架上引入 FiftyOne 做筛选，其余先用自己的脚本跑通，验证有效后再考虑迁 Data-Juicer。**

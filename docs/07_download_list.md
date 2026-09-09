# 下载清单：9 个源数据集 + 2 个混入数据集

> 与 [08 号文档](08_task_types.md) 的 4 类定稿一致。类别合并后 **DroneCrowd 从选配升为必需**
> （人员聚集主力），所以是 9 个而非 8 个。各自的处理命令见 [09 号文档](09_dataset_processing.md)。

链接均已核实。「⚠️」是需要注意的坑，「体积」为估算值，实际以下载页为准。

| # | 数据集 | 供哪个类 | 体积估算 | 获取方式 |
|---|---|---|---|---|
| 1 | MAR20 | `massing/equipment` **主力** | ~2–4 GB | 学术发布，质量最好 ⭐ |
| 2 | Mendeley UAV 军事目标 | `massing/equipment` 补充 | ~1–3 GB | ⚠️ Roboflow/Kaggle 二次汇编，质量参差，只取 tank |
| 3 | FASDD_UAV | `smoke` + `explosion` | ~5–10 GB | Science Data Bank，只下 `FASDD_UAV.zip` |
| 4 | ERA + CapERA | `personnel`/`explosion`/正常 | ~10–20 GB | 主页下载；CapERA 只是标注文件 |
| 5 | DroneCrowd | `massing/personnel` | ~15–25 GB | GitHub 页给出网盘链接 |
| 6 | VisDrone DET + MOT | `border_crossing` + 正常 | ~20–30 GB | GitHub 页给出各子集链接 |
| 7 | DOTA v2.0 | 困难负样本主力 | ~30–50 GB（切片后更大） | 官网需注册 |
| 8 | Drone-Anomaly | 正常帧 | ~10–15 GB | GitHub 页给出网盘链接 |
| 9 | UCF-Crime / XD-Violence + HIVAU-70k | `explosion` + 同场景负样本 | 视是否下原始视频，见下 | 建议先只下 HIVAU 的 HF 预处理版 |

---

## 1. MAR20 ⭐ 装备集结的主力

- 官方：<https://gcheng-nwpu.github.io/>（西北工业大学，可能是百度网盘）
- YOLO 镜像：<https://universe.roboflow.com/mar20/mar20-s3e1w>（Roboflow，需注册但下载方便）

3,842 张高分辨率遥感图 / 22,341 实例 / 20 类军机，取自全球 60 个军用机场（Google Earth），HBB + OBB 双标注。

**`massing/equipment` 只需约 2,800 张图，MAR20 一个就够。** 装备集结不必须是坦克——
军机集结同样是装备集结，且 MAR20 是学术发布、原始遥感影像，质量档次远高于社区汇编数据集。

```bash
python tools/prepare.py mar20 --root data/raw/MAR20 --out data/interim/mar20.jsonl      # 官方版/VOC 格式
python tools/prepare.py mar20 --root data/raw/MAR20 --format yolo --out ...             # Roboflow YOLO 版
```

**Roboflow 下载步骤**：页面左侧 Dataset 标签 → 选版本 → Download Dataset → Format 选
**Pascal VOC**（适配器默认格式）→ 「Download zip」或「Show download code」拿到 curl 命令。
需要一个免费账号。服务器上直接跑那条 `curl -L "https://universe.roboflow.com/ds/XXX?key=YYY" > roboflow.zip` 最省事。

**Roboflow 三个版本怎么选（已核对）**：

| 版本 | 图片数 | 预处理 | 判断 |
|---|---|---|---|
| v1 | 3,842 | 640×640 Stretch | ❌ 压缩，小目标糊 |
| v2 | 9,222 | 640×640 Stretch | ❌ 压缩 + 增强 |
| **v3** | 9,222 | **无 resize** | ✅ **选这个** |

⚠️ **唯一重要的标准是有没有 Resize**。MAR20 是高分辨率遥感图、飞机只占几十像素，
压到 640×640 后小目标糊掉，不能用于 grounding 训练。

v3 的 9,222 张是 3 倍增强的结果：原始 3,842 按 88/8/4 切成 train 2,690 / valid 768 / test 384，
train 增强 ×3 = 8,070，`8070 + 768 + 384 = 9222`。增强副本是同一张图的翻转/旋转/调色版本，
图片数虚高但信息量没涨，且同一场景的多个副本会分散到 train/test 两侧造成指标虚高。

**适配器默认会折叠这些副本**（Roboflow 的增强副本共享 `.rf.` 之前的文件名前缀，据此精确还原
到 3,842 张源图）。这比按图像哈希去重可靠——dHash 对水平翻转不是不变的，抓不到翻转副本。
想保留增强用 `--keep-augmented`。

适配器还会检查：若所有图尺寸都是 640×640，会告警提示该版本做了 Resize。

## 2. Mendeley UAV 军事目标 ⚠️ 降级为补充，只取 tank

<https://data.mendeley.com/datasets/9z7yrcrpjk/1>

类别构成（官方数字）：

| 类别 | 图片 | 实例 | 处理 |
|---|---|---|---|
| tank | 3,000 | 4,990 | ✅ 保留，唯一的地面军事装备来源 |
| people | 2,644 | 4,492 | ❌ 默认丢弃，让位给 DroneCrowd |
| drone | 1,359 | 1,296 | ❌ 默认丢弃，反无人机检测与本项目无关 |
| soldier | 982 | 3,240 | ⚠️ 保留但标记，**掺有 GTA5 引擎生成的合成图** |

**质量提示（重要）**：官方说明图像"主要采集自 **Roboflow 和 Kaggle**"，是二次汇编而非原始采集；
soldier 类因真实航拍素材不足用 GTA5 合成图做了增强；且只是"侧重"航拍视角，实际混有地面视角照片。

因此适配器默认只保留 `tank` 与 `soldier`，并给含 soldier 的图标 `meta.synthetic_risk`。

```bash
python tools/prepare.py mendeley --root data/raw/mendeley --out data/interim/mendeley.jsonl
python tools/prepare.py mendeley --root ... --keep-classes tank --out ...   # 更保守: 只要 tank

# 下完务必先单独过一遍筛选看保留率, 再决定用不用
python tools/derive_events.py --scenes data/interim/mendeley.jsonl --out data/interim/mendeley_ev.jsonl
python tools/screen.py --scenes data/interim/mendeley_ev.jsonl --out-dir data/screened_mendeley
```

保留率低于 50% 就说明不值得投入，直接靠 MAR20 撑 equipment 子类即可。

## 3. FASDD_UAV

<https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda>

⚠️ **只下 `FASDD_UAV.zip`**，另两个（`FASDD_CV` 通用视角、`FASDD_RS` 遥感）本项目用不到。
FASDD_UAV 含 36,308 个火焰实例 + 17,222 个烟雾实例。

💡 标注同时提供 YOLO / VOC / COCO / TDML 四种格式，适配器直接取 COCO 子目录。

## 4. ERA + CapERA

- ERA：<https://lcmou.github.io/ERA_Dataset/> · 备用 <https://ieee-dataport.org/open-access/era-dataset>
- CapERA：<https://github.com/yakoubbazi/CapEra>

2,864 段视频，每段 5 秒 / 30fps / 640×640，25 类事件。CapERA 给每段补 5 条人工 caption。

⚠️ **CapERA 只有标注文件，视频要用 ERA 的**，所以 ERA 必须先下。
⚠️ 抽帧需要 `ffmpeg`（`apt-get install -y ffmpeg`）或 `pip install opencv-python`。

## 5. DroneCrowd

<https://github.com/VisDrone/DroneCrowd>

112 段 / 33,600 帧 1920×1080 / 480 万人头点 / 20,800 条轨迹。

⚠️ 标注是**人头点**不是 bbox。发布形态可能是 `.mat`（需 `pip install scipy`）或 txt，两种都支持。
⚠️ 相邻帧几乎全是近重复，务必按 stride 稀疏抽（默认每 30 帧取 1）。

## 6. VisDrone DET + MOT

<https://github.com/VisDrone/VisDrone-Dataset>

**两个子集都要下**：

- `VisDrone2019-DET`（train/val）—— 航拍 bbox 底座与正常场景负样本
- `VisDrone2019-MOT`（train/val）—— 带 track_id，**越界移动的唯一来源**

## 7. DOTA v2.0

<https://captain-whu.github.io/DOTA/dataset.html>

188,282 实例 / 15 类 / OBB。⚠️ 原图可达 20000×20000，**必须先切片**（`prepare.py dota` 会自动做，默认 1024/overlap 200）。切片会额外占磁盘，预留 2 倍空间。

## 8. Drone-Anomaly

<https://github.com/Jin-Pu/Drone-Anomaly>

7 个场景，37 训练 / 22 测试视频序列，51,635 + 35,853 帧，640×640。
按设计**只取正常帧**（其异常类型与本项目 4 类不匹配，详见 09 号文档）。

## 9. UCF-Crime / XD-Violence + HIVAU-70k

- HIVAU-70k 标注：<https://github.com/pipixin321/HolmesVAU>
- HF 预处理版：`backseollgi/HIVAU-70k_UCF-Crime`、`backseollgi/HIVAU-70k_XD-Violence`
- UCF-Crime 视频：<https://www.crcv.ucf.edu/projects/real-world/>（1,900 段 / 128 小时）
- XD-Violence 视频：<https://roc-ng.github.io/XD-Violence/>（4,754 段 / 217 小时）

💡 **先只下 HuggingFace 的预处理版**，它已经切好片段。够用就不必下 345 小时的原始视频，
能省几百 GB 磁盘和大量下载时间。确实需要原始视频时再补。

---

## 混入数据集（不是源数据）

| 数据集 | 地址 | 用途 |
|---|---|---|
| **VRSBench** | <https://huggingface.co/datasets/xiang709/VRSBench> | 按 10% 混入训练防灾难性遗忘，顺带提供 grounding 范式 |
| **MOCO** | <https://github.com/Panlizhi/MOCO> | **申请制**，现在就发申请，批下来加入 B 组，不阻塞当前进度 |

---

## 建议顺序与磁盘预算

**先下 1、2、3**（MAR20、Mendeley、FASDD_UAV）：体量小、都是静态图、不需要 ffmpeg，
下完就能跑通 `prepare → derive_events → screen → make_golden → build_vqa` 全链路，
先验证流程再投入大数据集。

之后按 4→5→6→8→7→9 补齐。

**磁盘预算**：不下 UCF-Crime / XD-Violence 原始视频约 **150–250 GB**（含 DOTA 切片），
下了则要预留 **500 GB 以上**。

下载用 `aria2c -x 16 -s 16 <url>` 或 `wget -c <url>`（支持断点续传），大文件中断重来很浪费时间。

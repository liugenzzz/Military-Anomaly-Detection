# 各数据集处理代码说明

统一入口 `tools/prepare.py`，每个数据集一个子命令，各自的逻辑在 `tools/ds/<name>.py`。

**为什么不做一个通用适配器**：这几个数据集的差异是实质性的，不只是格式差异——

- 标注载体不同：VOC XML / COCO json / YOLO txt / DOTA OBB / MATLAB mat / 点标注 / 视频
- **类名语义不同**：MAR20 的 plane 全是军机，DOTA 的 plane 含民航客机，映射成同一个类名会让民用机场被判成装备集结
- 异常语义不同：Drone-Anomaly 的异常类型与本项目 4 类不匹配，只能取其正常帧
- 时间语义不同：UCF-Crime 需要事件时间边界才能区分异常段与正常段

---

## 需要准备的数据集（9 个）

| # | 数据集 | 供哪个异常类 | 优先级 |
|---|---|---|---|
| 1 | **Mendeley UAV 军事目标** | `massing/equipment`（tank/soldier） | P0 体量小，先拿它跑通 |
| 2 | **MAR20** | `massing/equipment`（军机） | P0 同上 |
| 3 | **FASDD_UAV** | `smoke` + `explosion`(火光) | P0 |
| 4 | **ERA + CapERA** | `massing/personnel`、`explosion`、正常 | P1 需抽帧 |
| 5 | **DroneCrowd** | `massing/personnel` 主力 | P1 ⬆ 类别合并后从选配升为必需 |
| 6 | **VisDrone DET + MOT** | `border_crossing`（唯一来源）+ 正常 | P1 |
| 7 | **DOTA v2.0** | **困难负样本**主力 | P2 需切片 |
| 8 | **Drone-Anomaly** | 正常（真实航拍常态帧） | P2 |
| 9 | **UCF-Crime / XD-Violence + HIVAU-70k** | `explosion` + 同场景困难负样本 | P3 体量最大，放最后 |

## 额外依赖

```bash
pip install -r requirements.txt          # PyYAML + Pillow, 必需
apt-get install -y ffmpeg                # 抽帧(ERA / UCF-Crime / XD-Violence) —— 推荐
pip install opencv-python                # 抽帧的备选方案
pip install scipy                        # 仅当 DroneCrowd 拿到的是 .mat 标注时需要
```

不下视频类数据集（ERA / UCF-Crime / XD-Violence）就不需要 ffmpeg。

---

## 逐个数据集

### 1. Mendeley UAV 军事目标 —— 格式自动探测

```bash
python tools/prepare.py mendeley --root data/raw/mendeley --out data/interim/mendeley.jsonl
```

Mendeley 社区数据集**不声明标注格式**，所以适配器自动探测 coco / voc / yolo，不用你去猜。类别映射 `tank→tank`、`soldier→soldier`、`people→person`、`drone→drone`，其中 `tank` 命中 ontology 的 `require_any`。

文件名含 `syn`/`aug`/`synthetic` 的标 `meta.synthetic=true`，方便后续控制合成占比（建议 ≤30%）。

### 2. MAR20 —— 20 类机型统一为 military-plane

```bash
python tools/prepare.py mar20 --root data/raw/MAR20 --out data/interim/mar20.jsonl
# 若拿到的是 Roboflow 的 YOLO 版:
python tools/prepare.py mar20 --root ... --format yolo --classes classes.txt --out ...
```

MAR20 的 20 个类别名是 `A1`..`A20`（机型代号），**全部是军机**，统一映射为 `military-plane`，机型代号保留在 `attrs.model` 里供属性题使用。会自动在 `Annotations/Horizontal Bounding Boxes/` 下找 XML。

### 3. FASDD_UAV —— 取 COCO 子目录

```bash
python tools/prepare.py fasdd --root data/raw/FASDD_UAV --out data/interim/fasdd.jsonl
```

四种标注格式里 COCO 最省事，适配器直接找它。`fire→explosion` 事件、`smoke→smoke` 事件，同时保留 bbox，所以这个数据集**同时供判定题与区域定位题**。两者都有的图会同时打两个事件标签。

FASDD 含大量既无火也无烟的图，默认保留作为负样本（`--drop-negatives` 可关闭）。

### 4. ERA + CapERA —— 25 类分三档处理

```bash
python tools/prepare.py era --root data/raw/ERA --frames-dir data/frames/era \
    --capera data/raw/CapEra/captions.json --out data/interim/era.jsonl
```

每段 5 秒视频抽 3 帧（首/中/尾），抽更多只会产生近重复。**25 个类别的三档划分是这个数据集的关键**：

- **ANOMALY**：`fire→explosion`；`conflict`/`parade_protest`/`party`/`concert`/`religious_activity` → `massing/personnel`
- **NORMAL**（明确无烟火，安全负样本）：`non_event`、`traffic_congestion`、`police_chase`、`harvesting`、`ploughing`、各类体育
- **EXCLUDE**（歧义，一律排除）：`post_earthquake`、`flood`、`landslide`、`mudslide`、`traffic_collision`、`car_racing`、`constructing`

排除的理由很实际：**地震后/滑坡/交通事故/赛车的画面里常带烟尘或火光，把它们当成"正常"喂进去，等于教模型"有烟也算正常"，会直接破坏烟雾类的判别能力。**

ERA 没有 bbox，所以这批数据供判定/分类/描述题，不供 grounding 题。CapERA 的 caption 会按 video stem 关联挂到 `scene.caption`。

### 5. DroneCrowd —— 点标注转小框

```bash
python tools/prepare.py dronecrowd --root data/raw/DroneCrowd \
    --ann-dir data/raw/DroneCrowd/annotations --stride 30 --out data/interim/dronecrowd.jsonl
```

标注是**人头点**不是 bbox，转成 ±8px 小方框统一走 bbox 逻辑（密度聚类只用中心点，框大小不影响判定）。`.mat` 与 txt 两种发布形态都支持，txt 按列数自动识别。

33,600 帧几乎全是相邻近重复，**必须按 stride 稀疏抽**，默认每 30 帧取 1。人头数少于 20 的帧不会被判为聚集，自动成为"稀疏人群"困难负样本。

### 6. VisDrone DET + MOT —— 自动放置越界边界

```bash
python tools/prepare.py visdrone-det --root data/raw/VisDrone2019-DET-train --out data/interim/vd_det.jsonl
python tools/prepare.py visdrone-mot --root data/raw/VisDrone2019-MOT-train --stride 30 --out data/interim/vd_mot.jsonl
```

**越界判定原本需要人工给每个序列画一条边界线（约 56 条，1 小时）。现在改为自动放置**：取所有轨迹的平均运动方向，在轨迹中心处作一条与之垂直的直线延伸到画面外。这样必然有相当比例的轨迹穿过它，无需人工。

`meta.boundary_source` 会记 `auto` 还是 `manual`。**仍建议抽查几个序列**确认边界位置合理。要人工指定就给 `--boundaries boundaries.json`，格式 `{"序列名": [[x,y],[x,y],...]}`（像素坐标折线）。

VisDrone 全是民用场景，车辆一律映射为 `vehicle` 而非 `military-*`——这正是它成为困难负样本来源的原因。

### 7. DOTA v2.0 —— 切片 + 以簇为中心补切

```bash
python tools/prepare.py dota --root data/raw/DOTA --tiles-dir data/tiles/dota \
    --tile 1024 --overlap 200 --min-objects 5 --out data/interim/dota.jsonl
```

三个处理要点：

1. **必须切片**。原图可达 20000×20000，不切无法送模型，且小目标缩放后直接消失。切片边缘截断的框按面积占比 <0.6 丢弃——留半个框会教出错误的尺寸先验。
2. **类名映射是这个数据集最关键的一步**。DOTA 的 `plane` 大量是民航客机、`ship` 大量是民用港口船舶，映射为 `civil-plane` / `ship`（而非 `military-plane` / `warship`），于是"民用机场停满飞机""港口密集停泊"会被判为 `hard_negative` 而不是装备集结。**这批样本教模型看目标类型，而不是看有没有一堆东西挤在一起。**
3. **以簇为中心额外补切**。规则网格切片会把跨切片边界的聚集切散——实测 9 架机群被切成 5+6，每片都不足阈值，聚集判定直接失效。所以先在原图坐标上聚类，再为每个簇裁一块能完整包住它的切片（`meta.cluster_centered=true`）。

体育场馆、泳池等无关类别直接丢弃，`difficult` 标记的目标也丢。

### 8. Drone-Anomaly —— 只取正常帧

```bash
python tools/prepare.py drone-anomaly --root data/raw/Drone-Anomaly --stride 10 --out data/interim/da.jsonl
```

**这个数据集按设计只产出正常样本。** 它的 10 类异常是面板缺陷、铁轨障碍物、不明物体一类的场景异常，与本项目的 4 类完全不同：标成我们的异常是错标，标成正常也是错标。所以：

- `training` 帧（全为正常）→ normal
- `testing` 中标签为 0 的帧 → normal
- `testing` 中标签为 1 的帧 → **排除**

它的价值是提供大量真实航拍正常帧，且与异常帧同场景同机位，场景多样性远好于随便找的正常图。

### 9. UCF-Crime / XD-Violence + HIVAU-70k —— 最需要小心的一个

```bash
# 格式不确定时先看结构
python tools/prepare.py hivau-inspect --ann data/raw/HIVAU-70k/annotations.json

# 抽帧(时间边界二者任一即可)
python tools/prepare.py hivau --video-root data/raw/UCF-Crime --frames-dir data/frames/ucf \
    --temporal-ann data/raw/UCF-Crime/Temporal_Anomaly_Annotation.txt --out data/interim/ucf.jsonl
python tools/prepare.py hivau --video-root data/raw/XD-Violence --frames-dir data/frames/xd \
    --ann data/raw/HIVAU-70k/annotations.json --out data/interim/xd.jsonl

# 把 HIVAU 自带的指令标注直接导出为 ShareGPT, 省掉这部分生成成本
python tools/prepare.py hivau-export --ann .../annotations.json \
    --frames-index data/interim/ucf.jsonl --out data/vqa_native/hivau.json
```

三个要点：

1. **13 类/6 类异常中只有少数映射到我们的 4 类**。`Explosion`/`Arson`/XD 的 `G` → `explosion`；`Riot`/`B4` → `massing/personnel`。`Fighting`、`Robbery`、`Shooting`、`Stealing`、`Vandalism` 等**一律排除**——它们是治安事件，不是我们定义的军事异常，混进来会把类别搞脏。XD-Violence 的文件名后缀编码（`A`=正常，`G`=爆炸，`B1`=打斗，`B2`=枪击，`B4`=骚乱，`B5`=虐待，`B6`=车祸）已内置。
2. **异常视频的正常段是最优质的负样本**。一段 4 分钟的 Explosion 视频可能只有 8 秒在爆炸，其余 3 分 52 秒是同机位同光照的正常画面。用事件时间边界切出来（两侧留 2 秒安全边距避开过渡帧），模型就无法靠"这个场景看着就危险"蒙对。这些帧标 `meta.hard_negative_source`。**注意：即使某个异常类型被排除（如 Fighting），它的正常段仍然采集**——正常段的价值与异常类型无关。
3. **没有时间边界的异常视频不能用**，会跳过并告警。不知道哪几秒是异常，抽出来的帧会大面积错标。正常视频不需要边界，整段都是正常。

---

## 处理完之后

```bash
# 合并 —— 注意别把输出文件本身 cat 进去
cat data/interim/mar20.jsonl data/interim/mendeley.jsonl ... > data/all_scenes.jsonl

python tools/derive_events.py --scenes data/all_scenes.jsonl --out data/all_ev.jsonl
python tools/screen.py       --scenes data/all_ev.jsonl --out-dir data/screened
python tools/make_golden.py  --scenes data/screened/scenes_kept.jsonl --out-dir data/golden
python tools/build_vqa.py    --scenes data/screened/scenes_kept.jsonl --out-dir data/vqa_rule \
                             --exclude-ids data/golden/exclude_ids.txt
python tools/llm_qa.py generate --scenes data/screened/scenes_kept.jsonl \
                             --exclude-ids data/golden/exclude_ids.txt --out data/interim/gen.jsonl
```

合并时**不要用 `cat data/interim/*.jsonl`**，如果输出文件也在同一目录会把自己 cat 进去。`screen.py` 会以 `duplicate_scene_id` 为原因剔除并告警，但明确列出文件名更省事。

---

## 已验证的行为

用按各数据集真实目录结构与标注格式构造的 fixture 跑通了 8 个适配器（视频类的抽帧逻辑依赖 ffmpeg/OpenCV，本环境未装，未做端到端验证）。关键语义核对：

| 输入 | 结果 |
|---|---|
| MAR20 9 架军机列队 | `massing/equipment` ✅ |
| DOTA 9 架**民航客机**列队 | `hard_negative` ✅ |
| DOTA 14 辆民用车密集成簇 | `hard_negative` ✅ |
| DroneCrowd 34 个人头 | `massing/personnel` ✅ |
| FASDD 同图含火与烟 | `explosion` + `smoke` ✅ |
| VisDrone MOT 轨迹穿过自动边界 | `border_crossing` ✅ |
| Drone-Anomaly 异常帧 | 已排除，只留正常帧 ✅ |

**同样是 9 架飞机、同样的列队构型，军机判为集结、民航判为困难负样本**——这是整套口径设计要达到的效果。

## 开发过程中修掉的三个真问题

1. **切片把聚集切散**：规则网格切片使 9 架机群变成 5+6，每片都不足阈值。修法是以簇为中心补切。
2. **聚类阈值尺度依赖**：`eps` 只按图像对角线取，同一片机群在 2400×1600 原图上间距 90px 能连通，裁成 1024×1024 后阈值收缩到 87px 就连不上。修法是加一条"3 倍目标尺寸"作为下限——相邻的物理含义本来就是"间距不超过目标本身的几倍"，而不是"占画幅的百分之几"。
3. **去重按 image_id 记录导致误删**：同一个 id 有两份时，本该保留的那份也在丢弃集合里，结果整组被清空（实测 49 个 scene 只剩 1 个）。修法是改用下标记录，并把重复 id 单独识别为 `duplicate_scene_id`。这条也顺带抓出了 Drone-Anomaly 适配器的一个 ID 冲突（`training/` 与 `testing/` 下子目录同名）。

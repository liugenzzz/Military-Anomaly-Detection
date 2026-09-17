# 交接文档：把 `concentration`（密集分布）接进流水线

> 基准提交：`982ef79`（本体 v1.0.0 刚落地那一版）
> 读者：接手实现的人/AI。**本文档已经把设计定死了，照着实现即可，不要重新设计。**
> 如果你发现设计本身有问题，**先停下来说**，不要一边怀疑一边改。

---

## 0. 一句话说清要做什么

本体 `configs/ontology.yaml` v1.0.0 里有三个类是 `status: ready` + `enabled: false` +
`blocked_by`：

```
concentration       密集分布（车辆/人群）   land/征候
air_concentration   机群密集停放           air/征候
sea_concentration   舰船密集停泊           sea/征候
```

**素材是现成的**（DroneCrowd 3176 + VisDrone 4045 + MAR20 3842 + DOTA），
卡住它们的是代码：现在的 `build_vqa.py` 会对这批图生成「未见异常」，
而这三个类是**正面标签**，一开就自相矛盾。

这份文档说清楚怎么接。做完之后把三个类的 `enabled` 改成 `true`。

---

## 1. 先理解这一个概念，其余都是它的推论

项目里原来只有两种图：**有异常** / **没异常**。
v1.0 引入了第三种：**没异常，但画面里有个值得说清楚的客观现象**。

| | 说的是 | 例子 | 是异常吗 |
|---|---|---|---|
| `massing` 集结 | 行为判断：力量正在汇聚 | 游行、坦克编队 | ✅ 是 |
| `concentration` 密集分布 | 客观事实：目标密集排布 | 停车场、停机坪、广场人流 | ❌ **不是** |

**两者在构型上完全一样**，区别只在「是不是正在汇聚」。
这是整个体系里最容易出错的一格 —— 混为一谈就是教模型看见任何停车场都报警。

所以要做的事只有一件：**让代码区分「事件类型」和「异常类型」**。
现在这两个概念在 `Scene.anomaly_types` 里是同一个东西，那就是全部问题的根源。

### 判定题的真值表（实现时对着它写）

| 有异常事件 | 有 concentration 事件 | `hard_negative` | 判定题应该答什么 |
|---|---|---|---|
| ✅ | 任意 | 任意 | 「存在异常，为X。」**现状，不动** |
| ❌ | ✅ | ✅ | 「未见异常。」+ 承认密集 + 为什么不算 → **现状的 hard_negative 分支已经对了** |
| ❌ | ✅ | ❌ | 同上。**需要新增：让 concentration 事件也能走进那个分支** |
| ❌ | ❌ | ✅ | 现状 hard_negative 分支，**不动** |
| ❌ | ❌ | ❌ | 现状 normal 分支，**不动** |

看出来了吗：**第 2、4 行的答案是同一套话术**。
所以这不是"再写一套文案"，是"把进入那套文案的闸门放宽一格"。

---

## 2. 任务 A：区分「事件类型」和「异常类型」

### A1. 本体加一个字段

`configs/ontology.yaml`，给这四个类加 `is_anomaly: false`：

```yaml
- id: normal            # 加
- id: concentration     # 加
- id: air_concentration # 加
- id: sea_concentration # 加
```

其余类**不写**这个字段，代码里默认 `True`。
（为什么不写成 `is_indicator`：`family: indicator` 已经有了，但 `smoke` 也是
indicator 而它确实是异常。族说的是"怎么判"，这个字段说的是"算不算异常"，
两件事，别合并。）

### A2. `tools/build_vqa.py` — `RuleBuilder.__init__`（约 L220）

现在是：

```python
self.enabled = [c["id"] for c in onto["classes"]
                if c["id"] != "normal" and c.get("enabled", True)]
```

改成保留 `self.enabled` 不变（别的地方还在用），**另外加两个集合和两个方法**：

```python
# 「启用的异常类」—— 出否定题/纠错题时要问的是异常，不是征候
self.anomaly_ids = {c["id"] for c in onto["classes"]
                    if c.get("enabled", True) and c["id"] != "normal"
                    and c.get("is_anomaly", True)}
# 「启用的非异常征候类」—— concentration 那三个
self.indicator_ids = {c["id"] for c in onto["classes"]
                      if c.get("enabled", True) and c["id"] != "normal"
                      and not c.get("is_anomaly", True)}

def _anom(self, s: Scene) -> list[str]:
    """这张图上的**异常**类型。Scene.anomaly_types 给的是所有事件类型,
    里面可能混着 concentration —— 那不是异常, 拿它去开判定题的异常分支,
    等于教模型看见停车场就报警。"""
    return [t for t in s.anomaly_types if t in self.anomaly_ids]

def _indic(self, s: Scene) -> list[str]:
    return [t for t in s.anomaly_types if t in self.indicator_ids]
```

⚠️ **不要改 `tools/scene.py` 的 `Scene.anomaly_types`。**
`Scene` 不认识本体，让它去判"算不算异常"是把本体知识漏进数据结构。
过滤发生在 `RuleBuilder` 里，因为只有它手上有 `onto`。

### A3. 逐个改调用点

基准提交 `982ef79` 的行号（会漂，同时给了锚点字符串）：

| 行 | 锚点 | 改成 | 为什么 |
|---|---|---|---|
| 259 | `names = [self.zh[t] for t in s.anomaly_types` | `self._anom(s)` | 否则判定句会说「存在异常，为烟雾、**密集分布**」 |
| 260 | `miss = [t for t in s.anomaly_types` | `self._anom(s)` | 同上 |
| 284 | `"anomaly": s.anomaly_types or ["normal"],` | **见 A4** | 配额口径 |
| 327 | `if s.anomaly_types:`（`judge` 里） | `if self._anom(s):` | 真值表第 2/3 行 |
| 410 | `if not s.anomaly_types:`（`locate_verbal`） | `if not self._anom(s):` | 密集区走 `dense_region`，不走「异常在哪」 |
| 421 | `if not s.anomaly_types:`（`locate_box`） | `if not self._anom(s):` | 同上 |
| 432 | `if not s.anomaly_types:`（`locate_box_multi`） | `if not self._anom(s):` | 同上 |
| 523 | `absent = [c for c in self.enabled ...]`（`negation`） | `self.anomaly_ids` + `self._anom(s)` | 否定题的问法池写死了"异常类型"，见 `configs/prompts/ask/negation.txt` 头部 |
| 556 | `if not s.anomaly_types:`（`coverage`） | `if not self._anom(s):` | 烟火覆盖题只对异常出 |
| 633-634 | `if style < 0.8 and self.enabled:` / `absent = ...` | `self.anomaly_ids` + `self._anom(s)` | 纠错题同 negation |
| 638 | `if s.anomaly_types` | `if self._anom(s)` | 同上 |
| 746 | `if s.anomaly_types or not s.meta.get("hard_negative"):` | **见 A5** | 核心改动 |
| 810 | `if not s.anomaly_types:`（`why` 末尾兜底） | `if not self._anom(s):` | 兜底理由句 |

### A4. metadata：`anomaly` 保持"只装异常"，新增 `indicators`

L284 附近改成：

```python
"anomaly": self._anom(s) or ["normal"],
"indicators": self._indic(s),        # 新增: 非异常的征候标签
```

**这是刻意的设计选择，别改成把 concentration 塞进 `anomaly`。** 理由：

- `apply_quota()` 按 `metadata["anomaly"]` 排配额，把 `!= "normal"` 当异常类、
  每类目标 25000，并据此反推负样本额度。concentration **不是异常类**，
  塞进去会虚增 `kept_pos`，负样本配额跟着虚增。
- `qc_consistency.py` 的第 3 项检查（`normal_but_asserts_anomaly`）判的是
  `anomalies == ["normal"]`。保持只装异常，这项检查**自动**就能覆盖
  「concentration 图不许断言异常」，一行都不用改。
- `check_dataset.py` 的负样本占比同理。

也就是说：这个口径选对了，三个下游文件一行不用动。选错了要改三处、还容易漏。

唯一的代价：concentration 落进 normal 桶，拿不到自己的 25000 配额。
**这是对的** —— 它本来就在负样本那一侧，只是带了个正面的描述标签。

`check_dataset.py` 可以顺手加一行把 `indicators` 的分布也打出来（纯可视化，不影响逻辑）。

### A5. `dense_region` 的闸门（L746，核心改动）

现在：

```python
if s.anomaly_types or not s.meta.get("hard_negative"):
    return None
box = s.meta.get("hard_negative_bbox") or (self._region_box(s) or [None])[0]
```

改成：

```python
# 有真异常 -> 区域信息走 locate_*，不走这里
if self._anom(s):
    return None
# 无异常但画面里确实有一片密集区: 困难负样本有, concentration 事件也有。
# 后者是 v1.0 新增的正面标签 —— 它的框来自 cluster_bbox，比 hard_negative_bbox
# 更准（后者只是簇中心的外接矩形）。
if not (s.meta.get("hard_negative") or self._indic(s)):
    return None
box = (self._region_box(s) or [None])[0] or s.meta.get("hard_negative_bbox")
```

⚠️ **取框的优先级要反过来。** 现在是 `hard_negative_bbox` 优先；有 concentration
事件时应该优先用事件的 `cluster_bbox`（`_region_box` 会取到它），因为
`hard_negative_bbox` 是"簇中心点的外接矩形"，不含目标本身的尺寸，框会偏小。

### A6. `judge()` 不要再用 `s.events[0]`（L328 和 `_judge_detail` L296）

```python
ev = s.events[0].evidence          # L328
e = s.events[0] if s.events else None   # _judge_detail
```

**这在多标签下是错的。** 事件顺序取决于「适配器挂的在前、规则派生的在后」，
一张 FASDD 冒烟图如果同时聚出 concentration，`events[0]` 到底是哪个不好说。
改成显式挑异常事件：

```python
e = next((e for e in s.events if e.type in self.anomaly_ids), None)
```

`_judge_detail` 需要多收一个参数，或者把挑选逻辑提到 `judge()` 里传进去。

### A6b. `region_box_of()` 也要优先挑异常事件

`tools/build_vqa.py` 的 `region_box_of()`（模块级函数，规则侧和 LLM 侧**共用**）：

```python
ev = next((e for e in s.events if "cluster_bbox" in e.evidence), None)
```

同一个"取第一个"问题：concentration 事件也有 `cluster_bbox`。
一张 smoke + concentration 的图问「异常在哪」，可能拿到密集分布的框、
标签还写成「密集分布区域」。

它是模块级函数、拿不到 `RuleBuilder` 的集合，所以加一个可选参数：

```python
def region_box_of(s, zh, prefer: set[str] | None = None):
    evs = [e for e in s.events if "cluster_bbox" in e.evidence]
    if prefer:
        evs.sort(key=lambda e: e.type not in prefer)   # 异常事件排前面
    ev = evs[0] if evs else None
```

调用点两处：`RuleBuilder._region_box()` 传 `self.anomaly_ids`；
`llm_qa.facet_applicable()` 里的 `region_box_of(s, {})` 保持不传（它只判有没有框）。

⚠️ 这一条和 A6 是同一个根因：**多标签之后，"取 events[0]" 到处都不成立了。**
实现时把三个地方一起改（`judge` L328、`_judge_detail`、`region_box_of`），
别只改一个 —— 这个项目有过"两个 bug 会叠加，修一个不够"的先例
（DroneCrowd 的目录假设 + 帧号解析）。

### A7. 判定题的第三种答案（`judge` 的 `elif` 分支，L357）

现在：

```python
elif s.meta.get("hard_negative"):
    ...
    kind = s.meta.get("hard_negative_kind", "civil_cluster")
    if kind == "ordinary_crowd": ...
    elif kind == "aircraft_parking": ...
    else: ...   # civil_cluster
```

改成 `elif s.meta.get("hard_negative") or self._indic(s):`，并且 `kind` 的兜底
要按 concentration 的类型来（一张没被标 hard_negative 的 concentration 图
是没有 `hard_negative_kind` 的）：

```python
kind = s.meta.get("hard_negative_kind")
if not kind:
    kind = {"air_concentration": "aircraft_parking",
            "sea_concentration": "harbor_berthing",
            "concentration": "civil_cluster"}.get(
                (self._indic(s) or [""])[0], "civil_cluster")
```

`harbor_berthing` 是新 kind，要在 `judge` 和 `dense_region` 两处各加一套措辞。
参考现有 `aircraft_parking` 的写法：

```
未见异常。画面中虽有 N 艘舰船密集停泊，但它们沿泊位排列、紧邻码头，
属于港口的日常停靠状态，不构成舰艇集结。
```

⚠️ **措辞必须按簇里到底是什么来定，绝不能写死一套。**
对一张满是军机的图说「均为民用目标，未见军机」是睁眼说瞎话，
而且正好是 `qc_consistency` 会抓的那类自相矛盾。这个坑已经踩过一次，
`derive_events.py` L176-198 的注释记着。

### A8. `why()` 加一条 concentration 分支（L806 之前）

现有顺序是：linear_formation → cluster → disaster → smoke/explosion → hard_negative → 无异常兜底。

问题：**`self._cluster(s)` 是按 `"cluster_bbox" in e.evidence` 找的，
concentration 事件也有 `cluster_bbox`** —— 会被第二条分支截胡，答出
「规模已达到**密集分布**的判定条件」，读起来像在说这是异常。

在 cluster 分支里加一句判断，或者在它之前插一条：

```python
ev = next((e for e in s.events if e.type in self.indicator_ids), None)
if ev and not self._anom(s):
    n = ev.evidence.get("count")
    cls = "、".join(CLS_ZH.get(c, c) for c in ev.evidence.get("classes", [])[:3])
    return (f"依据是目标的空间分布：{n}个{cls or '目标'}间距明显小于周边，聚成一片；"
            f"但它们排列规整、位置固定，属于常态的密集停放/活动，"
            f"没有向某处汇聚的过程，因此不按异常处置。")
```

---

## 3. 任务 B：`concentration` 的描述侧面（facet）

大模型侧（`tools/llm_qa.py`）按 facet 出描述题。三个新类现在**一个 facet 都没有**，
不补的话它们只能出规则侧的判定题，占不到「描述类 60%」那个盘子。

### B1. 新建目录与文件

```
configs/prompts/describe/concentration/
    density.txt      kind: density        密集程度量化
    layout.txt       kind: layout         排布形态（成排/成格/无序）
    site.txt         kind: site_civil     场地性质（停车场/广场/码头/停机坪）
    boundary.txt     kind: boundary       与"集结"的边界 —— 为什么不是
```

格式照抄 `configs/prompts/describe/normal/hard_neg.txt`，它是全套里写得最全的一个，
头部的 `#! kind / zh / anomaly / needs / must-not / min-items` 每一项都有示范。

`air_concentration` / `sea_concentration` 是否单独建目录：
**先共用 `concentration` 这一套**（在 `facets.py` 里把三个 id 都指向它）。
本体第 4 条硬性要求是「各异常类不得共用一套描述骨架」——
但这三个是**同一个征候的三个域**，不是三个异常类；真正要分开的是
`site` 和 `layout` 两个侧面的措辞，那由 FACTS 里的目标类别自然带出来。
等它们各自的图数上来了再拆。**这一条如果你不同意，先说，别自己改。**

⚠️ `boundary` 这个侧面是**全套数据里价值最高的一个**，理由和 `normal/hard_neg.txt`
里写的一样：模型误报的根源就是"看到一堆东西挤在一起就报集结"。
写的时候按那个文件的 answer-spec 结构来：**先承认哪里像，再说清为什么仍然不是**。
不要一上来就否定。

### B2. `tools/facets.py` 三处登记

1. `CROSS_CLASS_BAN`（L48）加三条。concentration 的禁忌词应包含
   `集结 越界 烟雾 火光 爆炸 灾害` —— 它是最容易被写成"疑似集结"的一类。
2. `EXPECTED`（L183）加 `"concentration": {"position", "full", "density", "layout",
   "site_civil", "boundary"}`。
   注意：**不要给它 `evidence` 和 `grounded`** —— 那两个侧面是"给异常的依据"。
3. 跑 `python -c "import sys;sys.path.insert(0,'tools');import facets;print(facets.check())"`
   必须返回 `[]`。这个自检会同时抓"缺侧面"和"多出未登记的侧面"。

### B3. `tools/llm_qa.py` 两处

- `facet_applicable()`（L62）加四个 kind 的门槛：
  ```python
  "density":    bool(cluster) and not anom,
  "layout":     bool(cluster) and not anom,
  "site_civil": bool(cluster) and not anom,
  "boundary":   bool(cluster) and not anom,
  ```
  （`anom` = 这张图有没有真异常事件，需要在函数里先算出来；
  函数签名里没有本体，最简单的办法是把 `anomaly_ids` 作为可选参数传进来。）
- `build_facts()` 的 `absent_anomaly_types`（L184）要**排除非异常类**，
  加上 `and c.get("is_anomaly", True)`。否则 FACTS 会告诉模型
  「本图不存在的异常类型包括：密集分布」，而密集分布根本不是异常。

`llm_qa.py` 的 L649 / L850 / L878 三处类别均衡**不用改** ——
那里按 `anomaly_types` 分桶是为了让各标签的描述题数量均衡，
concentration 参与均衡是对的。

---

## 4. 任务 C：`sea_concentration` 的家底要先量出来

本体里它是 `images: null` + `status: thin` + 「按图统计没跑过」。
**不要凭 DOTA 有 11424 个 ship 框就填一个估计值。**
（上一轮「车队能有几千」就是这么错的，实际是几十条量级。）

量法：把 `sea_concentration.enabled` 临时改 `true`，跑

```bash
python tools/derive_events.py --scenes data/all.jsonl --out /tmp/probe.jsonl --overwrite
```

`derive_events` 会打印每类的事件数**和否决原因分布**（`reject_report()`）。
看两个数：
- 事件数 → 填回本体的 `data.images`
- 否决原因 → 如果「单簇规模不足」占大头，说明 DOTA 的港口图里船是散的，
  这一类就该标 `thin` 甚至 `empty`；**不要为了让它好看去调 `min_cluster_size`**

然后照实改本体的 `images` / `status`，把 `enabled` 改回你量完之后该有的值。

同样的办法可以顺手把 `concentration` 的准确图数量出来（本体里也是 `null`）。

---

## 5. 任务 D：`massing` / `disaster` 的产量重统

v1.0 改了两处映射，旧数字全部作废：

- `tools/ds/era.py`：`concert` / `party` / `religious_activity` 从 `massing` 移到
  `concentration`（演唱会和宗教活动不是军事异常）
- `disaster` 从停用改为启用

所以要重跑一遍 `prepare`（ERA 那一支）→ `merge_scenes` → `derive_events`，
把真实产量填回 `CLAUDE.md` 的「语料现状」表和本体的 `data.images`。

**注意**：`massing` 清理后如果掉到几百，那它的 `status` 就该从 `thin` 保持不变
或更差，**不要通过放宽阈值把数字做回去**。本体 `qa_per_image` 的健康区间是 3–8，
用 `目标条数 ÷ 独立图数` 核对，落不进去就是该补数据源的信号，不是该硬出题的信号。

---

## 6. 验收标准

做完之后，这四条必须全绿：

```bash
python tests/qc_fixture.py        # QC 的 10 类检查各命中一次, 反例零告警
python tests/merge_fixture.py     # 合并器对 4 种事故处理正确
python tests/convoy_fixture.py    # 车队判定的 8 个对照场景
python -m pyflakes tools/ tests/
```

`pyflakes` 现有的三条告警是既有的，不要顺手去修（`f-string is missing placeholders`
那一类尤其不要批量正则改 —— 上一次这么干啃坏了 `"鿿"`）：

```
tools/inspect_data.py:273  'PIL' imported but unused
tools/ds/common.py:99      'cv2' imported but unused
tests/qc_fixture.py:11     'json' imported but unused
```

### 还要新增一个对照 fixture

照 `tests/qc_fixture.py` 的写法建 `tests/concentration_fixture.py`，
**正反例都要**，至少覆盖真值表的五行：

| 场景 | 期望 |
|---|---|
| 纯 smoke 图 | 判定题说「存在异常，为烟雾」；出 `locate_*`；不出 `dense_region` |
| DroneCrowd 图（concentration + hard_negative） | 判定题**不含**「存在异常」；出 `dense_region`；`metadata.anomaly == ["normal"]`；`metadata.indicators == ["concentration"]` |
| MAR20 图（air_concentration） | 判定题用 `aircraft_parking` 措辞，**不能**出现「均为民用目标」 |
| smoke + concentration 同框 | 判定题走异常分支，说的是烟雾不是密集分布 |
| 纯正常图 | 现状不变 |

然后对生成结果跑一遍 `qc_consistency.py`，**`normal_but_asserts_anomaly` 必须零命中**。

### 端到端自查

没有真数据时可以用合成数据跑通全链路：

```bash
python tools/adapters.py demo --out /tmp/s.jsonl
python tools/derive_events.py --scenes /tmp/s.jsonl --out /tmp/s_ev.jsonl --overwrite
python tools/build_vqa.py --scenes /tmp/s_ev.jsonl --out-dir /tmp/vqa --target-per-class 200
python tools/qc_consistency.py /tmp/vqa/*.json
```

---

## 7. 红线（这几件事不要做）

1. **不要改 `Scene.anomaly_types`。** 让数据结构去判"算不算异常"是把本体知识漏进
   `scene.py`。过滤在 `RuleBuilder` 里做。
2. **不要把 concentration 塞进 `metadata["anomaly"]`。** 见 A4，那会连带弄坏
   配额、QC、负样本占比三处。
3. **不要为了让某一类数字好看去调阈值。** 本体里每条阈值上面的注释都记着它是
   怎么量出来的（`max_area_per_object` 的 20 倍、`max_gap_obj_mult` 的 3.5 倍）。
   要改先造对照数据自证。
4. **不要写死困难负样本的措辞。** 必须按 `hard_negative_kind` / 事件类型分情况。
   对一片军机说「均为民用目标」是 `qc_consistency` 会抓的自相矛盾。
5. **不要动 `convoy` / `border_crossing` 的规则和测试。** 它们停用是"数据上分不出来"，
   不是"规则写错了"，规则逻辑必须一直可测。
6. **失败要响。** 读不到本体字段、facet 自检不过、零产出 —— 硬失败，
   不要 warn 了继续。这个项目被静默降级坑过四次，每次都是废数据一路混到最后才发现。
7. **改完先跑，别只看语法。** `compileall` 过了不等于跑得对 —— 有过一次把函数
   插进 `main()` 中间，后面整个循环成了 `return` 之后的死代码，语法完全合法。

---

## 8. 上下文在哪

| 想知道 | 看哪 |
|---|---|
| 为什么要推倒重来 | `docs/00_RESTART_BRIEF.md` |
| 类别体系怎么设计的 | `docs/01_taxonomy_draft.md` |
| 数据从哪来、哪些路子不能走 | `docs/02_data_strategy.md` |
| **类别的权威定义** | `configs/ontology.yaml`（文档是过程记录，配置是结论） |
| 需求、工具链、踩过的坑 | `CLAUDE.md` |
| 模型对真实图到底看到了什么 | `docs/15_free_look_findings_raw.md` |
| massing 为什么在 DroneCrowd 上失败 | `docs/16_free_look_massing_recheck.md` |

分析自由看图的结果时**不要用正则统计**。第一次分析用关键词匹配得出
「massing 只有 1/10 确认」，逐条读原文才发现模型明说「一队排列成纵队或楔形编队的
装甲车辆」是真阳性，只是"纵队编队"不在词表里。断章取义比不分析更糟。

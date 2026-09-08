# 指令型 VQA 的生成设计

你的训练目标是**指令模型**，所以数据必须长成指令的样子。但直接让 LLM 看图自由发挥会灾难性地产生幻觉——它会数错车、编出图里没有的坦克、给出臆造的坐标。这些错误进了训练集，模型只会学得更会编。

## 核心架构：事实在前，语言在后

```
Scene 标注
   ↓  规则计算(零幻觉)
FACTS 事实包          目标数量 / bbox 坐标 / 事件类型 / 边界关系 / 缺席的异常类型
   ↓  LLM 第一遍(temperature 0.8, 求多样性)
指令型 QA            LLM 只做"把事实改写成多样化的指令问答", 不做判断
   ↓  LLM 第二遍(temperature 0, 换一个模型)
校验裁决             pass / fix / drop
   ↓
ShareGPT 训练数据
```

**LLM 的职责被严格限定为语言表达，判断权在规则手里。** 这是这套设计唯一重要的地方。

`tools/llm_qa.py` 的 prompt 里写死了六条硬约束（见 `configs/prompts/generate_qa.txt`），其中三条最关键：

- 数量必须与 `objects_summary` 完全一致，不许估算
- 坐标只能直接引用 FACTS 里的 `box_1000`，不许自己看图估
- **`events` 为空时答案必须是"未见异常"**，不得因为画面里有车有人就判为异常

FACTS 里还专门放了一个 `absent_anomaly_types` 字段，就是为了让 LLM 有据可依地生成否定题——问一个确定不存在的异常类型。

## 校验必须换一个模型

第二遍校验**不要用生成时的同一个模型**。同模型自查会系统性地放过自己的错误——它按同样的方式理解图像，自然认为自己的输出是对的。用另一个厂商或另一个尺寸的模型做校验，才能真正挑出问题。

`--no-verify` 存在只是为了调试，**正式跑数据不要用**。

监控指标：**drop 率**。
- < 5%：约束生效，正常
- 5–15%：可接受
- **> 15%：生成端在编造事实**，回去收紧 prompt 或把 temperature 降到 0.5 以下

`llm_qa.py verify` 会在 drop 率超 15% 时直接打 warning。

## 模型与温度的分工

| 环节 | 模型建议 | temperature | 调用量 | 说明 |
|---|---|---|---|---|
| 闸5 图像质检 | 7B 级 VL | **0.0** | 每图 1 次 | 量最大，用小模型，只做分类判断 |
| QA 生成 | 最强的 VL（72B 或 API） | **0.8** | 每图 1 次 | 多样性全靠这一步，别省 |
| QA 校验 | 中等尺寸，**换一家** | **0.0** | 每图 1 次 | 只做事实比对，不需要创造力 |

生成环节的 temperature 不要低于 0.6，否则所有问题都长一个样，指令多样性就没了；也不要超过 1.0，会开始飘。

## 指令多样性怎么保证

只靠 temperature 不够，必须在 prompt 里显式要求覆盖多种**指令风格**：

| 风格 | 示例 |
|---|---|
| `direct` 直接提问 | "画面中是否存在异常军事活动？" |
| `task` 任务式指令 | "请分析该航拍画面，列出所有可疑目标及其位置。" |
| `roleplay` 角色设定 | "作为值班分析员，判断这张图是否需要上报，并说明理由。" |
| `format` 指定输出格式 | "用 JSON 输出，字段为 anomaly / regions / evidence / confidence。" |
| `multiturn` 多轮追问 | "有异常吗？" → "具体在哪个区域？" → "判定依据是什么？" |

**`format` 风格务必保留一定比例**。真实系统对接需要结构化输出，如果训练数据里全是自然语言散文，模型上线后你会发现它没法稳定吐 JSON。固定字段：

```json
{"anomaly": "massing", "regions": [[280,150,700,430]],
 "evidence": "15 个车辆目标密集成簇，间距均匀", "confidence": "high"}
```

`tools/check_dataset.py` 会打印指令风格分布和对话轮数分布，用它盯着别让某一种风格占比失控。

## 规则生成的 QA 不要丢

**规则生成（`build_vqa.py`）和 LLM 生成（`llm_qa.py`）应该同时保留，各占一半左右。**

| | 规则生成 | LLM 生成 |
|---|---|---|
| 计数题、grounding 题 | ✅ 答案是精确真值 | ⚠️ 可能漂移 |
| 语言多样性 | ❌ 模板味重 | ✅ |
| 推理与描述 | ❌ 骨架生硬 | ✅ |
| 成本 | 零 | 每图 2 次调用 |

建议配比：**规则 40%（主要承担计数与 grounding）+ LLM 60%（主要承担描述、推理、多轮、格式化输出）**。

计数和 grounding 这两类**优先用规则的版本**——这两项的价值就在于答案精确，交给 LLM 反而是把确定性换成不确定性。

## 成本估算

按筛选后 4 万张图、每图 6 条 QA 计：

- 闸5 质检：约 6 万次调用（筛选前的量），7B 本地 vLLM，A100 单卡约半天
- 生成：4 万次调用（72B）
- 校验：4 万次调用（中等模型，纯文本，无图，很快）
- 产出：约 24 万条 LLM QA + 16 万条规则 QA ≈ **40 万条指令数据**

全程有磁盘缓存（`--cache-dir`，按请求哈希），中断重跑不会重复计费。想完全离线就加 `--dry-run` 导出请求 jsonl，拿 vLLM 批推理跑完再回灌。

## 完整流水线

```bash
# 1. 归一化
python tools/adapters.py <sub> ... --out data/interim/<ds>.jsonl
cat data/interim/*.jsonl > data/interim/all.jsonl

# 2. 规则派生事件
python tools/derive_events.py --scenes data/interim/all.jsonl --out data/interim/all_ev.jsonl

# 3. 闸1-4 规则筛选
python tools/screen.py --scenes data/interim/all_ev.jsonl --out-dir data/screened

# 4. 闸5 VLM 质检, 再合并重筛
python tools/llm_qa.py review --scenes data/screened/scenes_kept.jsonl \
    --model qwen2.5-vl-7b-instruct --out data/screened/vlm_review.jsonl
python tools/screen.py --scenes data/interim/all_ev.jsonl --out-dir data/screened \
    --vlm-review data/screened/vlm_review.jsonl

# 5a. 规则 QA(计数/grounding 主力)
python tools/build_vqa.py --scenes data/screened/scenes_kept.jsonl --out-dir data/vqa_rule

# 5b. LLM 指令 QA(描述/推理/多轮主力)
python tools/llm_qa.py generate --scenes data/screened/scenes_kept.jsonl \
    --model qwen2.5-vl-72b-instruct --temperature 0.8 --out data/interim/gen.jsonl
python tools/llm_qa.py verify --generated data/interim/gen.jsonl \
    --model <换一个模型> --out data/vqa_llm/all.json

# 6. 体检
python tools/check_dataset.py data/vqa_rule/train.json data/vqa_llm/all.json
```

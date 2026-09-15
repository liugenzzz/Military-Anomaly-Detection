#!/usr/bin/env bash
# 一条命令跑完: 预处理 -> 事件派生 -> 质量筛选 -> golden set -> 规则生成 -> LLM 生成
#
#   bash run_all.sh /mnt/si003010kcx0/mmdata/data_process/military
#
# 没下到的数据集会自动跳过并在末尾列出, 不会中断流程。
set -uo pipefail
DATA="${1:?用法: bash run_all.sh <数据根目录> [输出目录]}"
OUT="${2:-${OUT_DIR:-data}}"       # 输出目录: 第二个参数, 或环境变量 OUT_DIR
PY="${PYTHON:-python3}"

# ── LLM 生成: 配了端点就真跑, 没配就只导出请求(离线批推理也能吃这个文件)
#   VLM_BASE_URL=http://192.168.78.36:3012/v1 VLM_MODEL=Qwen3.6-27B bash run_all.sh <数据根>
VLM_BASE_URL="${VLM_BASE_URL:-}"
VLM_MODEL="${VLM_MODEL:-Qwen3.8-27B}"
# 端点池配置(推荐)。填了它就不用 VLM_BASE_URL, 每一路各带自己的 model 与 key。
#   cp configs/endpoints.yaml.example configs/endpoints.local.yaml   # 再填真实 key
ENDPOINTS="${ENDPOINTS:-}"
[ -z "$ENDPOINTS" ] && [ -f configs/generate.yaml ] && ENDPOINTS=configs/generate.yaml
[ -z "$ENDPOINTS" ] && [ -f configs/endpoints.local.yaml ] && ENDPOINTS=configs/endpoints.local.yaml
# review 最好换一个模型: 同一个模型审自己写的答案基本全过, 六维形同虚设
REVIEW_MODEL="${REVIEW_MODEL:-$VLM_MODEL}"
TARGET_PER_CLASS="${TARGET_PER_CLASS:-25000}"   # 每个异常类的 QA 总目标
DESC_SHARE="${DESC_SHARE:-60}"             # 其中描述+推理(LLM 侧)占几成, 单位 %
LLM_TARGET=$(( TARGET_PER_CLASS * DESC_SHARE / 100 ))
RULE_TARGET=$(( TARGET_PER_CLASS - LLM_TARGET ))
WORKERS="${WORKERS:-0}"    # 0 = 跟着端点池的总并发走
FACETS_PER_IMAGE="${FACETS_PER_IMAGE:-4}"
# 试水批: 只跑 N 个 scene(按类别分层抽), 用来在烧算力之前先看看答案写成什么样
LLM_SAMPLE="${LLM_SAMPLE:-0}"
# 有多少比例的图把"判定 -> 带框描述 -> 异常说明"拼成一条多轮对话。
# 全拼会让模型以为回答必须是三段, 单问一句也长篇大论, 所以只拼三成。
CHAIN_RATIO="${CHAIN_RATIO:-0.30}"
INLINE_IMAGES="${INLINE_IMAGES:-0}"        # 远端 API 要 base64 内联; 本地 vLLM 挂同一块盘就不用
RELAX_BELOW="${RELAX_BELOW:-3000}"         # 产量低于这么多张图的类启用放宽档补量, 0 关闭
# 抽帧间隔。与 configs/ontology.yaml 的 stride 保持一致: 30 对 DroneCrowd 和
# VisDrone-MOT 太稀(24960 帧只取 832), 近重复由 screen.py 的闸3 兜底。
STRIDE_CROWD="${STRIDE_CROWD:-8}"
STRIDE_MOT="${STRIDE_MOT:-4}"   # 越界类的帧数就是它的产量上限, 比别的源抽得密一点
# 每个 MOT 序列铺几条平行边界线。只铺一条的话, 一个序列里只有中间几帧算越界,
# border_crossing 的产量就卡在那; 铺开之后不同的帧被不同的线截住, 同一段素材
# 产出的是**不同的**问答(越界目标、时机、方向都不同)。但图还是那些图,
# 调太高只是在同一批画面上反复出题, 别超过 5。
MOT_BOUNDARIES="${MOT_BOUNDARIES:-4}"
STRIDE_DA="${STRIDE_DA:-10}"
# 媒体归集: 填了就把引用到的图片/视频硬链到语料库目录, 并把 json 里的路径改过去
#   MEDIA_ROOT=/mnt/si003010kcx0/mmdata/data_process/corpus_media
MEDIA_ROOT="${MEDIA_ROOT:-}"
MEDIA_NAME="${MEDIA_NAME:-military_anomaly}"
MEDIA_MODE="${MEDIA_MODE:-hardlink}"       # hardlink | symlink | copy

mkdir -p "$OUT"/{interim,screened,golden,vqa_rule,vqa_llm,frames,tiles}
SKIPPED=()

# 解析数据集目录: 先按给定名字找, 再按"名字 + 很短的尾巴"找(MAR201 / MAR20_v2
# 这种手动解压时改过的名字), 找到有内容的就返回其路径。
has_media() {
  # 后缀要忽略大小写(很多数据集是 .JPG), 扩展名要全, 深度要够 ——
  # 只认四种小写后缀、最深四层的话, Mendeley 这种 .JPG/嵌套深的数据集会被判成"没下到"
  # 而直接跳过, 而 doctor 那边用的是全树 rglob, 两边结论会打架。
  [ -n "$(find "$1" -maxdepth 6 \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
          -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \
          -o -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' \) \
          -print -quit 2>/dev/null)" ]
}
resolve() {
  # 按名字从具体到笼统逐个试; 每个名字都走"本层 -> 下一层 -> 近似名"三步,
  # 试完一个名字再换下一个 —— 顺序不能打乱, 否则找 MOT 会先撞上笼统的 VisDrone/
  for n in "$@"; do
    for d in "$DATA/$n" "$DATA"/*/"$n" "$DATA/$n"? "$DATA/$n"?? "$DATA/$n"[-_.\ ]*; do
      [ -d "$d" ] && has_media "$d" && { echo "$d"; return 0; }
    done
  done
  return 1
}
# 已经跑出来的中间文件就别重跑。某一个数据集炸了(比如 DOTA 撞上巨图)时,
# 补跑那一个即可, 不用把另外八个几万张图再走一遍。
SKIP_PREPARED="${SKIP_PREPARED:-0}"
prepared() {
  [ "$SKIP_PREPARED" = "1" ] && [ -s "$1" ] && { echo "  [跳过] $(basename "$1") 已存在"; return 0; }
  return 1
}
step() { echo; echo "──── $* ────"; }
run()  { if "$@"; then return 0; else echo "  [失败] $*"; return 1; fi; }

echo "配额: 每类 $TARGET_PER_CLASS 条 = 描述/推理 $LLM_TARGET (${DESC_SHARE}%) + 规则 $RULE_TARGET"
echo "════ 0. 数据体检 ════"
$PY tools/doctor.py --root "$DATA" || true

step "1. 预处理"
if D=$(resolve MAR20 mar20);           then prepared "$OUT/interim/mar20.jsonl" || run $PY tools/prepare.py mar20 --root "$D" --out "$OUT/interim/mar20.jsonl"; else SKIPPED+=("MAR20"); fi
if D=$(resolve Mendeley-UAV-Military mendeley Mendeley); then prepared "$OUT/interim/mendeley.jsonl" || run $PY tools/prepare.py mendeley --root "$D" --out "$OUT/interim/mendeley.jsonl"; else SKIPPED+=("Mendeley"); fi
if D=$(resolve FASDD_UAV FASDD);       then prepared "$OUT/interim/fasdd.jsonl" || run $PY tools/prepare.py fasdd --root "$D" --out "$OUT/interim/fasdd.jsonl"; else SKIPPED+=("FASDD_UAV(smoke 唯一来源)"); fi
if D=$(resolve ERA era); then
  CAP=$(find "$DATA" -maxdepth 3 -iname '*.json' -ipath '*cap*' -print -quit 2>/dev/null)
  prepared "$OUT/interim/era.jsonl" || run $PY tools/prepare.py era --root "$D" --modality video --single-frames \
      ${CAP:+--capera "$CAP"} --out "$OUT/interim/era.jsonl"
else SKIPPED+=("ERA"); fi
if D=$(resolve DroneCrowd dronecrowd); then prepared "$OUT/interim/dronecrowd.jsonl" || run $PY tools/prepare.py dronecrowd --root "$D" --stride "$STRIDE_CROWD" --out "$OUT/interim/dronecrowd.jsonl"; else SKIPPED+=("DroneCrowd"); fi
if D=$(resolve VisDrone2019-DET-train VisDrone-DET VisDrone/VisDrone2019-DET-train); then
  prepared "$OUT/interim/vd_det.jsonl" || run $PY tools/prepare.py visdrone-det --root "$D" --out "$OUT/interim/vd_det.jsonl"
fi
# MOT 的 train / val / test-dev 都收。越界是四类里唯一缺口大的, 而它只有这一个
# 数据源 —— 与其在同一批画面上反复出题, 不如先把同数据集其他 split 的真实素材用上。
# 这些 split 只是 VisDrone 官方的划分, 与我们自己的 train/test 切分无关。
N_MOT=0
for SPLIT in train val test-dev; do
  D=$(resolve "VisDrone2019-MOT-$SPLIT" "VisDrone-MOT-$SPLIT" "VisDrone/VisDrone2019-MOT-$SPLIT") || continue
  OUTF="$OUT/interim/vd_mot_${SPLIT/-/}.jsonl"
  prepared "$OUTF" || run $PY tools/prepare.py visdrone-mot --root "$D" --stride "$STRIDE_MOT" \
      --boundaries-per-seq "$MOT_BOUNDARIES" --out "$OUTF"
  N_MOT=$((N_MOT + 1))
done
[ "$N_MOT" -gt 0 ] || SKIPPED+=("VisDrone-MOT(border_crossing 唯一来源)")
if D=$(resolve DOTA DOTA-v2.0 dota);   then prepared "$OUT/interim/dota.jsonl" || run $PY tools/prepare.py dota --root "$D" --tiles-dir "$OUT/tiles/dota" --out "$OUT/interim/dota.jsonl"; else SKIPPED+=("DOTA"); fi
if D=$(resolve Drone-Anomaly drone_anomaly); then prepared "$OUT/interim/drone_anomaly.jsonl" || run $PY tools/prepare.py drone-anomaly --root "$D" --stride "$STRIDE_DA" --out "$OUT/interim/drone_anomaly.jsonl"; else SKIPPED+=("Drone-Anomaly"); fi

shopt -s nullglob
# 只合并本流程自己产出的这几个文件, 不用 *.jsonl 通配 ——
# interim/ 下还躺着历史 demo 数据和 llm_requests.jsonl(那是请求不是 scene),
# 通配会把它们一起 cat 进去: 上一轮就混进了 40 条 DEMO-synthetic, 还制造了
# 两千多个 duplicate_scene_id。
INTERIM_FILES=(mar20 mendeley fasdd era dronecrowd vd_det
                vd_mot_train vd_mot_val vd_mot_testdev dota drone_anomaly)
FILES=()
for n in "${INTERIM_FILES[@]}"; do
  f="$OUT/interim/$n.jsonl"
  [ -s "$f" ] && FILES+=("$f")
done
[ ${#FILES[@]} -eq 0 ] && { echo "没有任何可用数据, 退出"; exit 1; }
# 用 awk 1 而不是 cat: 某个 jsonl 少了结尾换行时, cat 会把两条记录粘成一行,
# 而且是静默的 —— 下一步解析才报 "Extra data", 那时已经看不出是哪两条。
awk 1 "${FILES[@]}" > "$OUT/all_scenes.jsonl"
echo "合并 ${#FILES[@]} 个来源 -> $OUT/all_scenes.jsonl ($(wc -l < "$OUT/all_scenes.jsonl") 个 scene)"

step "2. 事件派生"
run $PY tools/derive_events.py --scenes "$OUT/all_scenes.jsonl" --out "$OUT/all_ev.jsonl" \
    --relax-below "$RELAX_BELOW"
step "3. 质量筛选";  run $PY tools/screen.py --scenes "$OUT/all_ev.jsonl" --out-dir "$OUT/screened"
KEPT="$OUT/screened/scenes_kept.jsonl"; [ -s "$KEPT" ] || KEPT="$OUT/all_ev.jsonl"
step "4. golden set"
N_SCENE=$(wc -l < "$KEPT")
PER_CELL=12; [ "$N_SCENE" -lt 2000 ] && PER_CELL=$(( N_SCENE / 100 + 1 ))
echo "  scene 总数 $N_SCENE, 每格取 $PER_CELL 张"
run $PY tools/make_golden.py --scenes "$KEPT" --out-dir "$OUT/golden" --per-cell "$PER_CELL"
EXC="$OUT/golden/exclude_ids.txt"
step "5. 规则生成"
run $PY tools/build_vqa.py --scenes "$KEPT" --out-dir "$OUT/vqa_rule" \
    --target-per-class "$RULE_TARGET" ${EXC:+--exclude-ids "$EXC"}

GEN="$OUT/interim/llm_generated.jsonl"
INLINE=(); [ "$INLINE_IMAGES" = "1" ] && INLINE=(--inline-images)
if [ -n "$VLM_BASE_URL" ] || [ -n "$ENDPOINTS" ]; then
  if [ -n "$ENDPOINTS" ]; then WHERE="端点池 $ENDPOINTS"; else WHERE="$VLM_MODEL @ $VLM_BASE_URL"; fi
  step "6a. LLM 生成描述/推理 ($WHERE)"
  # 先体检一遍。某一路模型名对不上, 那一路的产出全是废的, 早发现早改
  $PY tools/llm_qa.py ping ${ENDPOINTS:+--endpoints "$ENDPOINTS"} \
      ${VLM_BASE_URL:+--base-url "$VLM_BASE_URL"} --model "$VLM_MODEL" 2>&1 | sed 's/^/  /'

  run $PY tools/llm_qa.py generate --scenes "$KEPT" \
      --facets-per-image "$FACETS_PER_IMAGE" --target-per-class "$LLM_TARGET" \
      $( [ "$LLM_SAMPLE" != 0 ] && echo "--sample $LLM_SAMPLE" ) ${ENDPOINTS:+--endpoints "$ENDPOINTS"} ${VLM_BASE_URL:+--base-url "$VLM_BASE_URL"} \
      --model "$VLM_MODEL" --workers "$WORKERS" \
      ${INLINE[@]+"${INLINE[@]}"} ${EXC:+--exclude-ids "$EXC"} --out "$GEN"
  if [ -s "$GEN" ]; then
    if [ -n "$ENDPOINTS" ]; then RWHERE="审稿端点见 $ENDPOINTS 的 review 组"; else RWHERE="审稿模型 $REVIEW_MODEL"; fi
    step "6b. must-not 硬过滤 + 六维 review ($RWHERE)"
    [ -z "$ENDPOINTS" ] && [ "$REVIEW_MODEL" = "$VLM_MODEL" ] && \
      echo "  [注意] 审稿和生成是同一个模型, 自己审自己会虚高, 建议 REVIEW_MODEL 换一个"
    run $PY tools/llm_qa.py verify --generated "$GEN" \
        ${ENDPOINTS:+--endpoints "$ENDPOINTS"} ${VLM_BASE_URL:+--base-url "$VLM_BASE_URL"} \
        --model "$REVIEW_MODEL" --workers "$WORKERS" \
        ${INLINE[@]+"${INLINE[@]}"} --out "$OUT/vqa_llm/all.json"
  else
    echo "  生成结果为空, 跳过 review"
  fi
else
  step "6. LLM 生成(只导出请求, 未配 VLM_BASE_URL)"
  run $PY tools/llm_qa.py generate --scenes "$KEPT" \
      --facets-per-image "$FACETS_PER_IMAGE" --target-per-class "$LLM_TARGET" \
      ${EXC:+--exclude-ids "$EXC"} --dry-run --out "$OUT/interim/llm_requests.jsonl"
  echo "  请求已导出。配好服务后这样跑完整流程:"
  echo "    VLM_BASE_URL=http://192.168.78.36:3012/v1 VLM_MODEL=$VLM_MODEL \\"
  echo "    REVIEW_MODEL=<换一个模型> bash run_all.sh $DATA $OUT"
fi

step "7. 合并为 LLaMA-Factory 数据集"
run $PY tools/merge_dataset.py --in "$OUT/vqa_rule" "$OUT/vqa_llm" --out "$OUT/vqa" \
    --chain-ratio "$CHAIN_RATIO"

if [ -n "$MEDIA_ROOT" ]; then
  step "7b. 媒体归集 -> $MEDIA_ROOT/$MEDIA_NAME/{images,videos}/"
  run $PY tools/export_media.py --vqa-dir "$OUT/vqa" \
      --media-root "$MEDIA_ROOT" --name "$MEDIA_NAME" --mode "$MEDIA_MODE"
fi

step "8. 数据体检"
shopt -s nullglob
CHK=("$OUT"/vqa/train*.json "$OUT"/vqa/val*.json "$OUT"/vqa/test*.json)
if [ ${#CHK[@]} -gt 0 ]; then run $PY tools/check_dataset.py "${CHK[@]}"; else echo "  无输出可检"; fi

echo; echo "════ 完成 ════"
if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "跳过的数据集(未下到或未解压):"; printf '  - %s\n' "${SKIPPED[@]}"
fi

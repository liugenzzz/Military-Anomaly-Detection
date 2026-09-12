#!/usr/bin/env bash
# 一条命令跑完: 预处理 -> 事件派生 -> 质量筛选 -> golden set -> 规则生成 -> LLM 生成
#
#   bash run_all.sh /mnt/si003010kcx0/mmdata/mm_general/military
#
# 没下到的数据集会自动跳过并在末尾列出, 不会中断流程。
set -uo pipefail
DATA="${1:?用法: bash run_all.sh <数据根目录> [输出目录]}"
OUT="${2:-${OUT_DIR:-data}}"       # 输出目录: 第二个参数, 或环境变量 OUT_DIR
PY="${PYTHON:-python3}"

# ── LLM 生成: 配了端点就真跑, 没配就只导出请求(离线批推理也能吃这个文件)
#   VLM_BASE_URL=http://127.0.0.1:8000/v1 VLM_MODEL=qwen2.5-vl-72b-instruct bash run_all.sh <数据根>
VLM_BASE_URL="${VLM_BASE_URL:-}"
VLM_MODEL="${VLM_MODEL:-Qwen3.6-27B}"   # 与目标检测那个项目同一套服务
# review 最好换一个模型: 同一个模型审自己写的答案基本全过, 六维形同虚设
REVIEW_MODEL="${REVIEW_MODEL:-$VLM_MODEL}"
TARGET_PER_CLASS="${TARGET_PER_CLASS:-25000}"   # 每个异常类的 QA 总目标
DESC_SHARE="${DESC_SHARE:-60}"             # 其中描述+推理(LLM 侧)占几成, 单位 %
LLM_TARGET=$(( TARGET_PER_CLASS * DESC_SHARE / 100 ))
RULE_TARGET=$(( TARGET_PER_CLASS - LLM_TARGET ))
WORKERS="${WORKERS:-8}"
FACETS_PER_IMAGE="${FACETS_PER_IMAGE:-4}"
INLINE_IMAGES="${INLINE_IMAGES:-0}"        # 远端 API 要 base64 内联; 本地 vLLM 挂同一块盘就不用
RELAX_BELOW="${RELAX_BELOW:-3000}"         # 产量低于这么多张图的类启用放宽档补量, 0 关闭
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
  [ -n "$(find "$1" -maxdepth 4 \( -name '*.jpg' -o -name '*.png' -o -name '*.mp4' -o -name '*.avi' \) -print -quit 2>/dev/null)" ]
}
resolve() {
  for n in "$@"; do
    d="$DATA/$n"
    [ -d "$d" ] && has_media "$d" && { echo "$d"; return 0; }
  done
  for n in "$@"; do
    for d in "$DATA/$n"?  "$DATA/$n"??  "$DATA/$n"[-_.\ ]*; do
      [ -d "$d" ] && has_media "$d" && { echo "$d"; return 0; }
    done
  done
  return 1
}
step() { echo; echo "──── $* ────"; }
run()  { if "$@"; then return 0; else echo "  [失败] $*"; return 1; fi; }

echo "配额: 每类 $TARGET_PER_CLASS 条 = 描述/推理 $LLM_TARGET (${DESC_SHARE}%) + 规则 $RULE_TARGET"
echo "════ 0. 数据体检 ════"
$PY tools/doctor.py --root "$DATA" || true

step "1. 预处理"
if D=$(resolve MAR20 mar20);           then run $PY tools/prepare.py mar20 --root "$D" --out "$OUT/interim/mar20.jsonl"; else SKIPPED+=("MAR20"); fi
if D=$(resolve Mendeley-UAV-Military mendeley Mendeley); then run $PY tools/prepare.py mendeley --root "$D" --out "$OUT/interim/mendeley.jsonl"; else SKIPPED+=("Mendeley"); fi
if D=$(resolve FASDD_UAV FASDD);       then run $PY tools/prepare.py fasdd --root "$D" --out "$OUT/interim/fasdd.jsonl"; else SKIPPED+=("FASDD_UAV(smoke 唯一来源)"); fi
if D=$(resolve ERA era); then
  CAP=$(find "$DATA" -maxdepth 3 -iname '*.json' -ipath '*cap*' -print -quit 2>/dev/null)
  run $PY tools/prepare.py era --root "$D" --modality video --single-frames \
      ${CAP:+--capera "$CAP"} --out "$OUT/interim/era.jsonl"
else SKIPPED+=("ERA"); fi
if D=$(resolve DroneCrowd dronecrowd); then run $PY tools/prepare.py dronecrowd --root "$D" --stride 30 --out "$OUT/interim/dronecrowd.jsonl"; else SKIPPED+=("DroneCrowd"); fi
if D=$(resolve VisDrone2019-DET-train VisDrone-DET VisDrone/VisDrone2019-DET-train); then
  run $PY tools/prepare.py visdrone-det --root "$D" --out "$OUT/interim/vd_det.jsonl"
fi
if D=$(resolve VisDrone2019-MOT-train VisDrone-MOT VisDrone/VisDrone2019-MOT-train); then
  run $PY tools/prepare.py visdrone-mot --root "$D" --stride 30 --out "$OUT/interim/vd_mot.jsonl"
fi
[ -f "$OUT/interim/vd_mot.jsonl" ] || SKIPPED+=("VisDrone-MOT(border_crossing 唯一来源)")
if D=$(resolve DOTA DOTA-v2.0 dota);   then run $PY tools/prepare.py dota --root "$D" --tiles-dir "$OUT/tiles/dota" --out "$OUT/interim/dota.jsonl"; else SKIPPED+=("DOTA"); fi
if D=$(resolve Drone-Anomaly drone_anomaly); then run $PY tools/prepare.py drone-anomaly --root "$D" --stride 10 --out "$OUT/interim/drone_anomaly.jsonl"; else SKIPPED+=("Drone-Anomaly"); fi

shopt -s nullglob
FILES=("$OUT"/interim/*.jsonl)
[ ${#FILES[@]} -eq 0 ] && { echo "没有任何可用数据, 退出"; exit 1; }
cat "${FILES[@]}" > "$OUT/all_scenes.jsonl"          # 显式列文件, 不用 *.jsonl 以免把输出 cat 进去
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
if [ -n "$VLM_BASE_URL" ]; then
  step "6a. LLM 生成描述/推理 (模型 $VLM_MODEL @ $VLM_BASE_URL)"
  run $PY tools/llm_qa.py generate --scenes "$KEPT" \
      --facets-per-image "$FACETS_PER_IMAGE" --target-per-class "$LLM_TARGET" \
      --base-url "$VLM_BASE_URL" --model "$VLM_MODEL" --workers "$WORKERS" \
      ${INLINE[@]+"${INLINE[@]}"} ${EXC:+--exclude-ids "$EXC"} --out "$GEN"
  if [ -s "$GEN" ]; then
    step "6b. must-not 硬过滤 + 六维 review (审稿模型 $REVIEW_MODEL)"
    [ "$REVIEW_MODEL" = "$VLM_MODEL" ] && \
      echo "  [注意] 审稿和生成是同一个模型, 自己审自己会虚高, 建议 REVIEW_MODEL 换一个"
    run $PY tools/llm_qa.py verify --generated "$GEN" \
        --base-url "$VLM_BASE_URL" --model "$REVIEW_MODEL" --workers "$WORKERS" \
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
run $PY tools/merge_dataset.py --in "$OUT/vqa_rule" "$OUT/vqa_llm" --out "$OUT/vqa"

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

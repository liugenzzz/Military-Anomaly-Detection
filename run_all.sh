#!/usr/bin/env bash
# 一条命令跑完: 预处理 -> 事件派生 -> 质量筛选 -> golden set -> 规则生成 -> LLM 生成
#
#   bash run_all.sh /mnt/si003010kcx0/mmdata/mm_general/military
#
# 没下到的数据集会自动跳过并在末尾列出, 不会中断流程。
set -uo pipefail
DATA="${1:?用法: bash run_all.sh <数据根目录>}"
OUT="${2:-data}"
PY="${PYTHON:-python3}"

mkdir -p "$OUT"/{interim,screened,golden,vqa_rule,vqa_llm,frames,tiles}
SKIPPED=()

# 解析数据集目录: 依次试几个常见命名, 找到有内容的就返回其路径
resolve() {
  for n in "$@"; do
    d="$DATA/$n"
    [ -d "$d" ] && [ -n "$(find "$d" -maxdepth 4 \( -name '*.jpg' -o -name '*.png' -o -name '*.mp4' -o -name '*.avi' \) -print -quit 2>/dev/null)" ] && { echo "$d"; return 0; }
  done
  return 1
}
step() { echo; echo "──── $* ────"; }
run()  { if "$@"; then return 0; else echo "  [失败] $*"; return 1; fi; }

echo "════ 0. 数据体检 ════"
$PY tools/doctor.py --root "$DATA" || true

step "1. 预处理"
if D=$(resolve MAR20 mar20);           then run $PY tools/prepare.py mar20 --root "$D" --out "$OUT/interim/mar20.jsonl"; else SKIPPED+=("MAR20"); fi
if D=$(resolve Mendeley-UAV-Military mendeley Mendeley); then run $PY tools/prepare.py mendeley --root "$D" --out "$OUT/interim/mendeley.jsonl"; else SKIPPED+=("Mendeley"); fi
if D=$(resolve FASDD_UAV FASDD);       then run $PY tools/prepare.py fasdd --root "$D" --out "$OUT/interim/fasdd.jsonl"; else SKIPPED+=("FASDD_UAV(smoke 唯一来源)"); fi
if D=$(resolve ERA era); then
  CAP=$(find "$DATA" -maxdepth 3 -iname '*.json' -ipath '*cap*' -print -quit 2>/dev/null)
  run $PY tools/prepare.py era --root "$D" --modality video \
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

step "2. 事件派生";  run $PY tools/derive_events.py --scenes "$OUT/all_scenes.jsonl" --out "$OUT/all_ev.jsonl"
step "3. 质量筛选";  run $PY tools/screen.py --scenes "$OUT/all_ev.jsonl" --out-dir "$OUT/screened"
KEPT="$OUT/screened/scenes_kept.jsonl"; [ -s "$KEPT" ] || KEPT="$OUT/all_ev.jsonl"
step "4. golden set"
N_SCENE=$(wc -l < "$KEPT")
PER_CELL=12; [ "$N_SCENE" -lt 2000 ] && PER_CELL=$(( N_SCENE / 100 + 1 ))
echo "  scene 总数 $N_SCENE, 每格取 $PER_CELL 张"
run $PY tools/make_golden.py --scenes "$KEPT" --out-dir "$OUT/golden" --per-cell "$PER_CELL"
EXC="$OUT/golden/exclude_ids.txt"
step "5. 规则生成";  run $PY tools/build_vqa.py --scenes "$KEPT" --out-dir "$OUT/vqa_rule" ${EXC:+--exclude-ids "$EXC"}
step "6. LLM 生成(导出请求)"
run $PY tools/llm_qa.py generate --scenes "$KEPT" --facets-per-image 4 \
    ${EXC:+--exclude-ids "$EXC"} --dry-run --out "$OUT/interim/llm_requests.jsonl"
echo "  请求已导出。配好 VLM 服务后去掉 --dry-run 重跑, 再执行:"
echo "    $PY tools/llm_qa.py verify --generated <生成结果> --model <换一个模型> --out $OUT/vqa_llm/all.json"

step "7. 数据体检"
CHK=("$OUT"/vqa_rule/*.json)
if [ ${#CHK[@]} -gt 0 ]; then run $PY tools/check_dataset.py "${CHK[@]}"; else echo "  无输出可检"; fi

echo; echo "════ 完成 ════"
if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "跳过的数据集(未下到或未解压):"; printf '  - %s\n' "${SKIPPED[@]}"
fi

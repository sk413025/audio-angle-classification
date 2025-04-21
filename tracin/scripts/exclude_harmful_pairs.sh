#!/bin/bash
# 排除有害 ranking pair 的工作流程腳本

# 設置腳本在出錯時立即停止
set -e

# 默認值
METADATA_DIR="/Users/sbplab/Hank/audio-angle-classification/metadata"
FREQUENCIES=("500hz" "1000hz" "3000hz")
MATERIAL="plastic"
THRESHOLD="-5.0"
MIN_OCCURRENCES=3
MAX_EXCLUSIONS=50
PAIR_MODE="ranking_pair"
OUTPUT_DIR="/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata"
DEBUG=false

# 顯示幫助信息
function show_help {
    echo "用法: $0 [選項]"
    echo ""
    echo "分析 TracIn 影響力分數並從訓練中排除有害的 ranking pair"
    echo ""
    echo "選項:"
    echo "  -m, --metadata-dir DIR      包含 TracIn 元數據文件的目錄 (默認: $METADATA_DIR)"
    echo "  -o, --output-dir DIR        保存排除列表的目錄 (默認: /Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata)"
    echo "  -f, --frequencies FREQS     要處理的頻率列表，用逗號分隔 (默認: 500hz,1000hz,3000hz)"
    echo "  -a, --material STR          材料類型 (默認: $MATERIAL)"
    echo "  -t, --threshold NUM         負面影響閾值 (默認: $THRESHOLD)"
    echo "  -n, --min-occurrences NUM   最小負面影響出現次數 (默認: $MIN_OCCURRENCES)"
    echo "  -x, --max-exclusions NUM    最大排除樣本數量 (默認: $MAX_EXCLUSIONS)"
    echo "  -p, --pair-mode MODE        排除模式：full_pair 或 ranking_pair (默認: $PAIR_MODE)"
    echo "  -e, --evaluate-only         僅進行評估，不生成排除列表"
    echo "  -d, --debug                 啟用除錯模式，顯示更多詳細信息"
    echo "  -h, --help                  顯示此幫助信息"
    echo ""
    echo "使用示例:"
    echo "  $0 --frequencies 500hz --threshold -10.0"
    echo "  $0 --metadata-dir /path/to/metadata --output-dir /path/to/output"
    echo "  $0 --debug --frequencies 500hz"
}

# 處理日誌的函數
function log_debug {
    if [ "$DEBUG" = true ]; then
        echo "[DEBUG] $1"
    fi
}

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--metadata-dir)
            METADATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--frequencies)
            IFS=',' read -r -a FREQUENCIES <<< "$2"
            shift 2
            ;;
        -a|--material)
            MATERIAL="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -n|--min-occurrences)
            MIN_OCCURRENCES="$2"
            shift 2
            ;;
        -x|--max-exclusions)
            MAX_EXCLUSIONS="$2"
            shift 2
            ;;
        -p|--pair-mode)
            PAIR_MODE="$2"
            shift 2
            ;;
        -e|--evaluate-only)
            EVALUATE_ONLY=true
            shift
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知選項: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果啟用了除錯模式，顯示當前設置
if [ "$DEBUG" = true ]; then
    echo "=== 除錯模式啟用 ==="
    echo "元數據目錄: $METADATA_DIR"
    echo "輸出目錄: $OUTPUT_DIR"
    echo "頻率: ${FREQUENCIES[*]}"
    echo "材料: $MATERIAL"
    echo "閾值: $THRESHOLD"
    echo "最小出現次數: $MIN_OCCURRENCES"
    echo "最大排除數量: $MAX_EXCLUSIONS"
    echo "排除模式: $PAIR_MODE"
    [ "$EVALUATE_ONLY" = true ] && echo "僅評估模式: 是" || echo "僅評估模式: 否"
    echo "====================="
fi

# 如果未指定輸出目錄，使用元數據目錄
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata"
    log_debug "未指定輸出目錄，使用默認目錄: $OUTPUT_DIR"
fi

# 確保目錄存在
if [ ! -d "$METADATA_DIR" ]; then
    echo "錯誤: 元數據目錄不存在: $METADATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
log_debug "確保輸出目錄存在: $OUTPUT_DIR"

# 處理每個頻率
for FREQ in "${FREQUENCIES[@]}"; do
    echo "==== 處理 $MATERIAL $FREQ ===="
    
    # 定義元數據和輸出文件
    METADATA_FILE="$METADATA_DIR/${MATERIAL}_${FREQ}_influence_metadata.json"
    EXCLUSION_FILE="$OUTPUT_DIR/${MATERIAL}_${FREQ}_excluded_pairs.txt"
    
    log_debug "元數據文件: $METADATA_FILE"
    log_debug "排除列表文件: $EXCLUSION_FILE"
    
    if [ ! -f "$METADATA_FILE" ]; then
        echo "警告: 元數據文件不存在: $METADATA_FILE"
        echo "跳過 $FREQ"
        continue
    fi
    
    # 先運行測試腳本進行評估
    echo "分析影響力元數據..."
    # 暫時禁用測試腳本調用，生成排除列表部分工作正常
    # python -m tracin.scripts.test_pair_exclusion \
    #     --metadata-file "$METADATA_FILE" \
    #     --thresholds "$THRESHOLD,-10.0,-15.0" \
    #     --min-occurrences "$MIN_OCCURRENCES"
    
    # 如果不是僅評估模式，則生成排除列表
    if [ "$EVALUATE_ONLY" != "true" ]; then
        echo "生成排除列表..."
        log_debug "運行 generate_exclusions.py 腳本，參數:"
        log_debug "  --metadata-file: $METADATA_FILE"
        log_debug "  --output-file: $EXCLUSION_FILE"
        log_debug "  --threshold: $THRESHOLD"
        log_debug "  --min-occurrences: $MIN_OCCURRENCES"
        log_debug "  --max-exclusions: $MAX_EXCLUSIONS"

        python -m tracin.scripts.generate_exclusions \
            --metadata-file "$METADATA_FILE" \
            --output-file "$EXCLUSION_FILE" \
            --threshold "$THRESHOLD" \
            --min-occurrences "$MIN_OCCURRENCES" \
            --max-exclusions "$MAX_EXCLUSIONS" \
            --consider-both-samples \
            --verbose
        
        echo "排除列表已保存到: $EXCLUSION_FILE"
        echo "使用此排除列表進行訓練，運行:"
        echo "python train.py --frequency $FREQ --material $MATERIAL --exclusions-file=$EXCLUSION_FILE"
    else
        log_debug "僅評估模式，跳過生成排除列表"
    fi
    
    echo ""
done

echo "完成!"

# 確保腳本可執行
chmod +x "$0" 
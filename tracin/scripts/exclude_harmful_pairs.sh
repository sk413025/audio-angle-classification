#!/bin/bash
# 排除有害 ranking pair 的工作流程腳本

# 默認值
METADATA_DIR="/Users/sbplab/Hank/sinetone_sliced/step_018_sliced/metadata"
FREQUENCIES=("500hz" "1000hz" "3000hz")
MATERIAL="plastic"
THRESHOLD="-5.0"
MIN_OCCURRENCES=3
MAX_EXCLUSIONS=50
PAIR_MODE="ranking_pair"
OUTPUT_DIR=""

# 顯示幫助信息
function show_help {
    echo "用法: $0 [選項]"
    echo ""
    echo "分析 TracIn 影響力分數並從訓練中排除有害的 ranking pair"
    echo ""
    echo "選項:"
    echo "  -m, --metadata-dir DIR      包含 TracIn 元數據文件的目錄 (默認: $METADATA_DIR)"
    echo "  -o, --output-dir DIR        保存排除列表的目錄 (如未指定則使用元數據目錄)"
    echo "  -f, --frequencies FREQS     要處理的頻率列表，用逗號分隔 (默認: 500hz,1000hz,3000hz)"
    echo "  -a, --material STR          材料類型 (默認: $MATERIAL)"
    echo "  -t, --threshold NUM         負面影響閾值 (默認: $THRESHOLD)"
    echo "  -n, --min-occurrences NUM   最小負面影響出現次數 (默認: $MIN_OCCURRENCES)"
    echo "  -x, --max-exclusions NUM    最大排除樣本數量 (默認: $MAX_EXCLUSIONS)"
    echo "  -p, --pair-mode MODE        排除模式：full_pair 或 ranking_pair (默認: $PAIR_MODE)"
    echo "  -e, --evaluate-only         僅進行評估，不生成排除列表"
    echo "  -h, --help                  顯示此幫助信息"
    echo ""
    echo "使用示例:"
    echo "  $0 --frequencies 500hz --threshold -10.0"
    echo "  $0 --metadata-dir /path/to/metadata --output-dir /path/to/output"
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

# 如果未指定輸出目錄，使用元數據目錄
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$METADATA_DIR"
fi

# 確保目錄存在
if [ ! -d "$METADATA_DIR" ]; then
    echo "錯誤: 元數據目錄不存在: $METADATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 處理每個頻率
for FREQ in "${FREQUENCIES[@]}"; do
    echo "==== 處理 $MATERIAL $FREQ ===="
    
    # 定義元數據和輸出文件
    METADATA_FILE="$METADATA_DIR/${MATERIAL}_${FREQ}_influence_metadata.json"
    EXCLUSION_FILE="$OUTPUT_DIR/${MATERIAL}_${FREQ}_excluded_pairs.txt"
    
    if [ ! -f "$METADATA_FILE" ]; then
        echo "警告: 元數據文件不存在: $METADATA_FILE"
        echo "跳過 $FREQ"
        continue
    fi
    
    # 先運行測試腳本進行評估
    echo "分析影響力元數據..."
    python -m tracin.scripts.test_pair_exclusion \
        --metadata-file "$METADATA_FILE" \
        --thresholds "$THRESHOLD,-10.0,-15.0" \
        --min-occurrences "$MIN_OCCURRENCES"
    
    # 如果不是僅評估模式，則生成排除列表
    if [ "$EVALUATE_ONLY" != "true" ]; then
        echo "生成排除列表..."
        python -m tracin.scripts.generate_exclusions \
            --metadata-file "$METADATA_FILE" \
            --output-file "$EXCLUSION_FILE" \
            --threshold "$THRESHOLD" \
            --min-occurrences "$MIN_OCCURRENCES" \
            --max-exclusions "$MAX_EXCLUSIONS" \
            --pair-mode "$PAIR_MODE" \
            --verbose
        
        echo "排除列表已保存到: $EXCLUSION_FILE"
        echo "使用此排除列表進行訓練，運行:"
        echo "python train.py --frequency $FREQ --material $MATERIAL --exclusions-file=$EXCLUSION_FILE"
    fi
    
    echo ""
done

echo "完成!"

# 確保腳本可執行
chmod +x "$0" 
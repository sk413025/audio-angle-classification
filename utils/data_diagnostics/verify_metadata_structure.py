import json
import sys
from collections import defaultdict

# 加載元數據文件
metadata_file = '/Users/sbplab/Hank/audio-angle-classification/metadata/plastic_500hz_influence_metadata.json'
with open(metadata_file, 'r') as f:
    data = json.load(f)

print(f"元數據文件總共包含 {len(data)} 個項目")

# 檢查構造
first_key = list(data.keys())[0]
first_value = data[first_key]

print(f"第一個 KEY: {first_key}")
print(f"第一個項目的值包含 {len(first_value)} 個鍵值對")

# 分析前綴
prefixes = defaultdict(int)
key_types = defaultdict(int)

for key, value in first_value.items():
    # 獲取前綴（如果有的話）
    if '_' in key:
        prefix = key.split('_')[0]
        prefixes[prefix] += 1
    
    # 檢查鍵的類型
    if key.startswith('tracin_influence_'):
        key_types['tracin_influence'] += 1
    else:
        key_types['other'] += 1

print(f"\n前綴統計: {dict(prefixes)}")
print(f"鍵類型統計: {dict(key_types)}")

# 如果這是訓練樣本對的字典，則檢查是否有異常影響
threshold_low = -20.0
threshold_high = 20.0
score_prefix = "tracin_influence_"

# 計算異常影響分數
anomalous_influence_counts = defaultdict(int)
all_score_counts = defaultdict(int)

# 檢查每個訓練樣本對
for training_id, influence_records in data.items():
    # 異常影響的測試樣本數量
    anomalous_count = 0
    
    for key, score in influence_records.items():
        if not key.startswith(score_prefix):
            continue
        
        all_score_counts[training_id] += 1
        
        if score < threshold_low or score > threshold_high:
            anomalous_count += 1
            
    if anomalous_count > 0:
        anomalous_influence_counts[training_id] = anomalous_count

# 輸出異常影響統計
print(f"\n找到 {len(anomalous_influence_counts)} 個訓練樣本對對至少1個測試樣本有異常影響")
print(f"至少3個異常影響的訓練樣本對數量: {sum(1 for count in anomalous_influence_counts.values() if count >= 3)}")

# 顯示前幾個有最多異常影響的訓練樣本對
top_anomalous = sorted(anomalous_influence_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n影響最異常的訓練樣本對:")
for training_id, count in top_anomalous:
    print(f"  {training_id}: 影響 {count}/{all_score_counts[training_id]} 個測試樣本")
    
    # 顯示一些異常影響的示例
    examples = []
    for key, score in data[training_id].items():
        if key.startswith(score_prefix) and (score < threshold_low or score > threshold_high):
            test_id = key[len(score_prefix):]
            examples.append((test_id, score))
            if len(examples) >= 3:
                break
                
    for test_id, score in examples:
        print(f"    - 測試樣本 {test_id}: {score}")

# 檢查元數據中所有條目的類型（以確保一致性）
structure_consistent = True
for i, (key, value) in enumerate(list(data.items())[:10]):
    if not isinstance(value, dict):
        print(f"警告: 索引 {i}, 鍵 {key} 的值不是字典，而是 {type(value)}")
        structure_consistent = False
        
    for sub_key, sub_value in list(value.items())[:2]:
        if not sub_key.startswith('tracin_influence_') and sub_key != 'metadata':
            print(f"警告: 索引 {i}, 鍵 {key}, 子鍵 {sub_key} 不以 'tracin_influence_' 開頭")
            structure_consistent = False
        
        if not isinstance(sub_value, (int, float)) and sub_key != 'metadata':
            print(f"警告: 索引 {i}, 鍵 {key}, 子鍵 {sub_key} 的值不是數字，而是 {type(sub_value)}")
            structure_consistent = False

if structure_consistent:
    print("\n元數據結構驗證通過: 一致性良好")
else:
    print("\n元數據結構驗證失敗: 發現不一致")
    
# 最後一次檢查，確認我們對數據結構的理解
print("\n數據結構驗證:")
if 'tracin_influence_' in str(list(data.values())[0]):
    print("√ 元數據中包含 'tracin_influence_' 前綴的鍵")
    prefix_structure = "訓練樣本ID -> 影響力記錄(包含測試樣本ID)"
else:
    print("× 元數據中不包含 'tracin_influence_' 前綴的鍵")
    prefix_structure = "未知結構"

print(f"根據分析，元數據結構很可能是: {prefix_structure}") 
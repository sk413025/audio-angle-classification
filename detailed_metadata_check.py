import json
import sys

# 加載元數據文件
metadata_file = '/Users/sbplab/Hank/audio-angle-classification/metadata/plastic_500hz_influence_metadata.json'
with open(metadata_file, 'r') as f:
    data = json.load(f)

print(f"元數據文件總共包含 {len(data)} 個項目")

# 排除列表中的樣本
excluded_samples = [
    'plastic_deg180_500hz_06_plastic_deg108_500hz_01',
    'plastic_deg000_500hz_04_plastic_deg072_500hz_08'
]

# 檢查是否直接作為主鍵出現
for excluded in excluded_samples:
    is_direct_key = excluded in data
    print(f"樣本 '{excluded}' 是否作為元數據中的主鍵(測試樣本對): {is_direct_key}")

# 檢查是否作為值出現，不帶tracin_influence前綴
print("\n檢查是否作為訓練樣本影響出現:")
for excluded in excluded_samples:
    found_as_value = []
    for test_id, test_info in list(data.items())[:10]:  # 只檢查前10個測試樣本
        for key, value in test_info.items():
            if key == f"tracin_influence_{excluded}":
                found_as_value.append((test_id, value))
    
    if found_as_value:
        print(f"樣本 '{excluded}' 作為訓練樣本出現在影響力記錄中")
        print(f"  在前10個測試樣本中，發現 {len(found_as_value)} 個記錄")
        for test_id, value in found_as_value[:3]:
            print(f"  - 測試樣本 '{test_id}' 的影響力: {value}")
    else:
        print(f"樣本 '{excluded}' 未作為訓練樣本出現在前10個測試樣本的影響力記錄中")

# 直接檢查特定樣本
specific_test = "plastic_deg072_500hz_06_plastic_deg108_500hz_03"
if specific_test in data:
    print(f"\n檢查特定測試樣本: {specific_test}")
    test_info = data[specific_test]
    influence_records = [(k, v) for k, v in test_info.items() if k.startswith("tracin_influence_")]
    print(f"該測試樣本有 {len(influence_records)} 條影響力記錄")
    for key, value in influence_records:
        if any(excluded in key for excluded in excluded_samples):
            print(f"  {key}: {value}")

# 檢查特定樣本的結構
plastic_deg000 = "plastic_deg000_500hz_04_plastic_deg072_500hz_08"
search_result = []
for test_id, test_info in data.items():
    if plastic_deg000 in test_id:
        search_result.append(("test_id", test_id))
        
    for key, value in test_info.items():
        if plastic_deg000 in key:
            search_result.append(("influence_key", test_id, key, value))

print(f"\n檢索 '{plastic_deg000}' 的結果:")
for result in search_result:
    if result[0] == "test_id":
        print(f"作為測試樣本ID: {result[1]}")
    else:
        print(f"作為影響力記錄: 測試樣本={result[1]}, 鍵={result[2]}, 值={result[3]}")
        
# 最後確認
count_as_test = 0
count_as_train = 0

for key in data.keys():
    if plastic_deg000 == key:
        count_as_test += 1
        
for test_id, test_info in data.items():
    influence_key = f"tracin_influence_{plastic_deg000}"
    if influence_key in test_info:
        count_as_train += 1
        
print(f"\n最終確認 '{plastic_deg000}':")
print(f"  作為測試樣本出現次數: {count_as_test}")
print(f"  作為訓練樣本出現次數: {count_as_train}") 
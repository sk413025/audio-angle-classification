import json
import sys

# 加載元數據文件
metadata_file = '/Users/sbplab/Hank/audio-angle-classification/metadata/plastic_500hz_influence_metadata.json'
with open(metadata_file, 'r') as f:
    data = json.load(f)

# 被排除的樣本對
excluded_samples = [
    'plastic_deg180_500hz_06_plastic_deg108_500hz_01',
    'plastic_deg000_500hz_04_plastic_deg072_500hz_08'
]

print(f"元數據文件中有 {len(data)} 個測試樣本對")

# 檢查是否為測試樣本
for i, excluded in enumerate(excluded_samples):
    is_test_sample = excluded in data
    print(f"排除樣本 {i+1}: {excluded}")
    print(f"  是測試樣本對: {is_test_sample}")
    
    # 尋找作為訓練樣本的引用
    found_as_train = []
    for test_sample, test_data in data.items():
        influence_key = f'tracin_influence_{excluded}'
        if influence_key in test_data:
            found_as_train.append((test_sample, test_data[influence_key]))
    
    print(f"  作為訓練樣本影響了 {len(found_as_train)} 個測試樣本")
    if found_as_train:
        print("  前 5 個受影響的測試樣本:")
        for j, (test_sample, score) in enumerate(found_as_train[:5]):
            print(f"    {j+1}. {test_sample}: {score}")
            
    print()
    
# 檢查前幾個測試樣本的內容
print("前 3 個測試樣本的詳細信息:")
for i, (test_sample, test_data) in enumerate(list(data.items())[:3]):
    print(f"{i+1}. 測試樣本: {test_sample}")
    influence_keys = [k for k in test_data.keys() if k.startswith('tracin_influence_')]
    print(f"   有 {len(influence_keys)} 個訓練樣本的影響力記錄")
    if influence_keys:
        print("   前 3 個訓練樣本的影響力:")
        for j, key in enumerate(influence_keys[:3]):
            training_id = key.replace('tracin_influence_', '')
            score = test_data[key]
            print(f"     {j+1}. {training_id}: {score}")
    print() 
"""
簡單CNN主執行腳本
功能：
- 提供命令行界面，執行訓練和評估操作
- 支援不同頻率和材質的訓練
- 支援交叉驗證訓練
- 提供結果可視化選項
"""

import os
import argparse
from datetime import datetime
import torch

import config
from simple_cnn_train import train_cnn_model
from simple_cnn_evaluate import evaluate_cnn_model
from datasets import SpectrogramDatasetWithMaterial

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='CNN音頻排序模型訓練與評估')
    
    # 操作模式
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'train_all'],
                        help='執行模式：train=訓練單個模型, evaluate=評估模型, train_all=訓練所有頻率')
    
    # 資料參數
    parser.add_argument('--freq', type=str, default='3000hz',
                        choices=['500hz', '1000hz', '3000hz'],
                        help='使用的頻率數據')
    parser.add_argument('--material', type=str, default='plastic',
                        choices=config.MATERIALS,
                        help='使用的材質')
    parser.add_argument('--train_seqs', type=str, default=None,
                        help='訓練序列，例如："00,01,02,03,04,05"')
    parser.add_argument('--test_seqs', type=str, default=None,
                        help='測試序列，例如："06,07,08"')
                        
    # 模型參數
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路徑（用於評估模式）')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='訓練輪數')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='學習率')
    
    # 其他選項
    parser.add_argument('--no_plot', action='store_true',
                        help='禁用繪圖')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='覆寫模型保存目錄')
    
    args = parser.parse_args()
    
    # 處理序列參數
    if args.train_seqs:
        args.train_seqs = args.train_seqs.split(',')
    else:
        args.train_seqs = config.SEQ_NUMS[:-2]  # 默認使用前面的序列作為訓練集
        
    if args.test_seqs:
        args.test_seqs = args.test_seqs.split(',')
    else:
        args.test_seqs = config.SEQ_NUMS[-2:]   # 默認使用最後兩個序列作為測試集
    
    # 更新配置
    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size
        print(f"已更新批次大小為 {args.batch_size}")
        
    if args.epochs != config.NUM_EPOCHS:
        config.NUM_EPOCHS = args.epochs
        print(f"已更新訓練輪數為 {args.epochs}")
        
    if args.lr != config.LEARNING_RATE:
        config.LEARNING_RATE = args.lr
        print(f"已更新學習率為 {args.lr}")
        
    if args.save_dir:
        config.SAVE_DIR = args.save_dir
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        print(f"已更新模型保存目錄為 {args.save_dir}")
        
    return args

def main():
    """主函數"""
    args = parse_args()
    
    # 設置環境變量，防止Matplotlib錯誤
    os.environ['MPLBACKEND'] = 'Agg'  # 添加此行
    
    # 打印系統信息
    config.print_system_info()
    print(f"執行模式: {args.mode}")
    
    # 根據模式執行操作
    if args.mode == 'train':
        # 訓練單個模型
        print(f"訓練模型：頻率 {args.freq}, 材質 {args.material}")
        print(f"使用訓練序列: {args.train_seqs}")
        
        model = train_cnn_model(args.freq, args.material, args.train_seqs)
        
        # 訓練後立即評估
        print("\n訓練完成，開始評估...")
        
        # 找到最新訓練的模型
        model_dir = config.SAVE_DIR
        model_files = [f for f in os.listdir(model_dir) 
                       if f.startswith(f"simple_cnn_{args.material}_{args.freq}") 
                       and f.endswith(".pt")]
        
        if not model_files:
            print("找不到訓練好的模型文件，無法評估")
            return
            
        # 按修改時間排序，選擇最新的模型
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        latest_model = os.path.join(model_dir, model_files[0])
        
        # 評估模型
        print(f"使用模型 {latest_model} 進行評估")
        evaluate_cnn_model(
            latest_model, 
            args.freq, 
            args.material, 
            args.test_seqs,
            not args.no_plot
        )
    
    elif args.mode == 'evaluate':
        # 評估模式
        if not args.model_path:
            print("錯誤：評估模式需要提供模型路徑 (--model_path)")
            return
            
        if not os.path.exists(args.model_path):
            print(f"錯誤：找不到模型文件 {args.model_path}")
            return
            
        print(f"評估模型：{args.model_path}")
        print(f"使用測試序列: {args.test_seqs}")
        
        evaluate_cnn_model(
            args.model_path, 
            args.freq, 
            args.material, 
            args.test_seqs,
            not args.no_plot
        )
    
    elif args.mode == 'train_all':
        # 訓練所有頻率的模型
        print(f"訓練所有頻率模型，材質: {args.material}")
        print(f"使用訓練序列: {args.train_seqs}")
        
        for freq in config.FREQUENCIES:
            print(f"\n{'='*50}")
            print(f"開始訓練頻率 {freq} 的模型")
            print(f"{'='*50}")
            
            model = train_cnn_model(freq, args.material, args.train_seqs)
            
            # 訓練後評估
            print("\n評估當前頻率模型...")
            
            # 找到最新訓練的模型
            model_dir = config.SAVE_DIR
            model_files = [f for f in os.listdir(model_dir) 
                           if f.startswith(f"simple_cnn_{args.material}_{freq}") 
                           and f.endswith(".pt")]
            
            if not model_files:
                print(f"找不到頻率 {freq} 的模型文件，跳過評估")
                continue
                
            # 按修改時間排序，選擇最新的模型
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            latest_model = os.path.join(model_dir, model_files[0])
            
            # 評估模型
            print(f"使用模型 {latest_model} 進行評估")
            evaluate_cnn_model(
                latest_model, 
                freq, 
                args.material, 
                args.test_seqs,
                not args.no_plot
            )

def verify_data_availability():
    """檢查數據是否可用"""
    print("檢查數據集...")
    
    for freq in config.FREQUENCIES:
        for material in config.MATERIALS:
            dataset = SpectrogramDatasetWithMaterial(
                config.DATA_ROOT,
                config.CLASSES,
                config.SEQ_NUMS,
                freq,
                material
            )
            
            if len(dataset) > 0:
                print(f"頻率 {freq}, 材質 {material}: {len(dataset)} 個樣本可用")
            else:
                print(f"警告: 頻率 {freq}, 材質 {material} 無可用數據")

if __name__ == "__main__":
    print("="*50)
    print("簡單CNN音頻排序模型 - 執行腳本")
    print("="*50)
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 檢查數據可用性
    verify_data_availability()
    
    # 執行主函數
    main()

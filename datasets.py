"""
數據集定義：處理音頻及頻譜數據
功能：
- 讀取並處理音頻文件
- 將音頻轉換為頻譜圖（Spectrogram）
- 支持不同材質的音頻數據加載
- 實現訓練數據的排序對（用於排序學習）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import SAMPLE_RATE, MATERIAL, SEQ_NUMS  # 假設這些參數在 config.py 中定義

# STFT 參數
N_FFT = 4096
HOP_LENGTH = 64
WIN_LENGTH = 1024
WINDOW = 'hann'

class SpectrogramDatasetWithMaterial(Dataset):
    """
    頻譜數據集，支持材質子目錄
    文件結構: {data_dir}/{class}/{material}/{material}_{class}_{selected_freq}_{seq_num}.wav
    """
    def __init__(self, data_dir, classes, selected_seqs, selected_freq, material):
        self.data_dir = data_dir
        self.classes = classes
        self.selected_seqs = selected_seqs
        self.selected_freq = selected_freq
        self.material = material
        
        # 存儲數據和標籤
        self.data = []
        self.labels = []
        self.paths = []
        self.freqs = None
        self.load_dataset()
        
    def load_dataset(self):
        """加載數據集文件路徑和數據"""
        print(f"Loading Dataset for {self.selected_freq} with material {self.material}")
        
        # 抑制警告
        import warnings
        warnings.filterwarnings("ignore")
        import os
        os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
        
        for i, class_name in enumerate(self.classes):
            print(f"\nClass {class_name} (label {i}):")
            class_dir = os.path.join(self.data_dir, class_name, self.material)
            files_found = 0
            
            if not os.path.exists(class_dir):
                print(f"Directory does not exist: {class_dir}")
                continue
                
            # 尋找符合條件的文件
            for seq in self.selected_seqs:
                filename = f"{self.material}_{class_name}_{self.selected_freq}_{seq}.wav"
                file_path = os.path.join(class_dir, filename)
                
                if os.path.exists(file_path):
                    try:
                        # 讀取 WAV 文件並處理為頻譜圖
                        import wave
                        
                        with wave.open(file_path, 'rb') as wav_file:
                            n_channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                            framerate = wav_file.getframerate()
                            n_frames = wav_file.getnframes()
                            raw_data = wav_file.readframes(n_frames)
                            
                            # 轉換為 numpy 數組
                            dtype = np.int16 if sample_width == 2 else np.int8
                            audio = np.frombuffer(raw_data, dtype=dtype)
                            
                            # 如果是立體聲，轉為單聲道
                            if n_channels == 2:
                                audio = audio.reshape(-1, 2).mean(axis=1)
                            
                            # 將整數轉換為 float，範圍為 [-1, 1]
                            audio = audio.astype(np.float32) / (2**(8*sample_width - 1))
                            
                        # 計算 STFT
                        def custom_stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
                            win_length = n_fft
                            y_pad = np.pad(y, int(n_fft // 2), mode='reflect')
                            
                            n_frames = 1 + (len(y_pad) - win_length) // hop_length
                            stft_matrix = np.zeros((1 + n_fft // 2, n_frames), dtype=complex)
                            
                            window = np.hamming(win_length)
                            
                            for i in range(n_frames):
                                start = i * hop_length
                                segment = y_pad[start:start+win_length] * window
                                spectrum = np.fft.rfft(segment, n=n_fft)
                                stft_matrix[:, i] = spectrum
                            
                            return stft_matrix
                        
                        stft_matrix = custom_stft(audio)
                        
                        # 計算幅度和轉換為分貝刻度
                        magnitude = np.abs(stft_matrix)
                        ref = np.max(magnitude)
                        amin = 1e-10  # 避免 log(0)
                        spect_db = 20.0 * np.log10(np.maximum(magnitude, amin) / max(amin, ref))
                        
                        # 添加通道維度
                        spect_db = np.expand_dims(spect_db, axis=0)
                        
                        # 保存頻率資訊
                        freqs = np.fft.rfftfreq(N_FFT, 1.0/framerate)
                        if self.freqs is None:
                            self.freqs = freqs
                        
                        # 保存結果
                        self.data.append(spect_db)
                        self.labels.append(i)
                        self.paths.append(file_path)
                        files_found += 1
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
            
            print(f"Found {files_found} files for class {class_name}")
        
        # 轉換為張量
        if len(self.data) > 0:
            self.data = torch.FloatTensor(np.stack(self.data))
            self.labels = torch.LongTensor(self.labels)
            print(f"\nSpectrograms shape: {self.data.shape}")
            if self.freqs is not None:
                print(f"Frequency range: {min(self.freqs):.1f}-{max(self.freqs):.1f} Hz")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class RankingPairDataset(Dataset):
    """
    排序對數據集：生成符合 MarginRankingLoss 期望格式的數據
    返回: (x1, x2, target) 其中:
    - target=1 表示 x1 應排在 x2 前面 (x1 類別數值 > x2 類別數值)
    - target=-1 表示 x2 應排在 x1 前面 (x1 類別數值 < x2 類別數值)
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.pairs = self._create_ranking_pairs()
        print(f"Created {len(self.pairs)} ranking pairs")
        
    def _create_ranking_pairs(self):
        pairs = []
        # 獲取所有樣本及標籤
        all_data = []
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            label = label.item() if isinstance(label, torch.Tensor) else label
            all_data.append((idx, label))
        
        # 確定生成多少對
        n_samples = len(all_data)
        # 可以調整生成對的數量，這裡設置為原來可能對數的一半
        n_pairs = (n_samples * (n_samples - 1)) 
        
        # 隨機生成樣本對
        for _ in range(n_pairs):
            # 隨機選擇兩個不同的樣本
            sample1, sample2 = np.random.choice(len(all_data), 2, replace=False)
            idx1, label1 = all_data[sample1]
            idx2, label2 = all_data[sample2]
            
            # 根據類別順序確定目標值
            if label1 > label2:
                # x1 應排在 x2 前面
                pairs.append((idx1, idx2, 1))
            elif label1 < label2:
                # x2 應排在 x1 前面
                pairs.append((idx1, idx2, -1))
            # 如果標籤相同，則不添加此對（繼續循環）
        
        return pairs
    
    def __getitem__(self, idx):
        idx1, idx2, target = self.pairs[idx]
        
        data1, label1 = self.dataset[idx1]
        data2, label2 = self.dataset[idx2]
        
        # 確保 target 是標量浮點數張量
        target_tensor = torch.tensor(target, dtype=torch.float)
        
        return data1, data2, target_tensor, label1, label2
    
    def __len__(self):
        return len(self.pairs)

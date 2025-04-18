"""
Audio Dataset Classes

This module provides classes for loading and processing audio data,
with a focus on spectrograms for angle classification.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import wave
from datasets.base import ManagedDataset
from datasets.metadata import SampleMetadata
import config

# STFT 常數
N_FFT = 2048
HOP_LENGTH = 256

class AudioSpectrumDataset(ManagedDataset):
    """
    Audio spectrogram dataset with sample management features.
    
    Extends ManagedDataset with audio-specific functionality.
    """
    
    def __init__(self, 
                 data_root: str, 
                 classes: List[str], 
                 selected_seqs: List[int],
                 selected_freq: str,
                 material: str,
                 exclusion_file: Optional[str] = None,
                 metadata_file: Optional[str] = None):
        """
        Initialize audio spectrogram dataset.
        
        Args:
            data_root: Root directory containing data files
            classes: List of class names/angles
            selected_seqs: List of sequence numbers to include
            selected_freq: Frequency to use (e.g., '500hz', '1000hz')
            material: Material type (e.g., 'plastic', 'metal')
            exclusion_file: Path to file listing excluded sample IDs (optional)
            metadata_file: Path to file with sample metadata (optional)
        """
        super().__init__(data_root, exclusion_file, metadata_file)
        
        self.classes = classes
        self.selected_seqs = selected_seqs
        self.selected_freq = selected_freq
        self.material = material
        
        # 存儲數據和標籤
        self.data = []  # 存儲處理後的頻譜數據
        self.labels = []  # 存儲標籤（角度類別）
        self.paths = []  # 存儲文件路徑
        self.sample_list = []  # 存儲樣本信息，用於生成唯一ID
        self.freqs = None  # 存儲頻率信息
        
        # 加載數據集
        self.load_dataset()
        
    def load_dataset(self) -> None:
        """Load dataset files and process into spectrograms"""
        print(f"Loading Dataset for {self.selected_freq} with material {self.material}")
        
        # 抑制警告
        import warnings
        warnings.filterwarnings("ignore")
        import os
        os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
        
        for i, class_name in enumerate(self.classes):
            print(f"\nClass {class_name} (label {i}):")
            class_dir = os.path.join(self.data_root, class_name, self.material)
            files_found = 0
            
            if not os.path.exists(class_dir):
                print(f"Directory does not exist: {class_dir}")
                continue
                
            # 尋找符合條件的文件
            for seq in self.selected_seqs:
                filename = f"{self.material}_{class_name}_{self.selected_freq}_{seq}.wav"
                file_path = os.path.join(class_dir, filename)
                
                # 創建樣本信息，用於生成唯一ID
                sample_info = {
                    "class": class_name,
                    "material": self.material,
                    "frequency": self.selected_freq,
                    "seq_num": seq,
                    "path": file_path
                }
                
                # 生成樣本ID
                sample_id = self.get_sample_id(sample_info)
                
                # 檢查是否排除此樣本
                if sample_id in self.excluded_samples:
                    print(f"Skipping excluded sample: {filename}")
                    continue
                
                if os.path.exists(file_path):
                    try:
                        # 讀取 WAV 文件並處理為頻譜圖
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
                        stft_matrix = self._compute_stft(audio)
                        
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
                        
                        # 添加到樣本列表
                        self.samples.append(sample_info)
                        
                        # 保存結果
                        self.data.append(spect_db)
                        self.labels.append(i)
                        self.paths.append(file_path)
                        
                        # 保存或更新元數據
                        # 從類名中提取角度值（例如從"deg000"中提取0）
                        if class_name.startswith("deg"):
                            angle_str = class_name[3:]  # 提取"deg"之後的部分
                            angle = float(angle_str)
                        else:
                            # 嘗試直接解析，如果失敗則使用標籤索引
                            try:
                                angle = float(class_name)
                            except ValueError:
                                angle = float(i)  # 使用類別索引作為後備
                        
                        metadata = self.get_sample_metadata(sample_id)
                        if not metadata:
                            metadata = {
                                "id": sample_id,
                                "file_path": file_path,
                                "angle": angle,
                                "material": self.material,
                                "frequency": self.selected_freq,
                                "seq_num": seq,
                                "excluded": False,
                                "notes": "",
                                "ghm_bins": {}
                            }
                            self.set_sample_metadata(sample_id, metadata)
                        
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
    
    def _compute_stft(self, y: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            y: Audio signal data
            
        Returns:
            STFT matrix
        """
        win_length = N_FFT
        y_pad = np.pad(y, int(N_FFT // 2), mode='reflect')
        
        n_frames = 1 + (len(y_pad) - win_length) // HOP_LENGTH
        stft_matrix = np.zeros((1 + N_FFT // 2, n_frames), dtype=complex)
        
        window = np.hamming(win_length)
        
        for i in range(n_frames):
            start = i * HOP_LENGTH
            segment = y_pad[start:start+win_length] * window
            spectrum = np.fft.rfft(segment, n=N_FFT)
            stft_matrix[:, i] = spectrum
        
        return stft_matrix
    
    def get_sample_id(self, sample: Dict) -> str:
        """
        Generate a unique ID for a sample.
        
        Args:
            sample: Sample information dictionary
            
        Returns:
            Unique string identifier for the sample
        """
        if isinstance(sample, dict):
            return f"{sample['material']}_{sample['class']}_{sample['frequency']}_{sample['seq_num']}"
        else:
            # Fallback for legacy code
            idx = self.samples.index(sample)
            path = self.paths[idx]
            basename = os.path.basename(path)
            return basename.replace(".wav", "")
    
    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by its index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, label)
        """
        return self.data[idx], self.labels[idx]
    
    def get_path(self, idx: int) -> str:
        """
        Get the file path for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Path to the audio file
        """
        return self.paths[idx]
    
    def get_frequency_axis(self) -> np.ndarray:
        """
        Get the frequency axis for spectrograms.
        
        Returns:
            Array of frequency values
        """
        return self.freqs
    
    def record_ghm_bin(self, sample_id: str, epoch: int, bin_idx: int) -> None:
        """
        Record GHM bin for a sample in a specific epoch.
        
        Args:
            sample_id: Sample identifier
            epoch: Training epoch
            bin_idx: GHM bin index
        """
        metadata = self.get_sample_metadata(sample_id)
        if metadata:
            if 'ghm_bins' not in metadata:
                metadata['ghm_bins'] = {}
            metadata['ghm_bins'][str(epoch)] = bin_idx
            self.set_sample_metadata(sample_id, metadata)


# 向後兼容類 (與原始數據集實現兼容)
class SpectrogramDatasetWithMaterial(AudioSpectrumDataset):
    """
    Legacy compatibility class that matches the original interface.
    
    This allows existing code to continue working with minimal changes.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 classes: List[str], 
                 selected_seqs: List[int],
                 selected_freq: str,
                 material: str):
        """
        Initialize with legacy interface.
        
        Args:
            data_dir: Root directory containing data files
            classes: List of class names/angles
            selected_seqs: List of sequence numbers to include
            selected_freq: Frequency to use (e.g., '500hz', '1000hz')
            material: Material type (e.g., 'plastic', 'metal')
        """
        # Call parent constructor, but don't load exclusion or metadata files
        super().__init__(
            data_root=data_dir,
            classes=classes,
            selected_seqs=selected_seqs,
            selected_freq=selected_freq,
            material=material,
            # Skip exclusion and metadata for backward compatibility
            exclusion_file=None,
            metadata_file=None
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy get item implementation that doesn't include sample ID.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, label)
        """
        # Get the real index (accounting for exclusions)
        valid_indices = self.get_valid_indices()
        if idx >= len(valid_indices):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(valid_indices)} valid samples")
        
        real_idx = valid_indices[idx]
        return self.data[real_idx], self.labels[real_idx] 
import numpy as np
import wfdb
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pickle
import os

class ECGPreprocessor:
    """
    Preprocessor for MIT-BIH ECG data
    - Filters noise
    - Resamples to target frequency  
    - Segments around R-peaks
    - Normalizes
    """
    
    def __init__(self, target_fs=180, window_size=180):
        self.target_fs = target_fs
        self.window_size = window_size
        self.scaler = StandardScaler()
        
    def bandpass_filter(self, ecg_signal, fs):
        """Apply bandpass filter (0.5-40 Hz) to remove noise"""
        nyquist = fs / 2.0
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        return filtered
    
    def resample_signal(self, ecg_signal, original_fs):
        """Resample to target frequency"""
        if original_fs == self.target_fs:
            return ecg_signal
        
        num_samples = int(len(ecg_signal) * self.target_fs / original_fs)
        resampled = signal.resample(ecg_signal, num_samples)
        return resampled
    
    def segment_signal(self, ecg_signal, annotations):
        """
        Create fixed-size windows centered on R-peaks
        
        Returns:
            segments: (N, window_size) array
            labels: (N,) array - 0=Normal, 1=Arrhythmia
        """
        segments = []
        labels = []
        
        for ann in annotations:
            r_peak = ann.sample
            label = ann.symbol
            
            # Extract window centered on R-peak
            start = r_peak - self.window_size // 2
            end = start + self.window_size
            
            # Check bounds
            if start >= 0 and end < len(ecg_signal):
                segment = ecg_signal[start:end]
                segments.append(segment)
                
                # Binary classification: Normal vs Arrhythmia
                if label in ['N', '.']:  # Normal beats
                    labels.append(0)
                else:  # Arrhythmia
                    labels.append(1)
        
        return np.array(segments), np.array(labels)
    
    def normalize(self, segments):
        """Z-score normalization"""
        segments_flat = segments.reshape(-1, 1)
        self.scaler.fit(segments_flat)
        normalized = self.scaler.transform(segments_flat)
        return normalized.reshape(segments.shape)
    
    def process_record(self, record_name, data_dir='data/raw'):
        """Process a single ECG record"""
        record_path = os.path.join(data_dir, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        # Use lead II (channel 0)
        ecg_signal = record.p_signal[:, 0]
        fs = record.fs
        
        # Preprocessing pipeline
        ecg_filtered = self.bandpass_filter(ecg_signal, fs)
        ecg_resampled = self.resample_signal(ecg_filtered, fs)
        segments, labels = self.segment_signal(ecg_resampled, annotation)
        
        return segments, labels
    
    def process_all_records(self, record_list, save_dir='data/processed'):
        """Process entire dataset"""
        os.makedirs(save_dir, exist_ok=True)
        
        all_segments = []
        all_labels = []
        
        print(f"Processing {len(record_list)} records...")
        
        for record in record_list:
            try:
                print(f"Processing {record}...", end=' ')
                segments, labels = self.process_record(record)
                all_segments.append(segments)
                all_labels.append(labels)
                print(f"✓ ({len(segments)} segments)")
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        # Concatenate all
        all_segments = np.concatenate(all_segments, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Normalize
        all_segments = self.normalize(all_segments)
        
        # Save
        np.save(os.path.join(save_dir, 'segments.npy'), all_segments)
        np.save(os.path.join(save_dir, 'labels.npy'), all_labels)
        
        # Save scaler
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Statistics
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Total segments: {len(all_segments)}")
        print(f"Normal: {np.sum(all_labels == 0)} ({np.sum(all_labels == 0)/len(all_labels)*100:.1f}%)")
        print(f"Arrhythmia: {np.sum(all_labels == 1)} ({np.sum(all_labels == 1)/len(all_labels)*100:.1f}%)")
        print(f"Shape: {all_segments.shape}")
        print(f"Saved to: {save_dir}/")
        
        return all_segments, all_labels

if __name__ == "__main__":
    # Records to process
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', 
        '108', '109', '111', '112', '113', '114', '115', '116'
    ]
    
    preprocessor = ECGPreprocessor(target_fs=180, window_size=180)
    segments, labels = preprocessor.process_all_records(records)
    
    print("\n✅ Ready for model training!")
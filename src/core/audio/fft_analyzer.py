from scipy.ndimage import gaussian_filter1d
import numpy as np
import librosa


class FFTAnalyzer:
    def __init__(self, audio_loader, num_bands=64, fps=30):
        
        self.audio_loader = audio_loader
        self.num_bands = num_bands
        self.fps = fps
        
        self.y = audio_loader.y
        self.sr = audio_loader.sr
        self.duration = audio_loader.duration
        
        self.stft = np.abs(librosa.stft(self.y))
        
        self.frequencies = librosa.fft_frequencies(sr=self.sr)
        
        self.frequency_bands = self._compute_frequency_bands()
    
    def _compute_frequency_bands(self):
        total_num_stft_frames = self.stft.shape[1] 
        total_video_frames = int(self.duration * self.fps)
        
        freq_ranges = np.logspace(np.log10(20), np.log10(20000), self.num_bands + 1)
        
        bands = np.zeros((total_video_frames, self.num_bands))
# -----------------------------------------------------------------------------------
#                            MAPPING STFT/VIDEO FRAMES
# ----------------------------------------------------------------------------------
        for frame_idx in range(total_video_frames):
            stft_idx = int((frame_idx / total_video_frames) * total_num_stft_frames)
            stft_idx = min(stft_idx, total_num_stft_frames - 1)
            
            fft_timeframe = self.stft[:, stft_idx]
# -----------------------------------------------------------------------------------
#                              CREATING FREQUENCIES DIVISION
# ----------------------------------------------------------------------------------
            
            for band_idx in range(self.num_bands):
                freq_low = freq_ranges[band_idx]
                freq_high = freq_ranges[band_idx + 1]
                
                fft_bin = (self.frequencies >= freq_low) & (self.frequencies < freq_high)
# -----------------------------------------------------------------------------------
#                       MAXIMUM ENERGY DETECTION 
# ----------------------------------------------------------------------------------
                band_energy = np.sum(fft_timeframe[fft_bin])
                
                bands[frame_idx, band_idx] = band_energy
# -----------------------------------------------------------------------------------
#                       NORMALIZATION OF MAXIMUM ENERGY
# ----------------------------------------------------------------------------------
        
        for band_idx in range(self.num_bands):
            band_max = np.max(bands[:, band_idx])
            if band_max > 0:
                bands[:, band_idx] /= band_max
# -----------------------------------------------------------------------------------
#                               APPLYING GAUSSIAN SMOOTHING
# ----------------------------------------------------------------------------------
        
        for band_idx in range(self.num_bands):
            bands[:, band_idx] = gaussian_filter1d(bands[:, band_idx], sigma=2)
        
        return bands

# -----------------------------------------------------------------------------------
#                               AVERAGING AND RETURNING FREQUENCIES AT FRAME
# ----------------------------------------------------------------------------------
    
    def get_frequency_bands(self, frame_number):
        frame_number = min(frame_number, len(self.frequency_bands) - 1)
        return self.frequency_bands[frame_number]
    
    def get_bass(self, frame_number):
        bands = self.get_frequency_bands(frame_number)
        bass_bands = bands[:int(self.num_bands * 0.2)]
        return np.mean(bass_bands)
    
    def get_mid_frequencies(self, frame_number):
        bands = self.get_frequency_bands(frame_number)
        start = int(self.num_bands * 0.2)
        end = int(self.num_bands * 0.6)
        return np.mean(bands[start:end])
    
    def get_high_frequencies(self, frame_number):
        bands = self.get_frequency_bands(frame_number)
        treble_bands = bands[int(self.num_bands * 0.6):]
        return np.mean(treble_bands)
        





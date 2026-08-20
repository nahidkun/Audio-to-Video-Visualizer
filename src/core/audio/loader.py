import librosa 
import os
import numpy as np
import matplotlib.pyplot as plt

class AudioLoader: 

# -----------------------------------------------------------------------------------
#                                   VARIABLE INITIALIZATION
#------------------------------------------------------------------------------------
    def __init__(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.audio_path = audio_path
        self.y = None 
        self.sr = None
        self.amplitude = None
        self.s = None 
        self.to_db = None 
 
        self.tempo = None 
        self.beats_static = None
        self.click_track = None
        self.bpm = None 

        self.low_freq = None
        self.mid_freq = None
        self.high_freq = None 

        self.duration = None
        self._load_audio()

# -----------------------------------------------------------------------------------
#                      LOADING AUDIO (DURATION, TEMPO, FREQUENCIES DIVISION)
#------------------------------------------------------------------------------------
    def _load_audio(self):
        print(f"loading audio: {self.audio_path}")

        self.y, self.sr = librosa.load(self.audio_path, sr=None)

        self.duration = len(self.y) / self.sr # formula to have duration in seconds 

        self.tempo, self.beats_static = librosa.beat.beat_track(y=self.y, sr=self.sr, units='time', trim=False)
        self.click_track = librosa.clicks(times=self.beats_static, sr=self.sr, click_freq=660,
                                          click_duration=0.25, length=len(self.y)) 
        
        self.s = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=128) 
        self.to_db = librosa.power_to_db(self.s, ref=np.max)

        self.low_freq = self.to_db[0:30, :]
        self.mid_freq = self.to_db[30:80, :]
        self.high_freq = self.to_db[80:, :]


    def get_info(self):

        return {
            'path': self.audio_path,
            'duration': self.duration,
            'sample_rate': self.sr,
            'total_samples': len(self.y),
            'bpm' : round(self.tempo[0]),
            'low frequencies': self.low_freq,
            'mid frequencies': self.mid_freq,
            'high frequencies': self.high_freq
        }
    
    

    def getAmplitude(self):
        if self.amplitude is None:
            stft = np.abs(librosa.stft(self.y))
            self.amplitude = np.mean(stft, axis=0)
            self.amplitude = self.amplitude / np.max(self.amplitude)

        return self.amplitude 
    
    def amplitude_graph(self):
    # number of frequencies over time
        stft = np.abs(librosa.stft(self.y))
    # we get amplitude by getting the average of all ferquency data points
        amplitude = np.mean(stft, axis = 0)
    # make amplitude from 0 (lowest) to 1 (highest)
        amplitude = amplitude / np.max(amplitude)
        time = np.linspace(0, self.duration, len(amplitude))

        plt.figure(figsize=(12,4))
        plt.plot(time, amplitude, linewidth=0.5)
        plt.fill_between(time, amplitude, alpha=0.3)
        plt.xlabel('time (seconds)')
        plt.ylabel('amplitude')
        plt.title('amplitude graph')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

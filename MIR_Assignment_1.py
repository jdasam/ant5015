import matplotlib.pyplot as plt
import IPython.display as ipd
import torch
from math import pi, log
import torchaudio
from matplotlib.colors import LogNorm

SR = 16000


def hz_to_midi_pitch(frequency):
  '''
  입력: 음높이의 Hz 단위 주파수 (float)

  출력: 입력 음높이를 MIDI Pitch Scale로 변환한 값 (float). 반올림할 필요 없음
  '''
  # TODO: 주어진 입출력 형식과 내용을 만족하는 함수를 완성하시오
  return 

def midi_pitch_to_hz(midi_pitch):
  '''
  입력: 음높이의 MIDI Pitch Scale (float)

  출력: 입력 음높이를 Hz로 변환한 값 (float)
  '''
  # TODO: 주어진 입출력 형식과 내용을 만족하는 함수를 완성하시오
  return 

def make_sine_wave(freq, amp, dur, sr):
  num_samples = dur * sr
  time_frame = torch.arange(num_samples)
  time_frame_sec = time_frame / sr
  return amp * torch.sin(2 * pi * freq * time_frame_sec)

def generate_multi_pitch_tone(pitch_list, amp_list, duration, sr):
  '''
  입력: pitch_list: List of length n that contains n number of MIDI Pitch 
                   n 개의 MIDI Pitch 정수를 포함하고 있는 리스트 (n>=2)
       amp_list : List of length n that contains amplitude for each pitch   (0<amp<=1)
                   pitch_list에 있는 n개의 MIDI Pitch 각각에 해당하는 amplitude 
                   (각각의 amplitude는0보다 크고 1보다 작거나 같음)
      duration: 출력되는 소리의 길이 (float)
      sr: sampling rate

  출력: 입력으로 주어진 리스트 속에 있는 모든 pitch가 포함된 오디오 샘플 (sample_rate=sr, torch.Tensor 형태)

        예) 입력:  
          pitch_list = [60, 64, 67]
          amp_list = [1, 0.7, 0.5]
          출력: MIDI Pitch 60이며 amplitude가 1인 소리, Pitch 64이며 amp 0.7인 소리, 
               Pitch 67이며 amp 0.5인 소리 세 개가 duration만큼 동시에 재생되는 오디오 샘플 
    
  '''
  assert len(pitch_list) == len(amp_list)
  # TODO: 주어진 입출력 형식과 내용을 만족하는 함수를 완성하시오

  return


def make_major_scale_hz_sequence(midi_pitch):
  '''
  입력: 음높이의 MIDI Pitch Scale (float)

  출력 (list of float): 입력된 MIDI Pitch를 1음(Do)으로 하는 장음계 8개의 음의 Hz
  Hint: 장음계는 3음과 4음 (Mi, Fa) 사이가 반음, 7음과 8음 (Si, Do) 사이가 반음이며 나머지는 모두 온음 간격이다.

  예: 입력이 60 (C4)라면, C4, D4, E4, F4, G4, A4, B4, C5에 해당하는 음고 Hz를 list로 출력한다. 
  '''
  # TODO: 주어진 입출력 형식과 내용을 만족하는 함수를 완성하시오
  return


def generate_sequence_of_pitch(alist_of_hz, duration, sr):
  '''
  입력: 
    alist_of_hz (list): A list of float (Hz)
    duration (float): Duration of each frequency (seconds)

  출력: 입력으로 주어진 리스트 속에 있는 모든 Hz가 순서대로 등장하는 오디오 샘플 (sample_rate=sr, torch.Tensor 형태)

  예) 입력:  
          alist_of_hz = [440, 660, 880]
          duration = 1
          출력: 440 Hz 사인파가 1초 동안 재생된 뒤 660 Hz 사인파가 1초 동안 재생된 뒤 880 Hz 사인파가 1초 동안 재생되는 오디오 샘플
  '''
  # TODO: 주어진 입출력 형식과 내용을 만족하는 함수를 완성하시오
  return

def make_just_intonation_triad_in_midi_pitch(fundamental_pitch):
  '''
  This function takes a fundamental_pitch in MIDI pitch then returns a list of three MIDI pitch
  that corresponds to major triad for a given fundamental frequency

  fundamental_pitch: 근음의 음고 MIDI Pitch
  output= [근음의 MIDI_pitch, 근음 순정 장3도의 MIDI_Pitch, 근음기준 순정 5도의 MIDI Pitch]
  '''

  # TODO: 주어진 입출력 형식과 내용을 만족하는 함수를 완성하시오

  return



class ShepardToneGenerator:
  def __init__(self, sr, dur_per_step, num_iteration):
    self.sr = sr #sampling rate
    self.dur = dur_per_step # 한 음 당 길이 (초) 
    self.iter = num_iteration # 전체 시퀀스를 몇번 반복할 것인지
    self.fadeout_margin = self.dur*0.1 # 음 끝의 페이드아웃 길이
    
    self.pitch_combinations = self.define_pitch()
    self.amplitude_combinations = self.define_amplitude()
  
  def define_pitch(self):
    '''
    입력: 없음

    출력: List of List of MIDI pitch의 Tensor형태 
         MIDI Pitch 리스트들의 리스트
        (예: [ [60, 64, 67], [61, 65, 68], [62, 66, 69] ])
    '''
    output = torch.zeros([24, 3]) # 24 note combinations with 3 notes each
    # TODO: 셰퍼드 톤을 만들기 위한 output을 완성하시오
    return output

  def define_amplitude(self):
    '''
    입력: 없음

    출력: List of List of amplitude의 Tensor형태 
         self.pitch_combinations의 각각에 해당하는 소리의 amplitude
        (예: [ [1.0, 0.2, 0.8], [1.0, 0.5, 0.5], [1.0, 0.8, 0.2] ])

    힌트: 사람이 듣는 소리의 크기는 amplitude의 log값에 비례하므로 amplitude를 이에 맞게 조절해야한다
    '''
    output = torch.zeros([24,3]) # 24 note combinations with 3 notes each
    # TODO: 셰퍼드 톤을 만들기 위한 output을 완성하시오
    return output

  def generate(self):
    outputs = []
    for i in range(self.iter):
      for j in range(len(self.pitch_combinations)):
        current_pitches = self.pitch_combinations[j]
        current_amplitudes = self.amplitude_combinations[j]
        audio = generate_multi_pitch_tone(current_pitches, current_amplitudes, self.dur, self.sr)
        audio[-int(self.fadeout_margin*self.sr):] *= torch.logspace(0, -3, int(self.fadeout_margin*self.sr))
        outputs.append(audio)
    return torch.cat(outputs, dim=-1)



class SpectrogramConverter:
  def __init__(self, n_fft, hop_size, sample_rate, num_mels, power=2.0):
    self.n_fft = n_fft
    self.hop_size = hop_size
    self.sr = sample_rate
    self.num_mels = num_mels
    self.window_tensor = torch.hann_window(n_fft)
    self.power = power

    self.mel_fb = self.get_melfilterbank()

  def get_spectrogram(self, audio):
    '''
    audio: 입력 오디오 샘플(torch.Tensor).
    n_fft: fft size. 한 번에 몇 개의 오디오 샘플을 푸리에 변환할지 결정
    hop_size: 오디오 

    출력: 오디오에 대해 fft_size = n_fft를 사용한 스펙트로그램 (torch.Tensor)
        absolute value만을 사용
    '''
    if audio.ndim == 2:
      if audio.shape[1] > audio.shape[0]:
        audio = audio.mean(dim=0)
      else:
        audio = audio.mean(dim=1)
    assert audio.ndim == 1, "Audio must be 1D tensor"
    # 문제 4-1
    # TODO: Complete this function with only using torch.fft.fft() function
    # You have to use self.window_tensor, self.n_fft, self.hop_size, self.power, and audio
    # You have to use torch.fft.fft() and torch.abs()
    # Usually, we use SQUARED VALUE (power=2.0) of the absolute value of the fft result as the spectrogram
    # You have to use torch.stack() to concatenate the result

    return 
  
  def get_melfilterbank(self):
    '''
    입력: 없음

    출력: mel filterbank (torch.Tensor)
    '''
    mel_fb = torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=self.sr, f_min=0.0, f_max=self.sr/2, n_stft=self.n_fft//2+1).fb
    return mel_fb
  
  def frequency_bin_to_hz(self, bin_index):
    '''
    입력: bin_index: Spectrogram에서 frequency bin의 인덱스

    출력: sr,n_fft에 해당하는 스펙트로그램 frequency bin index의 frequency Hz
    '''
    # 문제 4-2
    # TODO: 함수를 완성하시오
    # Hint: use self.sr and self.n_fft
    return 

  def time_bin_to_second(self, bin_index):
    '''
    입력: bin_index: Spectrogram에서 time bin의 인덱스
        sr: sampling rate (Hz)
        n_fft: fft size (sample)
        hop_size: Spectrogram의 각 프레임이 입력 오디오 상에서 몇 오디오 샘플 씩 떨어져있는지를 나타냄 

    출력: sr,n_fft,hop_size에 해당하는 스펙트로그램 time bin index가 입력 오디오 몇 초에 해당하는지를 반환
      second (float): 스펙트로그램 time frame의 frame_index에 해당하는 오디오 샘플의 시간 위치 (초). Window가 시작되는 샘플을 기준으로 함
    '''
    # 문제 4-2
    # TODO: 함수를 완성하시오
    # Hint: use self.sr, and self.hop_size

    return 

  def hz_to_mel(self, hz):
    '''
    입력: hz: frequency (torch.Tensor)

    출력: hz에 해당하는 mel scale의 값 (torch.Tensor)

    입력과 출력의 shape은 동일함
    '''
    # 문제 4-3
    # TODO: 함수를 완성하시오
    return 
  
  def convert_spec_to_mel_spec(self, spec):
    '''
    입력: spec: spectrogram (torch.Tensor)

    출력: mel spectrogram (torch.Tensor)
    '''
    # 문제 4-3
    # TODO: torch.mm(), self.mel_fb, spec을 이용하여 함수를 완성하시오
    # Hint: self.mel_fb는 ( n_fft//2+1, num_mels)의 shape을 가짐. 
    # Hint: Output은 (num_mels, time_frame)의 shape을 가짐 
    return
  
  def amplitude_to_db(self, spectrogram, eps=1e-10):
    '''
    spectrogram: 입력 스펙트로그램 (torch.Tensor)

    출력: 입력 스펙트로그램을 dB scale로 변환한 결과 (torch.Tensor)
    '''
    # 문제4-4
    # TODO: Complete this function
    # You have to use torch.log10() and torch.clamp()
    # To avoid log(0), you have to use eps to clamp the spectrogram

    return 
  
  def __call__(self, audio):
    '''
    audio: 입력 오디오 샘플(torch.Tensor).
    
    output: mel spectrogram in dB scale (torch.Tensor)
    '''
    # 문제 4-5
    # TODO: Complete this function using self.get_spectrogram(), self.convert_spec_to_mel_spec(), and self.amplitude_to_db()

    return 

def change_ytick_to_frequency(spec_converter):
  prev_yticks = plt.yticks()[0][1:-1] 
  ytick_labels= [spec_converter.frequency_bin_to_hz(bin_index) for bin_index in prev_yticks]
  plt.yticks(ticks=prev_yticks, labels=ytick_labels)

def change_xtick_to_seconds(spec_converter):
  prev_xticks = plt.xticks()[0][1:-1]
  xtick_labels= [spec_converter.time_bin_to_second(bin_index) for bin_index in prev_xticks]
  plt.xticks(ticks=prev_xticks, labels=xtick_labels)


if __name__ == "__main__":

  for harmonic_index in range(1,10):
    test_freq = 440 * harmonic_index
    print(f'{test_freq} Hz is {hz_to_midi_pitch(test_freq):.2f} in MIDI Pitch')

  for midi_pitch in range(60, 73):
    print(f'{midi_pitch} in MIDI Pitch is {midi_pitch_to_hz(midi_pitch):.2f} Hz')

  c_major_chord = [60, 64, 67]
  amp_list = [1, 1, 1]
  chord_tone = generate_multi_pitch_tone(c_major_chord, amp_list, 3, sr=SR)
  torchaudio.save('chord_tone.wav', chord_tone.unsqueeze(0)/torch.max(chord_tone), SR)


  fund_note = 69 # MIDI pitch for A4
  major_scale_in_hz = make_major_scale_hz_sequence(fund_note)
  scale_sine = generate_sequence_of_pitch(major_scale_in_hz, duration=0.5, sr=SR)
  torchaudio.save('scale_sine.wav', scale_sine.unsqueeze(0)/torch.max(scale_sine), SR)


  fund_pitch = 60
  just_intonation_chord = make_just_intonation_triad_in_midi_pitch(fund_pitch)
  amp_list = [1, 1, 1]
  just_chord_tone = generate_multi_pitch_tone(just_intonation_chord, amp_list, 3, sr=SR)
  torchaudio.save('just_chord_tone.wav', just_chord_tone.unsqueeze(0)/torch.max(just_chord_tone), SR)


  equal_temperament_chord = [fund_pitch, fund_pitch+4, fund_pitch+7]
  equal_chord_tone = generate_multi_pitch_tone(equal_temperament_chord, amp_list, 3, sr=SR)
  torchaudio.save('equal_chord_tone.wav', equal_chord_tone.unsqueeze(0)/torch.max(equal_chord_tone), SR)


  shetone = ShepardToneGenerator(SR,0.3,3)
  shepard_audio = shetone.generate()
  torchaudio.save('shepard.wav', shepard_audio.unsqueeze(0)/torch.max(shepard_audio), SR)

  spec_converter = SpectrogramConverter(n_fft=1024, hop_size=512, num_mels=128, sample_rate=16000)
  spec_converter = SpectrogramConverter(n_fft=1024, hop_size=512, num_mels=128, sample_rate=16000)
  
  your_audio = torch.randn(1, 16000*3)
  spec = spec_converter.get_spectrogram(your_audio)

  plt.figure(figsize=(10,10))
  plt.imshow(spec, origin='lower', aspect='auto', interpolation='nearest', norm=LogNorm())
  plt.savefig('spectrogram.png')
  plt.close()

  fft_size=1024
  hop_size=512
  spec = spec_converter.get_spectrogram(your_audio)

  plt.figure(figsize=(10,10))
  plt.imshow(spec, origin='lower', aspect='auto', interpolation='nearest', norm=LogNorm())
  change_ytick_to_frequency(spec_converter)
  change_xtick_to_seconds(spec_converter)
  plt.ylabel('Frequency (Hz)') 
  plt.xlabel('Time (seconds)') 
  plt.savefig('spec_with_label.png')
  plt.close()


  mel_scale = spec_converter.get_melfilterbank()

  hz_linspace = torch.linspace(0, 8000, 100)

  # TODO: Complete SpectrogramConverter.hz_to_mel() and use it
  mel_linspace = spec_converter.hz_to_mel(hz_linspace)

  plt.figure(figsize=(10,15))
  plt.subplot(2,1,1)
  plt.imshow(mel_scale, aspect='auto', origin='lower', interpolation='nearest')
  change_ytick_to_frequency(spec_converter)
  plt.ylabel('Frequency (Hz)') 
  plt.xlabel('Mel Filterbank Index') 

  plt.subplot(2,1,2)
  plt.plot(mel_linspace, hz_linspace)
  plt.savefig('mel_scale.png')
  plt.close()

  mel_spec= spec_converter.convert_spec_to_mel_spec(spec)

  # Plot
  plt.figure(figsize=(10,10))
  plt.title("Mel Spectrogram")
  plt.ylabel("Mel Bin")
  plt.imshow(mel_spec, origin='lower', aspect='auto', interpolation='nearest', norm=LogNorm()) 
  change_xtick_to_seconds(spec_converter)
  plt.xlabel('Time (seconds)') 
  plt.savefig('mel_spec.png')
  plt.close()

  db_spec = spec_converter.amplitude_to_db(mel_spec)

  plt.figure(figsize=(15,10))
  plt.subplot(2,2,1)
  plt.title("Amplitude Mel Spectrogram")
  plt.imshow(mel_spec, origin='lower', aspect='auto', interpolation="nearest")
  plt.colorbar()
  change_xtick_to_seconds(spec_converter)
  plt.ylabel('Mel Bin')
  plt.subplot(2,2,2)
  plt.title("Amplitude Mel Spectrum (Single frame of spectrogram)")
  plt.plot(mel_spec[:,mel_spec.shape[1]//2] )
  plt.xlabel('Mel Bin')
  plt.subplot(2,2,3)
  plt.title("dB Mel Spectrogram")
  plt.ylabel('Mel Bin')
  plt.imshow(db_spec, origin='lower', aspect='auto', interpolation="nearest")
  change_xtick_to_seconds(spec_converter)
  plt.colorbar()
  plt.subplot(2,2,4)
  plt.title("dB Mel Spectrum (Single frame of spectrogram)")
  plt.plot(db_spec[:,db_spec.shape[1]//2] )
  plt.xlabel('Mel Bin')

  plt.savefig('mel_spec_db.png')
  plt.close()


  spec_converter = SpectrogramConverter(n_fft=1024, hop_size=512, num_mels=128, sample_rate=16000)
  mel_db_spec = spec_converter(your_audio)

  ref_spec = torchaudio.transforms.AmplitudeToDB()(torchaudio.transforms.MelSpectrogram(n_fft=1024, hop_length=512, center=False)(your_audio))

  plt.imshow(mel_db_spec, origin='lower', aspect='auto', interpolation='nearest')
  plt.savefig('spec_converter_call.png')
  plt.close()
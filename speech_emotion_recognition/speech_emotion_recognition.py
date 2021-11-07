import librosa as lb
import soundfile as sf
import numpy as np
 

def extract_features(file_title, mfcc, chroma, mel):
  '''
  Arguments:
  file_title: as a path
  mfcc, chroma, mel: as boolians

  retruns:
  sound features(mfcc, chroma, mel) as a matrix
  '''
  try:
    with sf.SoundFile(file_title) as audio_recording:
      audio = audio_recording.read(dtype="float32")
      sample_rate = audio_recording.samplerate
      if chroma:
          stft=np.abs(lb.stft(audio))
          result=np.array([])
      if mfcc:
          mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
          result=np.hstack((result, mfccs))
      if chroma:
          chroma=np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
          result=np.hstack((result, chroma))
      if mel:
          mel=np.mean(lb.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
          result=np.hstack((result, mel))
      return result
  except :
    raise FileNotFoundError



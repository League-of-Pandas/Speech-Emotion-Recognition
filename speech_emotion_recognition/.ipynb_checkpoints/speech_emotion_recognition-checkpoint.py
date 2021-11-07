import librosa as lb
import soundfile as sf
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split

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

emotion_labels = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
focused_emotion_labels = ['happy', 'sad', 'angry', 'neutral']

def loading_audio_data():
    """
    Arguments: None
    it loads the sound features for each file and the related emotion for each feature
    return: Arrays
    """
    x = []
    y = []
    for file in glob('sounds/Actor_*/*.wav'):
        print(file)
        file_path = os.path.basename(file)
        emotion = emotion_labels[file_path.split("-")[2]]
        if emotion not in focused_emotion_labels:
            continue
        feature = extract_features(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    final_dataset = train_test_split(np.array(x), y, test_size=0.1, random_state=9)
    return final_dataset

X_train, X_test, y_train, y_test = loading_audio_data()


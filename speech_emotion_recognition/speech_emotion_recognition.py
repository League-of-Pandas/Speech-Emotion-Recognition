import librosa as lb
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from IPython.display import Audio
from statistics import mode
from librosa import display
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from playsound import playsound
from gtts import gTTS

import pickle


infile = open("result/69,2.model", 'rb')
model = pickle.load(infile)
infile.close()


def extract_features(file_title, mfcc, chroma, mel):
      """
      Arguments:
      file_title: as a path
      mfcc, chroma, mel: as boolians

      retruns:
      sound features(mfcc, chroma, mel) as a matrix
      """
      try:
         with sf.SoundFile(file_title) as audio_recording:
                audio = audio_recording.read(dtype = "float32")
                sample_rate = audio_recording.samplerate
                if chroma:
                      stft = np.abs(lb.stft(audio))
                      result = np.array([])
                if mfcc:
                      mfccs = np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
                      result = np.hstack((result, mfccs))
                if chroma:
                      chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                      result = np.hstack((result, chroma))
                if mel:
                      mel = np.mean(lb.feature.melspectrogram(audio, sr=sample_rate).T, axis=0)
                      result = np.hstack((result, mel))
         return result
      except:
         raise FileNotFoundError


def visualizing_sound(file):
    """
    Argument:
    a path for a (.wav) file

    return:
    1. spectogram of the choosen file
    2. waveform of the choosen file
    """
    x, fs = lb.load(file)
    lb.display.waveplot(x, sr=fs)
    X = lb.stft(x)
    Xdb = lb.amplitude_to_db(abs(X))
    plt.title('Waveform')
    plt.figure(figsize=(14, 5))
    plt.title('Spectogram')
    lb.display.specshow(Xdb, sr=fs, x_axis='time', y_axis='hz')
    plt.colorbar()


def speak(text):
    try:
        tts = gTTS(text = text, lang = "en")
        filename = "voice.mp3"
        tts.save(filename)
        playsound(filename)
        return "recording1.wav"
    except Exception as e:
        print("Exception: " + str(e))


def extract_features_from_user_input(path):
    return extract_features(path, mfcc=True, chroma=True, mel=True)


def extract_emotion_from_user_input(path):
    features = extract_features(path, mfcc=True, chroma=True, mel=True)
    result = model.predict(features.reshape(1, -1))
    return result[0]


def extract_user_input_emotion():
    """
    Arguments:
    None
    Takes input from the user as a path for a (.wav) file.
    return:
    1. The extracted emotion.
    2. The accuracy.
    3. waveform and spectrogram graphs if the user approved.
    """
    # speak("Welcome to the speech emotion recognition app.")
    # speak("please enter (yes) to enter a path to the desired filer or enter (Enter) to quit.")
    answer = input('please enter (yes) to choose a file or enter (Any Thing Else) to quit. \n >')
    if answer.lower() == "yes":
        # speak('please enter the path')
        path = input('please enter the path: \n >')
        try:
            return extract_emotion_from_user_input(path)
        except Exception:
            # speak("The file doesn't work, enter another file please.")
            print("The file doesn't work, enter another file please.")


def analyzing_multiple_emotions():
    """
    Arguments: None
    function takes number of sound records and analyze each one to give it related emotion
    return: list(the list of emotions that given from records in all records)
    """
    arr = []
    # speak("Please enter the number of records you want to analyze? ")
    number_of_files = input('Please enter the number of records you want to analyze \n')
    while not number_of_files.isdigit():
        # speak("Please enter number")
        number_of_files = input('Please enter number \n')
    for i in range(1, int(number_of_files) + 1):
        # speak("Please choose sound record number {i} ")
        print(f'Please choose sound record {i}?\n')
        path = input('please enter the path: \n >')
        try:
            feature = extract_features(path, mfcc=True, chroma=True, mel=True)
            result = model.predict(feature.reshape(1, -1))
            arr.append(result[0])
        except Exception as e:
            # speak("The file doesn't work, enter another file please")
            print("The file doesn't work, enter another file please")
    return arr


def suggest_books(mood):
    """
    function to check on mood and return a suggestion book dealing with this mood
    Argument: string (most common emotion)
    return: string (recommend book for related emotion)
    """
    motivational = ["The 5 AM Club: Own Your Morning. Elevate Your Life", "Think and Grow Rich by Napoleon Hill", "Unlimited Power by Tony Robbins", "How To Win Friends and Influence People by Dale Carnegie",
                    "The Four Hour Work Week by Tim Ferris", "The 7 Habits of Highly Effective People by Stephen Covey"]
    calm_down = ["Big Magic: Creative Living Beyond Fear by Elizabeth Gilbert", "A Book That Takes Its Time: An Unhurried Adventure in Creative Mindfulness", "Deep Listening by Jillian Pransky", "Just Sit: A Meditation Guidebook for People Who Know They Should But Don't by Sukey Novogratz"]
    entertainment = ["Harry Potter and the Prisoner of Azkaban by J.K. Rowling, Mary GrandPré", "FURIOUSLY HAPPY BY JENNY LAWSON", "Me Talk Pretty One Day by David Sedaris"]
    if mood == "happy" or mood == "neutral":
        book = random.choice(entertainment)
        # speak(f"you seem {mood} today so read {book}")
        return book
    elif mood == "sad":
        book = random.choice(motivational)
        # speak(f"you seem {mood} today so read {book}")
        return book
    elif mood == "fearful":
        book = random.choice(calm_down)
        # speak(f"you seem {mood} today so read {book}")
        return book


def take_input_for_suggesting_songs_and_books():
    """
    this function takes an answer from the user if they want to listen to a song or not
    and if they  want a recommendation for a book
    Arguments: None
    Returns: object
    """
    the_most_common_emotion = analyzing_multiple_emotions()
    mode(the_most_common_emotion)
    mood = mode(the_most_common_emotion)
    # speak("do you want a recommendation for a book ? (yes/no) ")
    answer = input("do you want a recommendation for a book ? (yes/no) \n >")
    if answer.lower() == "yes":
        book = suggest_books(mood)
        return book


# if __name__ == "__main__":
#     extract_features('sounds/Actor_01/03-01-01-01-01-01-01.wav', True, True, True)[0]
#     extract_user_input_emotion()
#     analyzing_multiple_emotions()
#     suggest_books("happy")
#     suggest_songs("happy")
#     take_input_for_suggesting_songs_and_books()
#     extract_features_from_user_input('happy/female2_happy_1a_2.wav')
#     extract_emotion_from_user_input('happy/female2_happy_1a_2.wav')


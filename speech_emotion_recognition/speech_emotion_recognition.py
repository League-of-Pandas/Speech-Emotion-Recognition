import librosa as lb
import soundfile as sf
import numpy as np
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

from pydub import AudioSegment
from pydub.playback import play

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

def speak(text,value=0):
    """
    given abillty to make a test play as sound

    Arguments:
    text: text want to be play as sound.

    value: if value is zero it will play otherways will not played.

    """
    if value == 0:
        try:
            tts= gTTS(text=text, lang="en")
            filename = "voice.mp3"
            tts.save(filename)
            song = AudioSegment.from_mp3("voice.mp3")
            quieter_song = song - 0
            play(quieter_song)
            quieter_song.export("voice.mp3", format='mp3')
            return "voice.mp3"
        except Exception as e:
            print("Exception: "+ str(e))
    return "voice.mp3"

def extract_features_from_user_input(path):
    return extract_features(path, mfcc=True, chroma=True, mel=True)

def extract_emotion_from_user_input(path):
    features = extract_features(path, mfcc=True, chroma=True, mel=True)
    result = model.predict(features.reshape(1, -1))
    return result[0]

def welcoming():
    """
    Welcoming for user
    """
    return(""" 
        *************************************************
        **                                             **
        **    Welcome to speech emotion recognition!   **
        **                                             **
        *************************************************
    
    """)

def extract_user_input_emotion(value=0):
    """
    Arguments:
    None
    Takes input from the user as a path for a (.wav) file.
    return:
    1. The extracted emotion.
    2. The accuracy.
    3. waveform and spectrogram graphs if the user approved.
    """
    print(welcoming())
    
    speak("Welcome to the speech emotion recognition app.",value)
    speak("please enter (yes) to enter a path to the desired file or press (Enter) to quit.",value)
    answer = input('please enter (yes) to enter a path or press (Enter) to quit. \n >')
    if answer.lower() == "yes":
        speak('please enter the path',value)
        path = input('please enter the path: \n >')
        try:
            return extract_emotion_from_user_input(path)
        except Exception:
            speak("The file doesn't work, enter another file please.",value)
            print("The file doesn't work, enter another file please.")

def good_bye():
    """
    Good bye message for user
    """
    return(""" 
        **************************************
        **    Thanks for using our app      **
        **    hope you come here again      **
        **************************************
    
        """)

def analyzing_multiple_emotions(value=0):
    """
    Arguments: None
    function takes number of sound records and analyze each one to give it related emotion
    return: list(the list of emotions that given from records in all records)
    """
    print(""" 
        *********************************************************
        **    Welcome  to  speech  emotion  recognition!       **
        **    This app will help you to analyz your mood      **
        **    and give you recommendation that help you        **
        *********************************************************
    
        """)
    play_songs("melody-of-nature-main-6672.wav",value)
    arr = []

    speak("This app will help you to analyz your mood and give you recommendation that help you ",value)
    speak("Please enter the number of records you want to analyze? ",value)
    number_of_files = input('Please enter the number of records you want to analyze \n > ')
    while not number_of_files.isdigit():
        speak("Please enter a number",value)
        number_of_files = input('Please enter number \n')
    for i in range(1, int(number_of_files) + 1):
        print(f'Please choose sound path number {i}?')
        speak(f"Please choose sound path number {i} ",value)
        path = input("> ")
        try:
            feature = extract_features(path, mfcc=True, chroma=True, mel=True)
            result = model.predict(feature.reshape(1, -1))
            arr.append(result[0])
        except Exception as e:
            speak("The file doesn't work, enter another file please",value)
            print("The file doesn't work, enter another file please")
    return arr

def play_songs(path,value=0):
    """
    given abillty to make a .wav file played.

    Arguments:
    path: path of the file

    value: if value is zero it will play otherways will not played.

    """
    if value == 0 :
        import simpleaudio as sa
        wave_object = sa.WaveObject.from_wave_file(path)

        play_object = wave_object.play()
        x= input("Enter q to stop the music \n > ")
        if x == "q":
            play_object.stop()
        return "Played"
    return "Not Play"
        
def suggest_songs(mood,value=0):
    """
    this function takes the dominant emotion from a set of sound files for a person and suggests a song based on their mood
    Arguments: string
    returns: string, object
    """
    if mood == "happy" or mood == "neutral":
        speak(f"Because you are {mood} today listen to entertaining songs",value)
        song = random.choice(os.listdir("entertaining songs"))
        path = f'entertaining songs/{song}'
        play_songs(path,value)
        return path
    elif mood == "fearful":
        speak(f"Because you are {mood} today listen to calming songs ",value)
        song = random.choice(os.listdir("calming songs"))
        path = f'calming songs/{song}'
        play_songs(path,value)
        return path
    elif mood == "sad":
        speak(f"Because you are {mood} today listen to uplifting songs",value)
        song = random.choice(os.listdir("uplifting songs"))
        path = f'uplifting songs/{song}'
        play_songs(path,value)
        return path

def suggest_books(mood,value=0):
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
        print(f"you seem {mood} today so read {book}")
        speak(f"you seem {mood} today so read {book}",value)
        return book
    elif mood == "sad":
        book = random.choice(motivational)
        print(f"you seem {mood} today so read {book}")
        speak(f"you seem {mood} today so read {book}",value)
        return book
    elif mood == "fearful":
        book = random.choice(calm_down)
        print(f"you seem {mood} today so read {book}")
        speak(f"you seem {mood} today so read {book}",value)
        return book

def take_input_for_suggesting_songs_and_books(value=0):
    """
    this function takes an answer from the user if they want to listen to a song or not
    and if they  want a recommendation for a book
    Arguments: None
    Returns: object
    """
    the_most_common_emotion = analyzing_multiple_emotions(value)
    mode(the_most_common_emotion)
    mood = mode(the_most_common_emotion)
    speak(f"you seem {mood} today")
    print(f"you seem {mood} today")
    speak("do you want to listen to a song that helps you ? (yes/no)",value)
    answer = input("do you want to listen to a song that helps you ? (yes/no) \n >")
    if answer.lower() == "yes":
        path = suggest_songs(mood,value)
    
    speak("do you want a recommendation for a book ? (yes/no) ",value)
    answer = input("do you want a recommendation for a book ? (yes/no) \n >")
    if answer.lower() == "yes":
        book = suggest_books(mood,value)
        print(good_bye())
        play_songs("melody-of-nature-main-6672.wav",value)
        return book
    print(good_bye())
    play_songs("melody-of-nature-main-6672.wav",value)


    take_input_for_suggesting_songs_and_books(1)

if __name__ == "__main__":
    extract_features('sounds/Actor_01/03-01-01-01-01-01-01.wav',True,True,True)[0]
    extract_user_input_emotion(1)
    analyzing_multiple_emotions(1)
    suggest_books("happy",1)
    suggest_songs("happy",1)
    take_input_for_suggesting_songs_and_books(1)
    speak("hello",1)
    extract_features_from_user_input('happy/female2_happy_1a_2.wav')
    extract_emotion_from_user_input('happy/female2_happy_1a_2.wav')
    play_songs("uplifting songs/Alive-Sia (lyrics).wav", 1)

    welcoming()
    good_bye()
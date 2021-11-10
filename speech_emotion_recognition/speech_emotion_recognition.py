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
# from tkinter import filedialog
# from playsound import playsound
# from gtts import gTTS


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
                audio = audio_recording.read(dtype="float32")
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


emotion_labels = {
  '01': 'neutral',
  '02': 'calm',
  '03': 'happy',
  '04': 'sad',
  '05': 'angry',
  '06': 'fearful',
  '07': 'disgust',
  '08': 'surprised'
}
focused_emotion_labels = ['happy', 'sad', 'fearful', 'neutral']


def loading_audio_data():
    """
    Arguments: None
    it loads the sound features for each file and the related emotion for each feature
    return: Arrays
    """
    x = []
    y = []
    for file in glob('../sounds/Actor_01/*.wav'):
        file_path = os.path.basename(file)
        emotion = emotion_labels[file_path.split("-")[2]]
        if emotion not in focused_emotion_labels:
            continue
        feature = extract_features(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    final_dataset = train_test_split(np.array(x), y, test_size=0.1, random_state=9)
    return final_dataset


model = MLPClassifier(hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=400)
X_train, X_test, y_train, y_test = loading_audio_data()
model.fit(X_train, y_train)


def calculate_trained_model_accuracy():

    """
    Arguments:
    None

    this function trains the model and calculates the accuracy of it
    Arguments: None

    retruns: value (accuracy)
    """

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    return accuracy



def visualizing_sound(file):
    '''
    Argument:
    a path for a (.wav) file

    return:
    1. spectogram of the choosen file
    2. waveform of the choosen file
    '''
    x, fs = lb.load(file)
    lb.display.waveplot(x, sr=fs)
    X = lb.stft(x)
    Xdb = lb.amplitude_to_db(abs(X))
    plt.title('Waveform')
    plt.figure(figsize=(14, 5))
    plt.title('Spectogram')
    lb.display.specshow(Xdb, sr=fs, x_axis='time', y_axis='hz')
    plt.colorbar()


# def speek(text):
#     try:
#         tts= gTTS(text=text, lang="en")
#         filename = "voice.mp3"
#         tts.save(filename)
#         playsound(filename)
#         return "recording1.wav"
#     except Exception as e:
#         print("Exception: "+ str(e))


# def extract_user_input_emotion():
#     """
#     Arguments:
#     None
#     Takes input from the user as a path for a (.wav) file.
#     return:
#     1. The extracted emotion.
#     2. The accuracy.
#     3. waveform and spectogram graphs if the user approved.
#     """
#     speek("Welcome to the speech emotion recognitin app.")
#     speek("please enter (yes) to choose a file or enter (speak) to to analize your voice live or enter (q) to quit.")
#     answer = input('please enter (yes) to choose a file or enter (speak) to to analize your voice live or enter (q) to quit.')
#     if answer.lower() == "yes":
#         try:
#             path=filedialog.askopenfilename()
#             speek("Would you like to see the Spectogram and Waveform of the choosen file? (yes/no).")
#             visualization=input('Would you like to see the Spectogram and Waveform of the choosen file? (yes/no)')
#             if visualization.lower()=='yes':
#                 visualizing_sound(path)
#             features=extract_features(path,mfcc=True, chroma=True, mel=True)
#             result = model.predict(features.reshape(1,-1))
#             speek(f"The extracted emotion is : {result[0]}")
#             return f"The extracted emotion is : {result[0]}"
#         except Exception as e:
#             speek("The file doesn't work, enter another file please.")
#             print("The file doesn't work, enter another file please.")
#     elif answer.lower()=='speak':
#         speek('speak now')
#         print('speak now')
#         speech=listen()
#         speek("Would you like to see the Spectogram and Waveform of the choosen file? (yes/no).")
#         visualization=input('Would you like to see the Spectogram and Waveform of the choosen file? (yes/no)')
#         if visualization.lower()=='yes':
#             visualizing_sound(speech)
#         features=extract_features(speech,mfcc=True, chroma=True, mel=True)
#         result = model.predict(features.reshape(1,-1))
#         speek(f"The extracted emotion is : {result[0]}")
#         return f"The extracted emotion is : {result[0]}"

# def analyzing_multiple_emotions():
#
#     """
#     Arguments: None
#     function takes number of sound records and analyze each one to give it related emotion
#     return: list(the list of emotions that given from records in all records)
#     """
#     arr = []
#     number_of_files = input('Please enter the number of records you want to analyze \n')
#     while not number_of_files.isdigit():
#         number_of_files = input('Please enter number \n')
#     for i in range(1, int(number_of_files) + 1):
#         try:
#             file_path = input(f'Please choose sound record {i}?\n')
#             feature = extract_features(file_path, mfcc=True, chroma=True, mel=True)
#             result = model.predict(feature.reshape(1, -1))
#             arr.append(result[0])
#         except Exception as e:
#             print("The file doesn't work, enter another file please")
#     return arr

#
# def suggest_songs(mood):
#     """
#     this function takes the dominant emotion from a set of sound files for a person and suggests a song based on their mood
#     Arguments: string
#     returns: string, object
#     """
#     if mood == "happy" or mood == "neutral":
#         song = random.choice(os.listdir("entertaining songs"))
#         path = f'entertaining songs/{song}'
#         play=Audio(path, rate=250)
#         return play, path
#     elif mood == "fearful":
#         song = random.choice(os.listdir("calming songs"))
#         path = f'calming songs/{song}'
#         play=Audio(path, rate=250)
#         return play, path
#     elif mood == "sad":
#         song = random.choice(os.listdir("uplifting songs"))
#         path = f'uplifting songs/{song}'
#         play=Audio(path, rate=250)
#         return play, path

#
# def suggest_books(mood):
#     """
#     function to check on mood and return a suggestion book dealing with this mood
#     Argument: string (most common emotion)
#     return: string (recommened book for related emotion)
#     """
#     motivational = ["The 5 AM Club: Own Your Morning. Elevate Your Life", "Think and Grow Rich by Napoleon Hill", "Unlimited Power by Tony Robbins", "How To Win Friends and Influence People by Dale Carnegie",
#                     "The Four Hour Work Week by Tim Ferriss", "The 7 Habits of Highly Effective People by Stephen Covey"]
#     calm_down = ["Big Magic: Creative Living Beyond Fear by Elizabeth Gilbert", "A Book That Takes Its Time: An Unhurried Adventure in Creative Mindfulness", "Deep Listening by Jillian Pransky", "Just Sit: A Meditation Guidebook for People Who Know They Should But Don't by Sukey Novogratz"]
#     entertainment = ["Harry Potter and the Prisoner of Azkaban by J.K. Rowling,Mary GrandPré", "FURIOUSLY HAPPY BY JENNY LAWSON", "Me Talk Pretty One Day by David Sedaris"]
#     if mood == "happy" or mood == "neutral":
#         book = random.choice(entertainment)
#         print(f"you seem {mood} today so read {book}")
#         return book
#     elif mood == "sad":
#         book = random.choice(motivational)
#         print(f"you seem {mood} today so read {book}")
#         return book
#     elif mood == "fearful":
#         book = random.choice(calm_down)
#         print(f"you seem {mood} today so read {book}")
#         return book
#
#
# def take_input_for_suggesting_songs_and_books():
#     """
#     this function takes an answer from the user if they want to listen to a song or not
#     and if they  want a recommendation for a book
#     Arguments: None
#     Returns: object
#     """
#     the_most_common_emotion = analyzing_multiple_emotions()
#     mode(the_most_common_emotion)
#     mood = mode(the_most_common_emotion)
#     print(f"you seem {mood} today")
#     # answer = input("do you want to listen to a song that helps you ? (yes/no) \n >")
#     # if answer.lower() == "yes":
#     #     play,path=suggest_songs(mood)
#     #     return play
#     answer = input("do you want a recommendation for a book ? (yes/no) \n >")
#     if answer.lower() == "yes":
#         book = suggest_books(mood)
#         return book


if __name__ == "__main__":
    # take_input_for_suggesting_songs_and_books()
    print(visualizing_sound("sm1_cln.wav"))
    print('hello world')


import builtins
from speech_emotion_recognition import __version__
from speech_emotion_recognition import speech_emotion_recognition
from tkinter import filedialog
from mock import Mock
from unittest.mock import patch
import pytest
from statistics import mode

def test_version():
    assert __version__ == '0.1.0'


def test_extract_features():
    # Arrange
    expected=-726.2172241210938
    # Act
    actual= speech_emotion_recognition.extract_features('sounds/Actor_01/03-01-01-01-01-01-01.wav',True,True,True)[0]
    # Assert
    assert actual==expected

def test_extract_features_fail():
    with pytest.raises(FileNotFoundError):
        path = "missing.txt"
        speech_emotion_recognition.extract_features(path,True,True,True)


def test_extracting_features_from_input():
    actual = speech_emotion_recognition.extract_sound_features_from_user_input('/home/user/Downloads/JL(wav+txt)/female1_angry_1b_1.wav')
    expected = -529.4510498046875
    assert list(actual)[0] == expected

def test_analyzing_multiple_emotions():
    expected = 'None'
    common = 'None'
    if common == 'happy' or common == 'sad' or common == 'neutral' or common == 'fearful':
        expected = common
    actual = mode(speech_emotion_recognition.the_most_common_emotion)
    assert expected == actual




    
           
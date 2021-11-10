
import builtins
from speech_emotion_recognition import __version__
from speech_emotion_recognition import speech_emotion_recognition
from tkinter import filedialog
from unittest.mock import patch
import pytest


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

def test_extract_emotion_from_user_input():
    # Arrange
    expected=['happy']
    # Act
    actual= speech_emotion_recognition.extract_emotion_from_user_input('happy/female2_happy_1a_2.wav',True,True,True)[0]
    # Assert
    assert actual==expected

def test_extract_emotion_from_user_input_fail():
    with pytest.raises(FileNotFoundError):
        path = "missing.txt"
        speech_emotion_recognition.extract_emotion_from_user_input(path,True,True,True)

def test_extracting_features_from_input():
    actual = speech_emotion_recognition.extract_features_from_user_input('sm1_cln.wav')
    expected = -284.02972412109375
    assert list(actual)[0] == expected

def test_suggest_song():   
    arr = ['calming songs/Ava Max - So Am I [Official Music Video].mp3','calming songs/Josh Groban - You Raise Me Up (Official Music Video) _ Warner Vault.mp3','calming songs/Sia - Bird Set Free (Lyrics).mp3']
    play,path = speech_emotion_recognition.suggest_songs('fearful')
    actual = path
    assert actual in arr

def test_suggest_books():
    expected = ["Big Magic: Creative Living Beyond Fear by Elizabeth Gilbert", "A Book That Takes Its Time: An Unhurried Adventure in Creative Mindfulness", "Deep Listening by Jillian Pransky", "Just Sit: A Meditation Guidebook for People Who Know They Should But Don't by Sukey Novogratz"]
    actual = speech_emotion_recognition.suggest_books('fearful')
    assert actual in expected

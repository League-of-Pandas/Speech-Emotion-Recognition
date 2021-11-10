
import builtins
from speech_emotion_recognition import __version__
from speech_emotion_recognition import speech_emotion_recognition
from unittest.mock import patch
import pytest
import mock


def test_version():
    assert __version__ == '0.1.0'


def test_extract_features():
    # Arrange
    expected = -726.2172241210938
    # Act
    actual = speech_emotion_recognition.extract_features('sounds/Actor_01/03-01-01-01-01-01-01.wav', True, True, True)[0]
    # Assert
    assert actual == expected


def test_extract_features_fail():
    with pytest.raises(FileNotFoundError):
        path = "missing.txt"
        speech_emotion_recognition.extract_features(path, True, True, True)


def test_extracting_features_from_input():
    actual = speech_emotion_recognition.extract_features_from_user_input('sm1_cln.wav')
    expected = -284.02972412109375
    assert list(actual)[0] == expected


def test_suggest_song():
    arr = ['calming songs/Ava Max - So Am I [Official Music Video].mp3','calming songs/Josh Groban - You Raise Me Up (Official Music Video) _ Warner Vault.mp3','calming songs/Sia - Bird Set Free (Lyrics).mp3']
    play, path = speech_emotion_recognition.suggest_songs('fearful')
    actual = path
    assert actual in arr

def test_suggest_song_two():
    arr = ['entertaining songs/Imagine Dragons - On Top Of The World (Lyrics).mp3','entertaining songs/RISE (ft. The Glitch Mob, Mako, and The Word Alive) _ Worlds 2018 - League of Legends.mp3','entertaining songs/Touch The Sky (From _Brave__Soundtrack).mp3']
    play, path = speech_emotion_recognition.suggest_songs('happy')
    actual = path
    assert actual in arr

def test_suggest_song_three():
    arr = ['uplifting songs/Alive-Sia (lyrics).mp3', "uplifting songs/The Greatest Showman Cast - This Is Me (Official Lyric Video).mp3","uplifting songs/Imagine Dragons - It's Time (Lyrics).mp3"]
    play, path = speech_emotion_recognition.suggest_songs('sad')
    actual = path
    assert actual in arr

def test_mock_input_emotion_two():
    with mock.patch("builtins.input", return_value="yes"):
        assert speech_emotion_recognition.extract_emotion_from_user_input('sad/female1_sad_7a_1.wav') == 'sad'

def test_mock_input_emotion_three():
    with mock.patch("builtins.input", return_value="yes"):
        assert speech_emotion_recognition.extract_emotion_from_user_input('fearful/female2_fearful_6a_1.wav') == 'fearful'

def test_mock_input_emotion_four():
    with pytest.raises(Exception):
      speech_emotion_recognition.extract_emotion_from_user_input('female2_fearful_6a_1.wav')
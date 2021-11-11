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

def test_extract_emotion():
    with mock.patch("builtins.input", return_value = "yes"):
       assert speech_emotion_recognition.extract_user_input_emotion(2) == None
        
def test_suggest_song():   
    arr = ['calming songs/Ava Max - So Am I [Official Music Video].wav','calming songs/Josh Groban - You Raise Me Up (Official Music Video) _ Warner Vault.wav']
    path = speech_emotion_recognition.suggest_songs('fearful',2)
    actual = path
    assert actual in arr

def test_suggest_song_two():
    arr = ['entertaining songs/RISE (ft. The Glitch Mob, Mako, and The Word Alive) _ Worlds 2018 - League of Legends.wav',"entertaining songs/Brave _ Touch the Sky _ Disney Sing-Along.wav"]
    path = speech_emotion_recognition.suggest_songs('happy',2)
    actual = path
    assert actual in arr

def test_suggest_song_three():
    arr = ['uplifting songs/Alive-Sia (lyrics).wav', "uplifting songs/The Greatest Showman Cast - This Is Me (Official Lyric Video).wav"]
    path = speech_emotion_recognition.suggest_songs('sad',2)
    actual = path
    assert actual in arr

def test_suggest_books():
    expected = ["Big Magic: Creative Living Beyond Fear by Elizabeth Gilbert", "A Book That Takes Its Time: An Unhurried Adventure in Creative Mindfulness", "Deep Listening by Jillian Pransky", "Just Sit: A Meditation Guidebook for People Who Know They Should But Don't by Sukey Novogratz"]
    actual = speech_emotion_recognition.suggest_books('fearful',2)
    assert actual in expected

def test_suggest_books_two():
    expected = ["The 5 AM Club: Own Your Morning. Elevate Your Life", "Think and Grow Rich by Napoleon Hill", "Unlimited Power by Tony Robbins", "How To Win Friends and Influence People by Dale Carnegie",
                    "The Four Hour Work Week by Tim Ferris", "The 7 Habits of Highly Effective People by Stephen Covey"]
    actual = speech_emotion_recognition.suggest_books('sad',2)
    assert actual in expected

def test_suggest_books_three():
    expected = entertainment = ["Harry Potter and the Prisoner of Azkaban by J.K. Rowling, Mary GrandPré", "FURIOUSLY HAPPY BY JENNY LAWSON", "Me Talk Pretty One Day by David Sedaris"]
    actual = speech_emotion_recognition.suggest_books('happy',2)
    assert actual in expected

def test_suggest_books_input():
    with pytest.raises(Exception):
        speech_emotion_recognition.take_input_for_suggesting_songs_and_books(1)

def test_mock_input_emotion():
     with mock.patch("builtins.input", return_value="yes"):
        assert speech_emotion_recognition.extract_emotion_from_user_input('happy/female2_happy_1a_2.wav') == 'happy'

def test_mock_input_emotion_two():
    with mock.patch("builtins.input", return_value="yes"):
        assert speech_emotion_recognition.extract_emotion_from_user_input('sad/female1_sad_7a_1.wav') == 'sad'

def test_mock_input_emotion_three():
    with mock.patch("builtins.input", return_value="yes"):
        assert speech_emotion_recognition.extract_emotion_from_user_input('fearful/female2_fearful_6a_1.wav') == 'fearful'

def test_mock_input_emotion_four():
    with pytest.raises(Exception):
      speech_emotion_recognition.extract_emotion_from_user_input('..female2_fearful_6a_1.wav')

def test_analyzing_multiple_records():
    with mock.patch("builtins.input", return_value="1"):
        assert speech_emotion_recognition.analyzing_multiple_emotions(2) == []

def test_analyzing_multiple_records_two():
    with mock.patch("builtins.input", return_value="2"):
        assert speech_emotion_recognition.analyzing_multiple_emotions(2) == []

def test_analyzing_multiple_records_three():
    with pytest.raises(Exception):
      speech_emotion_recognition.analyzing_multiple_emotions(2)

def test_speak():
    actual = speech_emotion_recognition.speak("Hello", 1)
    expected = "voice.mp3"
    assert actual == expected

def test_play_songs():
    with pytest.raises(Exception):
        speech_emotion_recognition.play_songs("recording1.wav")

def test_play_songs_fail():
    actual = speech_emotion_recognition.play_songs("uplifting songs/Alive-Sia (lyrics).wav", 1)
    expected = "Not Play"
    assert actual == expected

def test_welcoming():
    actual = speech_emotion_recognition.welcoming()
    expected = """ 
        *************************************************
        **                                             **
        **    Welcome to speech emotion recognition!   **
        **                                             **
        *************************************************
    
    """
    assert actual == expected

def test_good_bye():
    actual = speech_emotion_recognition.good_bye()
    expected =  """ 
        **************************************
        **    Thanks for using our app      **
        **    hope you come here again      **
        **************************************
    
        """
    assert actual == expected

    
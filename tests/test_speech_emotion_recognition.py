from speech_emotion_recognition import __version__
from speech_emotion_recognition.speech_emotion_recognition import accuracy, extract_features
import pytest

def test_version():
    assert __version__ == '0.1.0'


def test_extract_features():
    # Arrange
    expected=-726.2172241210938
    # Act
    actual=extract_features('sounds/Actor_01/03-01-01-01-01-01-01.wav',True,True,True)[0]
    # Assert
    assert actual==expected

def test_extract_features_fail():
    with pytest.raises(FileNotFoundError):
        path = "missing.txt"
        extract_features(path,True,True,True)
   

    
           
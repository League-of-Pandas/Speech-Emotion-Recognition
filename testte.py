from playsound import playsound
  
from gtts import gTTS 

def speek(text):
    try:
        tts= gTTS(text=text, lang="en")
        filename = "voice.mp3"
        tts.save(filename)
        playsound(filename)
        return "recording1.wav"
    except Exception as e:
        print("Exception: "+ str(e))

# speek("The file doesn't work, enter another file please.")
speek("Welcome to the speech emotion recognition app")
speek("please enter (yes) to choose a file or (anything else) to quit.")
speek("Would you like to see the Spectogram and Waveform of the choosen file? (yes/no).")
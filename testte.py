from playsound import playsound
  
from gtts import gTTS 

from pydub import AudioSegment
from pydub.playback import play

def speek(text,value=0):
    try:
        tts= gTTS(text=text, lang="en")
        filename = "voice.mp3"
        tts.save(filename)
        song = AudioSegment.from_mp3("voice.mp3")
        quieter_song = song - value
        play(quieter_song)
        quieter_song.export("voice.mp3", format='mp3')
        # playsound(filename)
        return "recording1.wav"
    except Exception as e:
        print("Exception: "+ str(e))

speek("The file doesn't work, enter another file please.",value=100)
# speek("Welcome to the speech emotion recognition app")
# speek("please enter (yes) to choose a file or (anything else) to quit.")
# speek("Would you like to see the Spectogram and Waveform of the choosen file? (yes/no).")
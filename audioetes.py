import simpleaudio as sa
  
# define an object to play
wave_object = sa.WaveObject.from_wave_file('Brave _ Touch the Sky _ Disney Sing-Along.wav')
# print('playing sound using simpleaudio')
  
# define an object to control the play
play_object = wave_object.play()
x= input()
if x == "q":
    play_object.stop()
#
# pyinstaller --onefile gpt4_test.py
#
import io
import sys

import keyboard
import openai
import pyaudio
import pyttsx3
#from PyWave import *
import PyWave as wave
from mss import mss



import speech_recognition as sr

def get_input():
    r = sr.Recognizer()
    print(r)
    with sr.Microphone() as source:
        print(source)
        audio = r.listen(source)

    try:
        said = r.recognize_google(audio)
        print(said)
    except Exception as e:
        print("Exception:  " + str(e))
    return said

print(get_input())

sys.exit()

#################################################
# Speech recording
#
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5

# PyAudio inicializálása
p = pyaudio.PyAudio()


def record(p):
    frames = []
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Recording...")
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        # Ha a felhasználó lenyom egy billentyűt, akkor a felvétel leáll
        if keyboard.is_pressed('q'):
            break
    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return frames

# Get the wav audio in bytes
def save(frames):
    buffer = io.BytesIO()
    wf = wave.open(buffer, 'wb')

    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    data = buffer.getvalue()
    buffer.close()
    return data




# #################################################
# speech -to-text
#


########################################################x
# Screendump
#
# with mss() as sct:
#     sct.shot(mon=-1, output="fullscreen.png")


#####################################################
# initiate the text-to-speech engine
#
engine = pyttsx3.init()
# Set language
engine.setProperty('language', 'hu')
voices = engine.getProperty('voices')
for i, voice in enumerate(voices):
    if voice.id == 'hungarian' :
       engine.setProperty('voice', voices[i].id)
       break
 
# say it
def say(text) :
    engine.say(text)
    # play
    engine.runAndWait()

 
#The main loop


while True :
    # REcord speech
    frames = record(p)
    data = save(frames)
    # Send to speech-to-text
    # TODO: call remote method
    text = "Ez egy példa szöveg, amit bemondtunk, majd szövveggé alakítottuk."
    # TODE: Call ChatGPT
    say(text)




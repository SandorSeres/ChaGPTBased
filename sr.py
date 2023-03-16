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

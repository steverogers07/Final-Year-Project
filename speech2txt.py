import speech_recognition as sr
#speechtotext
def get_Question():
    print('Tell the question.')
    r= sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, phrase_time_limit = 6)
    print('Working on it...')
    said= ""
    try:
        said = r.recognize_google(audio, language="en-us")
        print(said)
    except Exception as e:
        print(e)
    return said
# get_Question()


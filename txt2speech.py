from gtts import gTTS 
from playsound import playsound 
def txt2speech(text): 
	var = gTTS(text = text,lang = 'en') 
	var.save('eng.mp3')
	playsound("eng.mp3")
    # close("./eng.mp3")
	return
# txt2speech('Satyasai Jagannath Nanda urf Lodu Nanda')
# from num2words import num2words
# from subprocess import call

# def txt2speech(text):
#     cmd_beg= 'espeak '
#     cmd_end= ' | aplay /home/pi/Desktop/Text.wav  2>/dev/null' # To play back the stored .wav file and to dump the std errors to /dev/null
#     cmd_out= '--stdout > /home/pi/Desktop/Text.wav ' # To store the voice file
#     print(text)

#     #Replacing ' ' with '_' to identify words in the text entered
#     text = text.replace(' ', '_')

#     #Calls the Espeak TTS Engine to read aloud a Text
#     call([cmd_beg+cmd_out+text+cmd_end], shell=True)
#     return
# txt2speech('Hello hello 1,2,3')



# # Import the required module for text
# # to speech conversion
# from gtts import gTTS

# # This module is imported so that we can
# # play the converted audio
# import os

# # The text that you want to convert to audio
# mytext = 'Welcome to geeksforgeeks!'

# # Language in which you want to convert
# language = 'en'

# # Passing the text and language to the engine,
# # here we have marked slow=False. Which tells
# # the module that the converted audio should
# # have a high speed
# myobj = gTTS(text=mytext, lang=language, slow=False)

# # Saving the converted audio in a mp3 file named
# # welcome
# myobj.save("welcome.mp3")

# # Playing the converted file
# os.system("mpg321 welcome.mp3")

import sys
import os
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils

import h5py
#import cv2
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models

import config
import data
import utils
from resnet import resnet as caffe_resnet
import speech_recognition as sr

from gtts import gTTS 
from playsound import playsound
from translate import Translator
from googletrans import Translator as GT

from num2words import num2words
from subprocess import call

from picamera import PiCamera
from time import sleep

from textblob import TextBlob
#speechtotext

said=""
def get_Question(language):
    r= sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, phrase_time_limit = 7)
        said= ""
        try:
            said = r.recognize_google(audio,language=language)
            print('Said:', said)
        except Exception as e:
            print(e)
    if(language == 'hi-In' or language == 'hi'):
        print('converting hindi to english')
        # trans = GT()
        trans = Translator(to_lang='english', from_lang = 'hindi')
        #hi_blob = TextBlob(said)
        #print('Language:', hi_blob.detect_language())
        #print('Translated : ',hi_blob.translate(to='en'))
        #said = str(hi_blob.translate(to='en'))
        #translation = trans.translate(said)
        # print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
        # said = translation.text
        said = trans.translate(said)
    said = said.replace('colour', 'color')
    return said
#get_Question()
#print(said)
def txt2speech(text, lan='hi'):
    if lan=='hi':
        cmd_beg = 'espeak -s110 -vhi'
    else:
        cmd_beg= 'espeak -s110 '
    cmd_end= ' | aplay /home/pi/Desktop/Text.wav  2>/dev/null' # To play back the stored .wav file and to dump the std errors to /dev/null
    cmd_out= '--stdout > /home/pi/Desktop/Text.wav ' # To store the voice file
    print(text)

    #Replacing ' ' with '_' to identify words in the text entered
    text = text.replace(' ', '_')

    #Calls the Espeak TTS Engine to read aloud a Text
    call([cmd_beg+cmd_out+text+cmd_end], shell=True)
    return

# def take_Photo():
#     key = cv2.waitKey(1)
#     webcam = cv2.VideoCapture(0)
#     while True:
#         try:
#             check, frame = webcam.read()
#             cv2.imshow("Capturing", frame) 
#             key = cv2.waitKey(1)
#             if key == ord('s'):
#                 cv2.imwrite(filename='image.jpg', img=frame)
#                 webcam.release()
#                 #img_new = cv2.imshow("Captured Image", img_new)  
#                 cv2.waitKey(1650)
#                 cv2.destroyAllWindows()
#                 break
#             elif key == ord('q'):
#                 print("Turning off camera.")
#                 webcam.release()
#                 print("Camera off.")
#                 print("Program ended.")
#                 cv2.destroyAllWindows()
#                 break
#         except(KeyboardInterrupt):
#                 print("Turning off camera.")
#                 webcam.release()
#                 print("Camera off.")
#                 print("Program ended.")
#                 cv2.destroyAllWindows()
#                 break
def take_Photo():
    print('Taking Photo......') 
    camera = PiCamera()
    camera.rotation = 180
    camera.start_preview()
    sleep(5)
    camera.capture('image.jpg')
    camera.stop_preview()
    camera.close()
    print('Picture taken.......')


def prepare_questions(questions):
    '''
    Remove punctuation marks and spaces. Returns list
    '''
    questions = [questions]
    for question in questions:
        question = question.lower()[:-1]
        yield question.split(' ')

def encode_question(question):
    '''
    Encode questions
    Get ids using vocabulary created using tokens during training
    '''
    vec = torch.zeros(len(question)).long()
    with open(config.vocabulary_path, 'r') as fd:
        vocab_json = json.load(fd)
    token_to_index = vocab_json['question']
    for i, token in enumerate(question):
        index = token_to_index.get(token, 0)
        vec[i] = index
    return vec, len(question)

class Net(nn.Module):
    '''
    Loading Resnet pretrained model to get image features
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def encode_img(net,img_path):
    '''
    Encoding input image using Resnet features. Resizes input image to config.image_size
    '''
    cudnn.benchmark = True
    transform = utils.get_transform(config.image_size, config.central_fraction)
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    ix,iy = img.size()[1],img.size()[2]
    net = Net() #.cuda()
    net.eval()
    with torch.no_grad():
        img = Variable(img)  #img.cuda(async=True)
        out = net(img.view(1,3,ix,iy))
        features = out.data.cpu().numpy().astype('float32')
    return features






import matplotlib.pyplot as plt
import skimage
import torch
import model
log = torch.load('50epoch.pth',map_location=torch.device('cpu'))
with open(config.vocabulary_path, 'r') as fd:
    vocab_json = json.load(fd)
def demo(img_path,question):
    '''
    Main demo function. Takes input image path and Question string as input
    Returns top 5 answers, shows input image and visualizes attention applied 
    '''
    print('The Question asked was: ',question)
    cudnn.benchmark = True
    # Load pre-trained image
    # Download from https://github.com/snagiri/ECE285_Jarvis_ProjectA/releases/download/v1.0/50epoch.pth
    # log = torch.load('50epoch.pth',map_location=torch.device('cpu'))

    # print('Log....') 
    tokens = len(log['vocab']['question']) + 1
    net = model.Net(tokens)
    # print('Now loading')
    net.load_state_dict(log['weights'])
    # print('Loading done.....now net eval')
    net.eval()
    # print('net eval done')
    questions = list(prepare_questions(question))
    questions = [encode_question(q) for q in questions]
    q,q_len = questions[0]
    q = q.unsqueeze(0)

    v = encode_img(net,img_path)
    v = torch.from_numpy(v).to(torch.float)
    q_len = torch.tensor([q_len])
    with torch.no_grad():
        v = Variable(v)
        q = Variable(q)
        q_len = Variable(q_len)  
    
    out,att_out = net.forward(v,q,q_len)
    out = out.data.cpu()
    _, answer5 = torch.topk(out,5)
    answers = []
    #with open(config.vocabulary_path, 'r') as fd:
       # vocab_json = json.load(fd)
    a_to_i = vocab_json['answer']
    for answer in answer5:
        answer = (answer.view(-1))
        for a in answer.data:
            answers.append(list(a_to_i.keys())[a.data])        
    print_answers(answers)
    #print('demo done')
    #visualize_attentn(att_out,img_path)
    return answers

def visualize_attentn(att_out,img_path):
    '''
    Takes output of attention layer and overlays on input image. Then shows both 
    '''
    att_out = att_out.view(-1,14,14)
    num_im = att_out.size()[0]
    im = Image.open(img_path)
    fig2,ax2 = plt.subplots(1)
    ax2.imshow(im)
    ax2.set_title('Original Image')
    ax2.axis('off')
    fig,axs = plt.subplots(1,num_im,figsize=(10,10))
    axs = axs.ravel()
    for i in range(0,num_im):
        a1 = att_out[i].cpu().detach()
        a1 = a1.numpy()
        a1 = skimage.transform.pyramid_expand(a1, upscale=64,multichannel=False)
        im = im.resize(a1.shape)
        axs[i].imshow(im)
        axs[i].imshow(a1,cmap='gray',alpha=0.65)
        axs[i].set_title('Attention image '+str(i))
    for ax in axs:
        ax.axis('off')
    return

def print_answers(answers):
    '''
    Function to print top 5 answers
    '''
    for i,a in enumerate(answers):
        print("The top ",i+1," answer is ",a)
    return


op = 'c'
while(op == 'c'):
# print( " for Hindi press 1 \n for English press 2")
    txt2speech("for Hindi press 1 and for English press 2")

#Choose Language 
    x = input()
    if x == '1':
        language = 'hi-In'
    else:
        language = 'en-UK'
    print(language)

#Capture Image
    take_Photo()

    print("Ask question...")
    user_question = get_Question(language)
    answers = demo('image.jpg',user_question)
#txt2speech("answers can be " +answers[0]+','+answers[1]+','+answers[2]+','+answers[3]+','+answers[4])
    ans = "for this question answers can be " +answers[0]+','+answers[1]+','+answers[2]+','+answers[3]+','+answers[4]
    if(language == 'hi-In' or language =='hi'):
  # language='hi'
  # trans = Translator(to_lang = 'hindi') 
  #blob = TextBlob(ans)
      print(ans)
  # ans = trans.translate(ans)  
    else:
        language='en'
    txt2speech(ans)
    print('Press q to quit and c to continue(q/c): ', end = '')
    op = input()
#answers = demo('saved_img.jpg','What is the girl wearing?')
#answers = demo('saved_img.jpg','What is the girl doing?')

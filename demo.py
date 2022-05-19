import sys
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

from speech2txt import get_Question
from imagecap import take_Photo 
# from txt2speech import txt2speech


from gtts import gTTS 
from playsound import playsound 
def txt2speech(text): 
    var = gTTS(text = text,lang = 'en') 
    var.save('eng.mp3')
    os.system("start eng.mp3")
    # playsound("./eng.mp3")
    # close("./eng.mp3")
    return

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

def demo(img_path,question):
    '''
    Main demo function. Takes input image path and Question string as input
    Returns top 5 answers, shows input image and visualizes attention applied 
    '''
    print('The Question asked was: ',question)
    cudnn.benchmark = True
    # Load pre-trained image
    # Download from https://github.com/snagiri/ECE285_Jarvis_ProjectA/releases/download/v1.0/50epoch.pth
    log = torch.load('50epoch.pth',map_location=torch.device('cpu'))
    tokens = len(log['vocab']['question']) + 1
    net = model.Net(tokens)
    net.load_state_dict(log['weights'])
    net.eval()
    
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
    with open(config.vocabulary_path, 'r') as fd:
        vocab_json = json.load(fd)
    a_to_i = vocab_json['answer']
    for answer in answer5:
        answer = (answer.view(-1))
        for a in answer.data:
            answers.append(list(a_to_i.keys())[a.data])        
    print_answers(answers)
    #visualize_attentn(att_out,img_path)
    return

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
        print("The top ",i+1," answer is ",a, ".")
    ans =  answers[0]
    # print(f'.{ans}.')
    # txt2speech(ans)
    txt2speech(f'The answer is {ans}')
    
    return

take_Photo()
text = get_Question()
demo('saved_img.jpg',text)
# demo('test_img.jpg',text + '?')
# demo('tennis.jpg','Which game is she playing?')
# demo('dogs.jpg','What is in the image?')
# demo('dogs.jpg','How many dogs are there?')
#demo('saved_img.jpg','What is the girl wearing?')
#demo('saved_img.jpg','What is the girl holding?')
#demo('saved_img.jpg','Is she wearing mask?')


"""
path = '50epoch.pth'
results = torch.load(path)

l =[37.16,46.96,55.07,58.12,59.76,60.65,60.94]
l = [i/100 for i in l]
x = [1000,3000,6000,12000,25000,50000,100000]
x = [i/1890 for i in x]

val_acc = torch.FloatTensor(results['tracker']['val_acc'])
val_acc = val_acc.mean(dim=1).numpy()

val_loss = torch.FloatTensor(results['tracker']['val_loss'])
val_loss = val_loss.mean(dim=1).numpy()

fig,ax = plt.subplots(1,2,figsize=(8,5),constrained_layout=True)
ax[0].plot(val_acc,'--^',label='ours')
ax[0].plot(x,l,marker='v',label='paper')
ax[1].plot(val_loss,'--*',color='green')
ax[0].set_title('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[1].set_title('Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[0].grid(which='both')
ax[1].grid(which='both')
"""


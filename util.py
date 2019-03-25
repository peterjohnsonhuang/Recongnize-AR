import os, time, math, random
from skimage import io
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset,DataLoader
from tensorboardX import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd
assert Variable and F and DataLoader and torchvision and random and misc and plt
from sklearn.cluster import MiniBatchKMeans
import pickle
import gc

class DataManager():
    def __init__(self, tensorboard_dir=None, character_file=None):
        self.character= Character(character_file)
        if tensorboard_dir== None: self.writer=None
        else: self.tb_setting(tensorboard_dir)
    def tb_setting(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for f in os.listdir(directory): 
            os.remove('{}/{}'.format(directory,f))
        self.writer = SummaryWriter(directory)

    def train_classifier(self, model, dataloader, epoch, optimizer, print_every= 2):
        start= time.time()
        model.train()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, y) in enumerate(dataloader):
            batch_index=b+1
            x, y= Variable(x).cuda(), Variable(y).squeeze(1).cuda()
            output= model(x)
            #print(len(output[0]),y)
            loss = criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            #print(y)
            #print(pred)
            correct = int(pred.eq(y.data).long().cpu().sum())
            batch_correct += correct/ len(x)
            total_correct += correct
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def val_classifier(self,model,dataloader, epoch, print_every= 2):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                batch_index=b+1
                x, y= Variable(x).cuda(), Variable(y).squeeze(1).cuda()
                output= model(x)
                loss = criterion(output,y)
                # loss
                batch_loss+= float(loss)
                total_loss+= float(loss)* len(x)
                # accu
                pred = output.data.argmax(1) # get the index of the max log-probability
                correct = int(pred.eq(y.data).long().cpu().sum())
                batch_correct += correct/ len(x)
                total_correct += correct
                if batch_index% print_every== 0:
                    print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                    batch_loss= 0
                    batch_correct= 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Val Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def test_classifier(self,model,dataloader, print_every= 2):
        start= time.time()
        model.eval()
        
        data_size= len(dataloader.dataset)
        result = []
        for b, x in enumerate(dataloader):
            with torch.no_grad():
                batch_index=b+1
                x = Variable(x).cuda()
                output= model(x)
                #pred = output.data.argmax(1)
                #print(np.array(output.data[0]))
                output = output.cpu()
                pred = np.array(output.data[0]) # get the index of the max log-probability
                result.append(pred)
                
        #result = torch.cat(result, 0)
        return result
    
    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    def save(self, model, path, cluster= 256):
        model = model.cpu().eval()
        state_dict = model.state_dict()
        weight = {}
        #kmeans= KMeans(n_clusters=cluster, random_state=0)
        kmeans= MiniBatchKMeans(n_clusters=cluster, random_state=0, n_jobs=4)

        for key in state_dict:
            print(key)
            print(state_dict[key].numel())
            if state_dict[key].numel()<= cluster:
            #layer = key.split('.')
            #if state_dict[key].numel()<= cluster:
                weight[key] = state_dict[key].numpy()
            else:
                size = state_dict[key].size()
                params = state_dict[key].view(-1,1).numpy()
                kmeans.fit(params)
                quantized_table = kmeans.cluster_centers_.reshape((-1,))
                quantized_weight = kmeans.labels_.reshape(size).astype(np.uint8)
                weight[key] = (quantized_table, quantized_weight)

        with open(path, 'wb') as f:
                pickle.dump(weight, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self, path, model):
        with open(path, 'rb') as f:
            weight= pickle.load(f)
        state_dict = {}
        for key in weight:
            #print(key)
            if isinstance(weight[key], np.ndarray):
                #print(weight[key].shape)
                state_dict[key] = torch.from_numpy(weight[key])
            else:
                quantized_table = weight[key][0]
                quantized_weight = weight[key][1]
                #print(quantized_weight.shape)
                state_dict[key] = torch.from_numpy(quantized_table[quantized_weight.reshape((-1))].reshape((quantized_weight.shape)))
        model.load_state_dict(state_dict)
        return model

class CNN_squeezenet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, initial='kaiming_normal'):
        super(CNN_squeezenet, self).__init__()
        self.features = models.squeezenet1_1(pretrained=pretrained).features
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            self.final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=1)
        )
        if initial=='xavier_normal':
            nn.init.xavier_normal_(self.final_conv.weight.data)
        elif initial=='xavier_uniform':
            nn.init.xavier_uniform_(self.final_conv.weight.data)
        elif initial=='kaiming_normal':
            nn.init.kaiming_normal_(self.final_conv.weight.data)
        elif initial=='kaiming_uniform':
            nn.init.kaiming_uniform_(self.final_conv.weight.data)
        #self._initialize_weights_densenet()
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x
    def save(self, path):
        torch.save(self,path)

class transfer_CNN_squeezenet(nn.Module):
    def __init__(self, pretrained=True, num_classes=0, initial='kaiming_normal',model=models.squeezenet1_1(pretrained=True)):
        super(transfer_CNN_squeezenet, self).__init__()
        self.features = model.features
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            self.final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(15, stride=1)
        )
        if initial=='xavier_normal':
            nn.init.xavier_normal_(self.final_conv.weight.data)
        elif initial=='xavier_uniform':
            nn.init.xavier_uniform_(self.final_conv.weight.data)
        elif initial=='kaiming_normal':
            nn.init.kaiming_normal_(self.final_conv.weight.data)
        elif initial=='kaiming_uniform':
            nn.init.kaiming_uniform_(self.final_conv.weight.data)
        #self._initialize_weights_densenet()
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x
    def save(self, path):
        torch.save(self,path)



class Character:
    def __init__(self, character_file= None):
        self.character2index= {}
        self.index2character= {}
        self.n_character = 0
        if character_file != None:
            self.load(character_file)
    def addCharacter(self, character):
        character=int(character)
        if character not in self.character2index:
            self.character2index[character] = self.n_character
            self.index2character[self.n_character] = character
            self.n_character += 1
        return self.character2index[character]
    def save(self, path):
        index_list= sorted( self.character2index, key= self.character2index.get)
        with open( path, 'w') as f:
            f.write('\n'.join([str(i) for i in index_list]))
    def load(self, path):
        self.character2index= {}
        self.index2character= {}
        self.color2index= {}
        self.color2count= {}
        self.index2color= {}
        with open(path,'r') as f:
            for line in f:
                character=line.replace('\n','')
                self.addCharacter(character)


def train_batch_generator(train_dir = './data/train/', num_files = 1):
    files = sorted(os.listdir(train_dir))
    train_files=[]
    train_labels=[]
    for file in files:
        if 'trainx' in file:
            train_files.append(file)
        elif 'trainy' in file:
            train_labels.append(file)


    count = num_files
    embeddings, labels = [], []
    for file in train_files:
        print('Reading file {}...........'.format(file))
        gc.collect()
        embeddings = np.load(os.path.join(train_dir,file))
        for label_file in train_labels:

            if (label_file.split('.')[0]).split('_')[1] == (file.split('.')[0]).split('_')[1]:
                print(label_file)
                labels = np.load(os.path.join(train_dir,label_file))
        count -= 1
        if count == 0: 
            X_train, Y_train = embeddings,labels
            
            count = num_files
            embeddings, labels = [], []
            gc.collect()
            yield (X_train, Y_train)

def val_batch_generator(val_dir = './data/val/', num_files = 1):
    files = sorted(os.listdir(val_dir))
    val_files=[]
    val_labels=[]
    for file in files:
        if 'valx' in file:
            val_files.append(file)
        elif 'valy' in file:
            val_labels.append(file)


    count = num_files
    embeddings, labels = [], []
    for file in val_files:
        print('Reading file {}...........'.format(file))
        gc.collect()
        embeddings = np.load(os.path.join(val_dir,file))
        for label_file in val_labels:
            if (label_file.split('.')[0]).split('_')[1] == (file.split('.')[0]).split('_')[1]:
                print(label_file)
                labels = np.load(os.path.join(val_dir,label_file))
        count -= 1
        if count == 0: 
            X_val, Y_val = embeddings,labels
            count = num_files
            embeddings, labels = [], []
            gc.collect()
            yield (X_val, Y_val)


import numpy as np 
import os
import sys
from skimage import io
import skimage.color
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import models, transforms
import torch
import random
from PIL import Image
import json
import gc

class Data_manager: #change raw data into preproccesed training set
    def __init__(self, input_path, output_data, output_label,name):
        self.input_path = input_path
        self.output_data = output_data
        self.output_label = output_label
        self.name=name
    def get_label_name(self): #get label's name from its directory name

        return sorted(list(os.listdir(self.input_path)))
    def preprocess(self,img): #convert every image into rgb, 512*512 image
        if len(img.shape)==2:
            img = skimage.color.grey2rgb(img)
        shape=[sorted(img.shape)[-1],sorted(img.shape)[-2]]
        img = transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(max(shape),pad_if_needed=True),transforms.Resize(128) ])(img)
        img = np.array(img)
        return img
    def read_file(self,label_names=None): #list out and store images and labels

        if os.path.isfile(self.output_data) and os.path.isfile(self.output_label): #check if file existed
            return np.load(self.output_data), np.load(self.output_label)

        #generate a dictionary of labels 
        label_name = os.listdir(self.input_path)
        label_name = sorted(label_name)
        label_name_dict = dict([(i,label_names[i]) for i in range(len(label_names))])
        dict_json = json.dumps(label_name_dict)
        f = open("./dict.json","w")
        f.write(dict_json)
        f.close()
        
          
        datas = []
        labels = []
        one_hot_labels = []
        label_format = np.zeros(12)

        for i in range(len(label_name)): #match image with labels
            
            dir = os.path.join(self.input_path,label_name[i])
            if label_name[i] in label_name_dict.values():
                    
                for j in range(len(label_name_dict.keys())):
                    if label_name[i] == label_name_dict[j]:
                        
                        id = j
                for img in os.listdir(os.path.join(self.input_path,label_name_dict[id])) :
                    img=io.imread(os.path.join(dir,img))
                    img = self.preprocess(img)
                    
                    datas.append(img)
                    labels.append(id)
                    label_format = np.zeros(len(label_names))
                    label_format[id] = 1
                    one_hot_labels.append(label_format)

        datas = np.array(datas)
        labels = np.array(labels)
        one_hot_labels = np.array(one_hot_labels) #if need one_hot vector, then return this one as out_label
        #np.save(self.output_data+'_bg0', datas)
        #np.save(self.output_label+'_bg0', labels)
        return datas, labels
    def generate_data(self,in_image,bg_image): #generate training data using data augmentation method
        rand_num1=random.uniform(0, 1)
        rand_num2=random.uniform(0, 1)
        rand_num3=random.uniform(0, 1)
        if rand_num1>0.5:
            in_image=np.array(np.absolute(skimage.util.random_noise(in_image, mode='salt', seed=29, clip=True)*255),dtype='uint8')
        #print(in_image)
        if rand_num2>0.5:
            in_image=np.array(np.absolute(skimage.util.random_noise(in_image, mode='pepper', seed=29, clip=True)*255),dtype='uint8')
        in_image=(transforms.ToPILImage()(in_image)).convert('RGBA')
        out_image=transforms.Compose([
            
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(90, translate=(0.2,0.2), scale=(0.5,1.1), shear=15, resample=False, fillcolor=None),
            #transforms.Resize(128)
            
            ])(in_image)
        #print(out_image)
        #insert background image to avoid totally black background
        bg_image=(transforms.ToPILImage()(bg_image)).convert('RGB')
        bg_image=transforms.RandomCrop(128,pad_if_needed=True)(bg_image)
        bg_image=transforms.Resize(128,128)(bg_image)
        #print(bg_image)
        #merge background and object
        if self.name=='val' or self.name=='train':
            if rand_num3 > 0.5:
                bg_image.paste(out_image,out_image)
                out_image = bg_image
                #print('bg')
            else:
                out_image = (out_image).convert('RGB')
                #print('no')
        else:
            out_image = (out_image).convert('RGB')
            #print('no')
        out_image = transforms.Compose([
            #transforms.RandomCrop(128,pad_if_needed=True),
            #transforms.RandomGrayscale(p=1),
            transforms.ToTensor()
            ])(out_image)
        #out_image=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(out_image)
        return out_image
    def save_gen(self, datas, labels,gen_num,bg_path=None): #visualize training data and save
        img_gen=datas.tolist()
        label_gen=labels.tolist()
        bg_name=os.listdir(bg_path)
        for k in bg_name:
            bg_image=io.imread(os.path.join(bg_path,k))
            for i in range(len(datas)):
                for j in range(gen_num):
                    img=np.array(255*self.generate_data(datas[i],bg_image),dtype='uint8')
                    img=np.transpose(img,(1,2,0))
                    
                    #io.imsave('./image/{}_1/{}_{}_{}.jpg'.format(self.name, k.split('.')[0],labels[i],j),img)
                    
                    img_gen.append(img)


                    label_gen.append(labels[i])
                    print('\rreading image from {}...{}/{}'.format(k,len(label_gen),len(datas)*(gen_num)),end='')

        img_gen=np.array(img_gen)
        label_gen=np.array(label_gen)
        np.save(self.output_data+'.npy', img_gen)
        np.save(self.output_label+'.npy', label_gen)
            #img_gen=[]
            #label_gen=[]
        return None
    def add_new(self, datas, labels,gen_num,bg_path=None):
        img_gen=[]
        label_gen=[]
        bg_name=os.listdir(bg_path)
        for k in bg_name:
            bg_image=io.imread(os.path.join(bg_path,k))
            now_datas = np.load(self.output_data+'_'+k.split('.')[0]+'.npy')
            now_labels = np.load(self.output_label+'_'+k.split('.')[0]+'.npy')
            for i in range(len(datas)):
                
                for j in range(gen_num):
                    img=np.array(255*self.generate_data(datas[i],bg_image),dtype='uint8')
                    img=np.transpose(img,(1,2,0))
                    
                    #io.imsave('./image/{}_1/{}_{}_{}.jpg'.format(self.name, k.split('.')[0],labels[i],j),img)
                    
                    img_gen.append(img)


                    label_gen.append(labels[i])
                    print('\rreading image from {}...{}/{}'.format(k,len(label_gen)+len(now_labels),len(datas)*gen_num+len(now_labels)),end='')
            out_img = np.zeros((len(label_gen)+len(now_labels),128,128,3),dtype='uint8')
            out_img[:len(label_gen)] = img_gen
            out_img[len(label_gen):] = now_datas
        
            print('data append')
            label_gen=np.concatenate((label_gen,now_labels))
            print('label append')
            out_img=np.array(out_img,dtype='uint8')
            label_gen=np.array(label_gen)


            np.save(self.output_data+'_'+k.split('.')[0]+'.npy', out_img)
            np.save(self.output_label+'_'+k.split('.')[0]+'.npy', label_gen)
            img_gen=[]
            label_gen=[]
        return None
    
    def read_new(self,label_dict,new_labels,label_num):
        out_img=[]
        out_label=[]
        for i,label_name in enumerate(new_labels):
            img_path = os.listdir(self.input_path+'/'+label_name)
            in_img = io.imread(os.path.join((self.input_path+'/'+label_name),img_path[0]))
            in_img = self.preprocess(in_img)
            in_label = label_num+i
            #print(in_label)
            out_img.append(in_img)
            out_label.append(in_label)
            label_dict[in_label]=label_name
        #print(label_dict)
        dict_json = json.dumps(label_dict)
        f = open("./dict.json","w")
        f.write(dict_json)
        f.close()
        out_img=np.array(out_img)
        out_label=np.array(out_label)
        now_datas = np.load(self.output_data+'_bg0.npy')
        now_labels = np.load(self.output_label+'_bg0.npy')
        datas = np.zeros((len(out_label)+len(now_labels),128,128,3),dtype='uint8')
        datas[:len(out_label)] = out_img
        datas[len(out_label):] = now_datas
        labels=np.concatenate((out_label,now_labels))
        np.save(self.output_data+'_bg0', datas)
        np.save(self.output_label+'_bg0', labels)
        return(out_img,out_label)

    


class EasyDataset(Dataset):
    def __init__(self, image, label= None, flip = False, rotate = False, angle = 5):
        self.image = image
        self.label = label

        self.flip_n= int(flip)+1
        self.rotate= rotate
        self.preprocess=transforms.Compose([
            transforms.ToPILImage()


            ])
        self.transform_rotate= transforms.Compose([transforms.RandomRotation(angle),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_norotate= transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.angle= angle
    def __getitem__(self, i):
        index= i// self.flip_n 
        flip = bool( i % self.flip_n )

        if flip == True: x= np.flip(self.image[index],1).copy()
        else: x= self.image[index]
        x=self.preprocess(x)
        if self.rotate: 
            x= self.transform_rotate(x)
        else: 
            x= self.transform_norotate(x)

        if self.label is not None:
            y=torch.LongTensor([self.label[index]])
            return x,y
        else :
            return x
    def __len__(self):
        return len(self.image)*self.flip_n
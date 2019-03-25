import torch
from torch.utils.data import DataLoader
from util import*
import argparse
import torch.nn as nn
from resnet import ResNet,resnet50,resnet152,resnet18
from data_process_split import Data_manager, EasyDataset
assert torch and DataLoader and CNN_squeezenet and EasyDataset
import os,sys
import numpy as np
import json
import time
import gc


if os.path.isdir('./runs')==False:
	os.mkdir('./runs')


TENSORBOARD_DIR= './runs/stl10'

dm= DataManager(tensorboard_dir= TENSORBOARD_DIR)

EPOCH = 200
BATCH_SIZE = 16
LEARNING_RATE = 1E-4
DROPOUT = 0.5
PRETRAIN = False
retrain=True
#OUTPUT_CHARACTER = 'data/character.txt'
train_dir='./train1/'
val_dir='./train1'
train_path=['./data/train/trainx','./data/train/trainy']
val_path=['./data/val/valx','./data/val/valy']
bg_path='./val_background'
val_bg_path='./val_background'

if os.path.isdir('./data')==False:
	os.mkdir('./data')
if os.path.isdir('./model')==False:
	os.mkdir('./model')
if os.path.isdir('./data/train')==False:
	os.mkdir('./data/train')
if os.path.isdir('./data/val')==False:
	os.mkdir('./data/val')

start_time=time.time()

train=Data_manager(train_dir, train_path[0], train_path[1] ,'train')
val=Data_manager(val_dir, val_path[0], val_path[1],'val')
if os.path.isfile('./dict.json'):
	with open('./dict.json') as json_data:
		label_dict = json.load(json_data)
		label_name_old=label_dict.values()
	label_num=len(label_name_old)



'''
check if already have processed training data and validation data 
'''
if os.path.isfile(train_path[0]+'_bg0.npy') and os.path.isfile(val_path[0]+'_bg1.npy'):
	print('continue!')
	label_name_now=train.get_label_name()
	new_labels=[]
	for label in label_name_now:
		if label not in label_name_old:
			new_labels.append(label)
	if new_labels!=[]:
		retrain=True
		new_datas=train.read_new(label_dict,new_labels,label_num)
		train.add_new(new_datas[0],new_datas[1],3,bg_path)
		val.add_new(new_datas[0],new_datas[1],2,val_bg_path)
else:
	print('new set!')
	train=Data_manager(train_dir, train_path[0], train_path[1] ,'train')
	val=Data_manager(val_dir, val_path[0], val_path[1],'val')
	label_name=train.get_label_name()
	data=train.read_file(label_names=label_name)
	train.save_gen(data[0],data[1],4,bg_path)
	val.save_gen(data[0],data[1],2,val_bg_path)

gc.collect()
cnt = len(os.listdir(train_dir))
initial=['xavier_normal','xavier_uniform','kaimimg_normal','kaimimg_uniform']

train_start_time=time.time()


'''
start traing with loaded model
'''
train_data = np.uint8(np.load('./data/train/trainx.npy'))
train_label = np.uint8(np.load('./data/train/trainy.npy'))
val_data = np.uint8(np.load('./data/val/valx.npy'))
val_label = np.uint8(np.load('./data/val/valy.npy'))

train_dataloader= DataLoader((EasyDataset(train_data,train_label,flip=False, rotate=False)),batch_size= BATCH_SIZE,shuffle= True, num_workers = 8)
val_dataloader= DataLoader(EasyDataset(val_data,val_label,flip=False, rotate=False),batch_size= BATCH_SIZE, shuffle= False, num_workers = 8)
#model = torch.load('./model/resnet18_3_97.05966724039013.pt')
for i in range(4):
	if os.path.isfile("./model/resnet18_85_{}.pt".format(initial[i])):
		model = torch.load("./model/resnet18_85_{}.pt".format(initial[i]))
		if retrain:
			model = resnet18(pretrained=True, num_classes=cnt, initial=initial[i],model=model).cuda()
	else:
		model = resnet18(pretrained=True, num_classes=cnt, initial=initial[i]).cuda()
	print('Model parameters: {}'.format(dm.count_parameters(model)))
	#print(train_data)
	optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
	acc=[]
	accu_record=80
	thresh=85
	for epoch in range(1,EPOCH+1):
		if epoch>=20:
			optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE/2)
		if epoch>=40:
			optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE/10)

			
		gc.collect()
		dm.train_classifier( model, train_dataloader, epoch, optimizer)
		print('='*80)
		record=dm.val_classifier( model, val_dataloader, epoch)
		acc.append(record[1])

		print(acc)
		acc=sum(acc)/len(acc)
		if acc> accu_record:
			model.save("./model/resnet18_85_{}.pt".format(initial[i]))
			print('Model saved!!!')
			accu_record=acc
			if accu_record>thresh :
				break
		acc=[]
		print('='*80)

end_time=time.time()

print('data process time:', train_start_time-start_time)
print('training time:',end_time-train_start_time)

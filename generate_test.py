import numpy as np 
import os
import shutil
import sys
from skimage import io
import skimage.color
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import models, transforms
import torch
import random
from PIL import Image



'''
this is just a small process to make each pivture into one category
'''
if os.path.isdir('./train1'):
	shutil.rmtree('./train1')
if os.path.isdir('./train1_img'):
	shutil.rmtree('./train1_img')


target_path='./all_img'
os.mkdir('./train1')
os.mkdir('./train1_img')
dirs=sorted(os.listdir(target_path))

cnt=0
'''
os.mkdir('./train1_img')
for root, dirs, files in os.walk(target_path):
	if cnt>=500:
		break
	for name in files:
		img=os.path.join(root,name)
		try:
			img=io.imread(img)
			if img.shape[0]>196 and img.shape[1]>196:

				io.imsave('./train1_img/{}'.format(name),img)
				cnt+=1
		except IOError:
			pass
		
		if cnt >= 500:
			break
print(cnt)

'''
for root, dirs, files in os.walk(target_path):
	if cnt>=300:
		break
	for name in files:
		img=os.path.join(root,name)
		try:
			img=io.imread(img)
			if img.shape[0]>196 and img.shape[1]>196:
				os.mkdir('./train1/{}'.format(name.split('.')[0]))
				io.imsave('./train1/{}/{}.jpg'.format(name.split('.')[0],name.split('.')[0]),img)
				io.imsave('./train1_img/{}.jpg'.format(name.split('.')[0]),img)
				cnt+=1
		except IOError:
			pass
		
		if cnt >= 300:
			break
print(cnt)

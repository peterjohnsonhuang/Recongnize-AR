import cv2
import torch
from torch.utils.data import DataLoader
from util import DataManager
import time
import threading

from data_process_split import Data_manager, EasyDataset
from torch.autograd import Variable
import os,sys
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import skimage
from skimage import io
import json

BATCH_SIZE = 32
found = 0
source_dir = './train1_img'

INPUT_MODEL1 = './model/resnet18_85_xavier_normal.pt'#huge_training xavier_normal
INPUT_MODEL2 = './model/resnet18_85_xavier_uniform.pt'#small_training kaiming_normal
INPUT_MODEL3 = './model/resnet18_85_kaimimg_normal.pt'#huge_training xavier_uniform
INPUT_MODEL4 = './model/resnet18_85_kaimimg_uniform.pt'#small_training kaiming_uniform


start_time=time.time()
dm=DataManager()
with open('./dict.json') as json_data:
	label_name=json.load(json_data)

model1= torch.load(INPUT_MODEL1).cuda()
model1.eval()
model2= torch.load(INPUT_MODEL2).cuda()
model2.eval()
model3= torch.load(INPUT_MODEL3).cuda()
model3.eval()
model4= torch.load(INPUT_MODEL4).cuda()
model4.eval()

load_time=time.time()
print('loading time:',load_time-start_time)
orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
name_num=len(label_name)
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out_vid = cv2.VideoWriter('out_vid/output1.mp4', fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))

def check_target(current,result):
	orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
	found = 0
	frame_cnt = 0
	while(1):
		start_time = time.time()
		#print('hola',frame_cnt)
		frame_cnt +=1
		start_time=time.time()
		# get a frame
		while True:
			if cv2.waitKey(1) :
				ret_val, frame = cap.read()
				cv2.imshow("Homography", frame)
				out_vid.write(frame)
				#cv2.imwrite("./test_1.jpg", frame)
				break
		grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		kp1, des1 = orb.detectAndCompute(grayframe, None)

		data = frame
		if len(data.shape)==2:
			data = skimage.color.grey2rgb(data)
		#print(data.shape)
		test=transforms.Compose([

            
			transforms.ToPILImage(),
			transforms.RandomCrop(max(data.shape),pad_if_needed=True),
    		transforms.Resize(128),
    		transforms.ToTensor()
    		])(data)
		#print(np.array(test).shape)

		img=np.transpose(np.array(test),(1,2,0))
	
		#io.imsave('./test.jpg',img)
		#cv2.imwrite("./test.jpg", img)
	
		test= DataLoader(EasyDataset([test]),batch_size= BATCH_SIZE, shuffle= False, num_workers = 0)
	
		out1 = dm.test_classifier(model=model1, dataloader=test)
		out2 = dm.test_classifier(model=model2, dataloader=test)
		out3 = 	dm.test_classifier(model=model3, dataloader=test)
		out4 = 	dm.test_classifier(model=model4, dataloader=test)
		out_one = out1[0][:name_num]+out2[0][:name_num]+out3[0][:name_num]+out4[0][:name_num]

		out = out_one.argsort()

		target_cand = []
		cand = [label_name[str(int(out[-1]))],label_name[str(int(out[-2]))],label_name[str(int(out[-3]))],label_name[str(int(out[-4]))],label_name[str(int(out[-5]))]]
		try:
			for name in cand:
				img = cv2.imread(os.path.join(source_dir,name)+'.jpg',0)
				kp, des = orb.detectAndCompute(img, None)
				matches = bf.knnMatch(des, des1, k=2)
				#print(len(matches))
				good_points = []
				for m, n in matches:
					if m.distance < 0.6*n.distance:
						good_points.append(m)
				if len(good_points) >= 50:
					found += 1
					target_cand.append(name)
					
		except IOError:
			pass

			


		end_time=time.time()
		#print('predict time:',end_time-start_time)
		if found == 1:
			target_out = target_cand[0]
			print(target_out)
			break
		else:
			found = 0
			target_out = current
		'''
		if frame_cnt == 1:
			break
		'''
		end_time = time.time()
		print('classify time:',end_time-start_time)
	print('done')
	result[0] = target_out
	
	return None




while(1):
	start_time=time.time()
	# get a frame
	while True:
		if cv2.waitKey(1) :
			ret_val, frame = cap.read()
			cv2.imshow("Homography", frame)
			cv2.imwrite("./test_1.jpg", frame)
			out_vid.write(frame)
			break
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	kp1, des1 = orb.detectAndCompute(grayframe, None)

	data = frame
	if len(data.shape)==2:
		data = skimage.color.grey2rgb(data)
	#print(data.shape)
	test=transforms.Compose([

            
		transforms.ToPILImage(),
		transforms.RandomCrop(max(data.shape),pad_if_needed=True),
    	transforms.Resize(128),
    	transforms.ToTensor()
    	])(data)
	#print(np.array(test).shape)

	img=np.transpose(np.array(test),(1,2,0))
	
	io.imsave('./test.jpg',img)
	#cv2.imwrite("./test.jpg", img)
	
	test= DataLoader(EasyDataset([test]),batch_size= BATCH_SIZE, shuffle= False, num_workers = 0)
	
	out1 = dm.test_classifier(model=model1, dataloader=test)
	out2 = dm.test_classifier(model=model2, dataloader=test)
	out3 = 	dm.test_classifier(model=model3, dataloader=test)
	out4 = 	dm.test_classifier(model=model4, dataloader=test)
	out_one = out1[0][:name_num]+out2[0][:name_num]+out3[0][:name_num]+out4[0][:name_num]

	out = out_one.argsort()

	target_cand = []
	cand = [label_name[str(int(out[-1]))],label_name[str(int(out[-2]))],label_name[str(int(out[-3]))],label_name[str(int(out[-4]))],label_name[str(int(out[-5]))]]
	try:
		for name in cand:
			img = cv2.imread(os.path.join(source_dir,name)+'.jpg',0)
			kp, des = orb.detectAndCompute(img, None)
			matches = bf.knnMatch(des, des1, k=2)
			#print(len(matches))
			good_points = []
			for m, n in matches:
				if m.distance < 0.6*n.distance:
					good_points.append(m)
			if len(good_points) >= 50:
				found += 1
				target_cand.append(name)
				print(name)
	except IOError:
		pass

			


	end_time=time.time()
	#print('predict time:',end_time-start_time)
	if found == 1:
		target = target_cand[0]
		break
	else:
		found = 0

# Features
img = cv2.imread("train1_img/{}.jpg".format(target), cv2.IMREAD_GRAYSCALE) # queryiamge
video_dir = "./videos"
video = []
for vid in os.listdir(video_dir):
	name = vid.split('.')[0]
	video.append(name)
if target in video:
	view_cap = cv2.VideoCapture(os.path.join(video_dir,'{}.m4v'.format(target)))
else:
	view_cap = cv2.VideoCapture("view.m4v")
orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
kp_image, desc_image = orb.detectAndCompute(img, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#img2 = cv2.cv2.drawKeypoints(img, kp_image, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('fast_true.png',img2)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
i = 0
dst = []
target_now = [target]
vanish_cnt = 0

while True:
	start_time = time.time()
	if target_now[0] != target:
		target = target_now[0]
		img = cv2.imread("train1_img/{}.jpg".format(target), cv2.IMREAD_GRAYSCALE)
		kp_image, desc_image = orb.detectAndCompute(img, None)
		if target in video:
			view_cap = cv2.VideoCapture(os.path.join(video_dir,'{}.m4v'.format(target)))
		else:
			view_cap = cv2.VideoCapture("videos/view.m4v")
	tic = time.time()
	#print ("i = ", i)
	i += 1
	_, frame = cap.read()
	print(frame.shape)
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # trainimage
    
    
	kp_grayframe, desc_grayframe = orb.detectAndCompute(grayframe, None)
	#img3 = cv2.drawKeypoints(grayframe, kp_grayframe, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	#cv2.imwrite('fast_video.png',img3)
    
	toc = time.time()
	#print('* Elapsed time1: %f sec.' % (toc - tic))

	matches = bf.knnMatch(desc_image, desc_grayframe, k=2)
	#matches = sorted(matches, key = lambda x:x.distance)

	good_points = []
	for m, n in matches:
		if m.distance < 0.6*n.distance:
			good_points.append(m)
	#img4 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
	#cv2.imwrite('matches.png',img4)

	# Homography
	if len(good_points) > 10:
		vanish_cnt = 0
		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
		train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
 
		matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
		matches_mask = mask.ravel().tolist()
		last_mastrix = matrix
 
		# Perspective transform
		h, w = img.shape
		pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
		#print(pts.shape)
		dst = cv2.perspectiveTransform(pts, matrix)
		#print(dst)
		_, img_to_paste = view_cap.read()
		#print(img_to_paste)
		if img_to_paste is None:
			if target in video:
				view_cap = cv2.VideoCapture(os.path.join(video_dir,'{}.m4v'.format(target)))
			else:
				view_cap = cv2.VideoCapture("videos/view.m4v")
			_, img_to_paste = view_cap.read()

		h1, w1 = img.shape
		h, w, _ = img_to_paste.shape
		if h1*w/w1>h:
			h2, w2 = int(h1*w/w1), w
			new_w = 0
			new_h = int((h2-h)/2)-1
		elif h*w1/h1 >w:
			h2, w2 = h, int(h*w1/h1)
			new_h = 0
			new_w = int((w2-w)/2)-1
		else:
			h2, w2 = h, w
			new_w, new_h = 0, 0
		img_to_paste_2 = np.zeros((h2,w2,3))
		img_to_paste = Image.fromarray(np.uint8(img_to_paste)).convert("RGBA")
		img_to_paste_2 = Image.fromarray(np.uint8(img_to_paste_2)).convert("RGBA")
		img_to_paste_2.paste(img_to_paste, (new_w,new_h), img_to_paste)
		img_to_paste = np.array(img_to_paste_2.convert('RGB'))

		h, w, _ = img_to_paste.shape



		pts_small = np.array([[0, 0], [0, h], [w, h], [w, 0]])
		dst = np.int32(dst).reshape((4,2))
		pts_big = np.array([[dst[0,0], dst[0,1]],[dst[1,0], dst[1,1]],[dst[2,0], dst[2,1]],[dst[3,0], dst[3,1]]])
		H, status = cv2.findHomography(pts_small, pts_big)
		homography = cv2.warpPerspective(img_to_paste, H, (frame.shape[1],frame.shape[0]))
		homography = cv2.polylines(homography, [np.int32(dst)], True, (255, 0, 0), 3)
        
		#paste together
		im = Image.fromarray(np.uint8(homography)).convert("RGBA")
		imArray = np.asarray(im)
		# create mask
		polygon = [(dst[0,0], dst[0,1]),(dst[1,0], dst[1,1]),(dst[2,0], dst[2,1]),(dst[3,0], dst[3,1])]
		maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
		ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
		mask = np.array(maskIm)
		# assemble new image (uint8: 0-255)
		newImArray = np.empty(imArray.shape,dtype='uint8')

		# colors (three first columns, RGB)
		newImArray[:,:,:3] = imArray[:,:,:3]

		# transparency (4th column)
		newImArray[:,:,3] = mask*255

		# back to Image from numpy
		newIm = Image.fromarray(newImArray, "RGBA")
		newIm_back = Image.fromarray(np.uint8(frame)).convert("RGBA")
		newIm_back.paste(newIm, (0, 0), newIm)
		homography = np.asarray(newIm_back)

		#homography = H_transform(img_to_paste, frame, pts_small, h)
		cv2.namedWindow('Homography', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Homography', 600,600)
		cv2.imshow("Homography", homography)
	else:
		vanish_cnt += 1
		if dst == []:
			homography = frame
			homography = Image.fromarray(np.uint8(homography)).convert("RGBA")
			homography = np.array(homography)
			cv2.namedWindow('Homography', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('Homography', 600,600)
			cv2.imshow("Homography", homography)
 
		elif vanish_cnt > 3:
			check_target(target,target_now)
			_, frame = cap.read()
			homography = frame
			homography = Image.fromarray(np.uint8(homography)).convert("RGBA")
			homography = np.array(homography)
			
		else:
			dst = cv2.perspectiveTransform(pts, matrix)
			h, w, _ = img_to_paste.shape
			pts_small = np.array([[0, 0], [0, h], [w, h], [w, 0]])
			dst = np.int32(dst).reshape((4,2))
			pts_big = np.array([[dst[0,0], dst[0,1]],[dst[1,0], dst[1,1]],[dst[2,0], dst[2,1]],[dst[3,0], dst[3,1]]])
			H, status = cv2.findHomography(pts_small, pts_big)
			homography = cv2.warpPerspective(img_to_paste, H, (frame.shape[1],frame.shape[0]))
			homography = cv2.polylines(homography, [np.int32(dst)], True, (255, 0, 0), 3)
			#paste together
			im = Image.fromarray(np.uint8(homography)).convert("RGBA")
			imArray = np.asarray(im)
			# create mask
			polygon = [(dst[0,0], dst[0,1]),(dst[1,0], dst[1,1]),(dst[2,0], dst[2,1]),(dst[3,0], dst[3,1])]
			maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
			ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
			mask = np.array(maskIm)
			# assemble new image (uint8: 0-255)
			newImArray = np.empty(imArray.shape,dtype='uint8')
			# colors (three first columns, RGB)
			newImArray[:,:,:3] = imArray[:,:,:3]
			# transparency (4th column)
			newImArray[:,:,3] = mask*255
			# back to Image from numpy
			newIm = Image.fromarray(newImArray, "RGBA")
			newIm_back = Image.fromarray(np.uint8(frame)).convert("RGBA")
			newIm_back.paste(newIm, (0, 0), newIm)
			homography = np.asarray(newIm_back)
			cv2.namedWindow('Homography', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('Homography', 600,600)
			cv2.imshow("Homography", homography)
	#cv2.imshow("Image", img)
	#cv2.imshow("grayFrame", grayframe)
	#cv2.imshow("img3", img3)
	print(homography.shape)
	out_frame = np.array(Image.fromarray(homography, "RGBA").convert('RGB'))
	out_vid.write(out_frame)
	end_time = time.time()
	
	print('* Elapsed time2: %f sec.' % (end_time-start_time))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break




out_vid.release()   
cap.release()
cv2.destroyAllWindows()
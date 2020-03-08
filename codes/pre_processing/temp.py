import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import random
from utils import maintain_list

IMG_SIZE = 32
TOTAL_IMG = 100
global count
count = 0

def process_volume(image_volume,label_volume,save_path):
	# count = 0
	for i in range(256):
		temp(count)
		# generate_random_data(current_image,current_label,count,save_path)
		print(count)
		

		# current_image = image_volume[:,:,i]
		# current_label = label_volume[:,:,i]
		
		# if(np.sum(current_image) >0):
		# 	print("*"*50)
		# 	generate_random_data(current_image,current_label,count,save_path)
		# 	# break
		

def temp(count):
	count +=100

def generate_random_data(image,label,count,save_path):

	count +=100
	

	# x_list= set(random.sample(range(0, 128), 100))
	# y_list= set(random.sample(range(0, 256), 100))
	
	# while(bool(x_list)):

	# 	x = x_list.pop()
	# 	y = y_list.pop()

	
	# 	current_image_patch = image[x:x+IMG_SIZE,y:y+IMG_SIZE]
	# 	current_label_patch = label[x:x+IMG_SIZE,y:y+IMG_SIZE]
		
	# 	if(np.sum(current_image_patch)>0):
	# 		count +=1
	# 		save_data(current_image_patch,current_label_patch,count,save_path)
		
			
	# 	else:
	# 		x_list,y_list = maintain_list(x_list,y_list)

	# print(count)
			

def save_data(_current_image_patch,current_label_patch,count,save_path):
	
	name = "{}.png".format(str(count).zfill(8))

	image_save_path = save_path + "/images/" + name
	label_save_path = save_path + "/labels/" + name

	# cv2.imwrite(image_save_path,current_image_patch)
	# cv2.imwrite(label_save_path,current_label_patch)



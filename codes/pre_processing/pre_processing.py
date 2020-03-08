import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import random
import cv2
from utils import maintain_list

IMG_SIZE = 64
TOTAL_IMG = 24
count = 0



def process_volume(image_volume,label_volume,save_path,volume):
	count = 0
	for i in range(200):

		current_image = image_volume[:,:,i]
		current_label = label_volume[:,:,i]
		
				
		if(np.sum(current_image) >9000):
			
			generate_random_data(current_image,current_label,save_path,volume)
			
		

def generate_multichannle_label(label):	
	# print(np.unique(label))
	new_label = []
	for i in range(1,4):
		mask = np.zeros((IMG_SIZE,IMG_SIZE)) 
		mask[label == i] = 1

		new_label.append(mask)

	new_label = np.array(new_label)

	return new_label


# def generate_multichannle_label(label):
	

# 	mask = np.zeros((IMG_SIZE,IMG_SIZE)) 
# 	mask[label == 2] = 1

# 	return mask


def generate_random_data(image,label,save_path,volume):

	x_list= set(random.sample(range(0, 128-IMG_SIZE), TOTAL_IMG))
	y_list= set(random.sample(range(0, 256-IMG_SIZE), TOTAL_IMG))
	
	while(bool(x_list)):

		x = x_list.pop()
		y = y_list.pop()

	
		current_image_patch = image[y:y+IMG_SIZE,x:x+IMG_SIZE]
		current_label_patch = label[y:y+IMG_SIZE,x:x+IMG_SIZE]
		

		
		
		if(np.sum(current_image_patch)>400 and current_image_patch.shape[0] == 64 and current_image_patch.shape[1] == 64):
			
			current_label_patch = generate_multichannle_label(current_label_patch)
			save_data(current_image_patch,current_label_patch,save_path,volume)
		
			
		else:
			x_list,y_list = maintain_list(x_list,y_list)

	print(count)
			

def save_data(current_image_patch,current_label_patch,save_path,volume):
	global count
	
	
	image_name = "{}.png".format(str(count).zfill(8))
	label_name = "{}.npy".format(str(count).zfill(8))
	
	# print(image_name)

	image_save_path = save_path +"/images/"+ volume + "/" + image_name
	label_save_path = save_path +"/labels/"+ volume + "/" + label_name
	
	

	cv2.imwrite(image_save_path,current_image_patch)
	np.save(label_save_path,current_label_patch)
	count +=1
	



import os
import random

def create_dir(path,volume):
	if not os.path.exists(path):
		os.makedirs(path)
		os.makedirs(path+"/images/"+volume)
		os.makedirs(path+"/labels/"+volume)
		


	if not os.path.exists(path+"/images/"+volume):
		os.makedirs(path+"/images/"+volume)
		os.makedirs(path+"/labels/"+volume)
	



def maintain_list(x_list,y_list):
	while(True):
		new_x = random.sample(range(0, 128), 1)[0]
		new_y = random.sample(range(0, 256), 1)[0]

		if(new_x not in x_list and new_y not in y_list):
			x_list.add(new_x)
			y_list.add(new_y)
			break

	return x_list,y_list
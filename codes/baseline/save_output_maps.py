from data_loader import trainDataLoaderFn, testDataLoaderFn
from trainer import Trainer
from utils import prepareDirs,prepareData
import config
import numpy as np
import cv2


def main(config):
	
	trainLoader = trainDataLoaderFn(config.train_images_path,config.train_labels_path,config.train_csv_path,config.batch_size)
	validLoader = testDataLoaderFn(config.validation_images_path,config.validation_labels_path,config.validation_csv_path,config.batch_size)
	testLoader  = testDataLoaderFn(config.test_images_path,config.test_labels_path,config.test_csv_path,config.batch_size)

	# temp = iter(trainLoader)
	# temp.next()
	
	
	
	model=Trainer(config,trainLoader,validLoader)
	
	input_images,output_maps = model.test(testLoader)
	
	
	input_images = prepareData(input_images)
	output_maps = prepareData(output_maps)
	
	print(input_images.shape)
	print(output_maps.shape)


	for i in range(len(input_images)):

		current_input_image = input_images[i,:,:,:]
		current_input_image = np.transpose(current_input_image,(1,2,0))
		
		current_input_image = 255.0 * current_input_image
		current_input_image = current_input_image.astype("uint8")

		current_output_map = output_maps[i,:,:,:]
		current_output_map = np.transpose(current_output_map,(1,2,0))
		current_output_map = 255.0 * current_output_map
		current_output_map = current_output_map.astype("uint8")

		print(current_input_image.shape)
		print(current_output_map.shape)

		rgb_map = cv2.cvtColor(current_output_map,cv2.COLOR_GRAY2RGB)
		output_image = cv2.addWeighted(current_input_image, 0.5, rgb_map,0.5, 0.0)

		cv2.imwrite('outputs/saved_images/input_images/'+str(i)+".png",current_input_image)
		cv2.imwrite('outputs/saved_images/output_maps/'+str(i)+".png",current_output_map)
		cv2.imwrite('outputs/saved_images/output_images/'+str(i)+".png",output_image)
		





	del(model)


if __name__ == "__main__":

	resumeFlag = True
	best_model = False
	configObject = config.configClass(resumeFlag,best_model)

	main(configObject)



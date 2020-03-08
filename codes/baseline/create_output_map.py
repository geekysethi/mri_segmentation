from data_loader import trainDataLoaderFn, testDataLoaderFn
from trainer import Trainer
from utils import prepareDirs, prepareData 
import config

import numpy as np


def main(config):
	
	prepareDirs(config)
	print('==> Preparing data...')
	trainLoader = trainDataLoaderFn(config.train_path,config.train_csv_path,config.batch_size)
	validLoader = testDataLoaderFn(config.validation_path,config.validation_csv_path,config.batch_size)
	testLoader  = testDataLoaderFn(config.test_path,config.test_csv_path,config.batch_size)

	# it = iter(trainLoader)
	# next(it)
	
	
	model=Trainer(config,trainLoader,validLoader)
	output_map = model.test(testLoader)
	# output_map
	final_output_map = prepareData(output_map)
	np.save("output_maps.npy",final_output_map)
	print(final_output_map.shape)
	# print(len(output_map))
	
	del(model)


if __name__ == "__main__":

	resumeFlag = True
	best_model = True
	extension = "T1"
	configObject = config.configClass(resumeFlag,best_model,extension)

	main(configObject)



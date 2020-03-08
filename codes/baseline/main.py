from data_loader import trainDataLoaderFn, testDataLoaderFn
# from trainer import Trainer
# from utils import prepareDirs
import config



def main(config):
	
	# prepareDirs(config)
	# print('==> Preparing data...')
	trainLoader = trainDataLoaderFn(config.train_images_path,config.train_labels_path,config.train_csv_path,config.batch_size)
	validLoader = testDataLoaderFn(config.validation_images_path,config.validation_labels_path,config.validation_csv_path,config.batch_size)
	# testLoader  = testDataLoaderFn(config.test_images_path,config.test_labels_path,config.test_csv_path,config.batch_size)
	
	temp = iter(trainLoader)
	temp2 = temp.next()
	print(temp2[0].size(),temp2[1].size())
	
	
	# model=Trainer(config,trainLoader,validLoader)
	# model.train()
	# model.test(testLoader)
	# del(model)


if __name__ == "__main__":

	resumeFlag = False
	best_model = False
	configObject = config.configClass(resumeFlag,best_model)

	main(configObject)



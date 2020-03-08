import os
from pathlib import Path


class configClass():
	def __init__(self,resume,best_model):

		self.output_dir = Path("./outputs")
		self.bestModel = best_model

		self.save_model_dir = "./outputs/saved_models" 
		self.tensorboard_path = "./outputs/runs" 


		self.train_images_path = "../../data/training/train/images"
		self.train_labels_path = "../../data/training/train/labels"
		self.train_csv_path =  	"../../data/training/train/train.csv"


		self.validation_images_path = "../../data/training/validation/images"
		self.validation_labels_path = "../../data/training/validation/labels"
		self.validation_csv_path =  	"../../data/training/validation/validation.csv"

		# self.test_images_path = "../../data/test/images"
		# self.test_labels_path = "../../data/test/labels"
		# self.test_csv_path =  	"../../data/test/data.csv"


		self.epochs=200
		self.batch_size = 128     # of images in each batch of data
		self.num_workers = 2       # of subprocesses to use for data loading
		self.lr=0.01
		self.momentum=0.9
		self.weight_decay=5e-4

		self.train_paitence = 20
		
		
		self.resume=resume

		self.use_gpu=True

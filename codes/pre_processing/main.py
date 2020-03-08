from pre_processing import process_volume
from utils import create_dir
import scipy.io
import glob


if __name__ == "__main__":

	save_path = "../../data/training/train/"

	# volume = "vol_01"
	# create_dir(save_path,volume)
	# image_volume = scipy.io.loadmat("../../data/raw_data/training/Vol_01_input.mat")
	# label_volume = scipy.io.loadmat("../../data/raw_data/training/Vol_01gt.mat")

	# volume = "vol_02"
	# create_dir(save_path,volume)
	# image_volume = scipy.io.loadmat("../../data/raw_data/training/Vol_02_input.mat")
	# label_volume = scipy.io.loadmat("../../data/raw_data/training/Vol_02gt.mat")
	
	# volume = "vol_05"
	# create_dir(save_path,volume)
	# image_volume = scipy.io.loadmat("../../data/raw_data/training/Vol_05_input.mat")
	# label_volume = scipy.io.loadmat("../../data/raw_data/training/Vol_05gt.mat")
	
	save_path = "../../data/training/validation/"
	volume = "vol_06"
	create_dir(save_path,volume)
	image_volume = scipy.io.loadmat("../../data/raw_data/validation/Vol_06_input.mat")
	label_volume = scipy.io.loadmat("../../data/raw_data/validation/Vol_06gt.mat")
	
	image_volume = image_volume["ana"]
	label_volume = label_volume["gt"]
	
	

	process_volume(image_volume,label_volume,save_path,volume)

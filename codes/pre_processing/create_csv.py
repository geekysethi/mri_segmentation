import numpy as np
import pandas as pd 
import os

# save_path = "../../data/training/train/"
# volumes = ["vol_01","vol_02","vol_05"]


save_path = "../../data/training/validation/"
volumes = ["vol_06"]
df_rows = []

for current_vol in volumes:
	print("*"*40)
	dir = save_path+"/images/"+current_vol+ "/"
	onlyfiles = next(os.walk(dir))[2]
	total_len = len(onlyfiles)

	print(total_len)

	for i in range(total_len):

		image_name = str("{}.png".format(str(i).zfill(8)))
		label_name = str("{}.npy".format(str(i).zfill(8)))
		
		df_rows.append([image_name,label_name,current_vol])



columns = ['image_id',"label_id", 'volume']


print("[INFO] SAVING DATA IN DATAFRAME")
df = pd.DataFrame(index=np.arange(len(df_rows)), columns=columns)
df.loc[:] = df_rows


# df.to_csv(save_path+"/"+"train.csv", encoding="utf-8", index=False)
# df.to_csv(save_path+"/"+"validation.csv", encoding="utf-8", index=False)
import os
import argparse
import random
from tqdm import tqdm
import numpy as np

import torch

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-dir",type=str,default="./data")
	parser.add_argument("--result-dir",type=str,default="./result")
	parser.add_argument("--dataset",type=str,default="Florence_3d_actions")
	args = parser.parse_args()
	return args
def build_Florence_3d_actions(dataset_dir,result_dir):
	load_file_name = os.path.join(dataset_dir,"Florence_dataset_WorldCoordinates.txt")
	with open(load_file_name,mode="r",encoding="utf-8") as rfp:
		lines = rfp.readlines()
		prev_video = int(lines[0][0])
		prev_categ = int(lines[0][2])
		frames_list = []
		train_list = []
		valid_list = []
		test_list = []
		train_label = []
		valid_label = []
		test_label = []
		for line in tqdm(lines,"processing dataset "):
			line = line.split(' ')
			vid = int(line[0])
			aid = int(line[1])
			cid = int(line[2])-1
			features = list(map(float, line[3:]))
			if prev_video == vid:
				frames_list.append(np.reshape(np.asarray(features), (-1,3)))
			else:
				if len(frames_list) >= 32:
					frames_list = random.sample(frames_list, 32)
					frames_list = torch.from_numpy(np.stack(frames_list, 0))
				else:
					frames_list = np.stack(frames_list, 0)
					xloc = np.arange(frames_list.shape[0])
					new_xloc = np.linspace(0, frames_list.shape[0], 32)
					frames_list = np.reshape(frames_list, (frames_list.shape[0], -1)).transpose()

					new_datas = []
					for data in frames_list:
						new_datas.append(np.interp(new_xloc, xloc, data))
					frames_list = torch.from_numpy(np.stack(new_datas, 0)).t()
				frames_list = frames_list.view(32, -1, 3)
				if prev_actor < 9:
					train_list.append(frames_list)
					train_label.append(prev_categ)
				elif prev_actor < 10:
					valid_list.append(frames_list)
					valid_label.append(prev_categ)
				else:
					test_list.append(frames_list)
					test_label.append(prev_categ)
				frames_list = [np.reshape(np.asarray(features), (-1,3))]
			prev_actor = aid
			prev_video = vid
			prev_categ = cid
	if len(frames_list) >= 32:
		frames_list = random.sample(frames_list, 32)
		frames_list = torch.from_numpy(np.stack(frames_list, 0))
	else:
		frames_list = np.stack(frames_list, 0)
		xloc = np.arange(frames_list.shape[0])
		new_xloc = np.linspace(0, frames_list.shape[0], 32)
		frames_list = np.reshape(frames_list, (frames_list.shape[0], -1)).transpose()

		new_datas = []
		for data in frames_list:
			new_datas.append(np.interp(new_xloc, xloc, data))
		frames_list = torch.from_numpy(np.stack(new_datas, 0)).t()
	frames_list = frames_list.view(32, -1, 3)
	if aid < 9:
		train_list.append(frames_list)
		train_label.append(prev_categ)
	elif aid < 10:
		valid_list.append(frames_list)
		valid_label.append(prev_categ)
	else:
		test_list.append(frames_list)
		test_label.append(prev_categ)

	train_label = torch.from_numpy(np.asarray(train_label))
	valid_label = torch.from_numpy(np.asarray(valid_label))
	test_label  = torch.from_numpy(np.asarray(test_label))

	save_train_file_name = os.path.join(result_dir,"train.pkl")
	torch.save((torch.stack(train_list, 0), train_label),save_train_file_name)
	save_valid_file_name = os.path.join(result_dir,"valid.pkl")
	torch.save((torch.stack(valid_list, 0), valid_label), save_valid_file_name)
	save_test_file_name = os.path.join(result_dir,"test.pkl")
	torch.save((torch.stack(test_list, 0),  test_label),save_test_file_name)

def main():
	args = get_args()
	dataset_dir = os.path.join(args.data_dir,args.dataset)
	result_dir = os.path.join(args.result_dir,args.dataset)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	build_Florence_3d_actions(dataset_dir,result_dir)
if __name__ == "__main__":
	main()

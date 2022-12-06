import os
import argparse
def check_args(args):
	result_dir = os.path.join(args.result_dir,args.dataset)
	args.train_path = os.path.join(result_dir,"train.pkl")
	args.valid_path = os.path.join(result_dir,"valid.pkl")
	args.test_path =  os.path.join(result_dir,"test.pkl")
	args.train_label_path =  os.path.join(result_dir,"train_label.pkl")
	args.valid_label_path =  os.path.join(result_dir,"valid_label.pkl")
	args.test_label_path =  os.path.join(result_dir,"test_label.pkl")
	args.model_path = os.path.join(args.log_dir,args.dataset)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',  type=str, default='./data')
	parser.add_argument('--result_dir',  type=str, default='./result')
	parser.add_argument('--log_dir',  type=str, default='./log')
	parser.add_argument("--dataset",type=str,default="Florence_3d_actions")
	parser.add_argument('--batch_size',  type=int, default=1)
	parser.add_argument('--learning_rate',type=int, default=0.005)
	parser.add_argument('--beta1',type=int, default=0.5)
	parser.add_argument('--beta2',type=int, default=0.99)
	parser.add_argument('--dropout_rate',type=int, default=0.0)
	parser.add_argument('--weight_decay',type=int, default=0.0)

	parser.add_argument('--num_epochs',type=int, default=30)
	parser.add_argument('--start_epoch',type=int, default=0)
	parser.add_argument('--test_epoch',type=int, default=30)
	parser.add_argument('--val_step',type=int, default=2)

	parser.add_argument('--num_classes',type=int, default=9)
	parser.add_argument('--feat_dims',type=int, default=13)
	
	args = parser.parse_args()

	return args

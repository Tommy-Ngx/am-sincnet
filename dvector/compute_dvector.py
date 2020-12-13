import os
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import scipy.io.wavfile
import torch
import torchaudio
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import sys
import numpy as np
from mobilenet1d import MobileNetV2
from data_io import ReadList, read_conf, str_to_bool
from tqdm import tqdm
np.set_printoptions(threshold=1e6)

model_file = '/home/ubuntu/drive_a/data_21_03/AM-MobileNet1D/exp/AM_MobileNet1D_TIMIT/old_model/model_raw_48.pkl'
# Config file of the speaker-id experiment used to generate the model
cfg_file = '/home/ubuntu/drive_a/data_21_03/AM-MobileNet1D/cfg/AM_MobileNet1D_TIMIT.cfg'
te_lst = 'data_lists/TIMIT_test_sample.scp'  # List of the wav files to process
# output dictionary containing the a sentence id as key as the d-vector as value
out_dict_file = '/home/ubuntu/drive_a/data_21_03/AM-MobileNet1D/d_vect_timit_1.npy'
data_folder = '/home/ubuntu/drive_a/data_21_03/AM-MobileNet1D/timit_normalized/'

# Reading cfg file
options = read_conf()

# [data]
#tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
#class_dict_file = options.lab_dict
#data_folder = options.data_folder+'/'
output_folder = options.output_folder

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [class]
class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(
	map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(
	map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))


# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)


# training list
# wav_lst_tr=ReadList(tr_lst)
# snt_tr=len(wav_lst_tr)

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# Folder creation
try:
	os.stat(output_folder)
except:
	os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# Converting context and shift in samples
wlen = int(fs*cw_len/1000.00)
wshift = int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev = 128

# ----- MobileNet -----
MOBILENET_net = MobileNetV2(num_classes=class_lay)
MOBILENET_net.cuda()


print('LOADING MODEL.')
checkpoint_load = torch.load(model_file)
MOBILENET_net.load_state_dict(checkpoint_load['MOBILENET_model_par'])

optimizer_MOBILENET = optim.RMSprop(MOBILENET_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

# for epoch in range(N_epochs):   #### changes ####

test_flag = 0
MOBILENET_net.train()

loss_sum = 0
err_sum = 0

MOBILENET_net.eval() #### new changes ####

test_flag = 1
#   loss_sum=0
#   err_sum=0
#   err_sum_snt=0

d_vect_dict={}
correct = 0

with torch.no_grad():			

	for i in range(snt_te):
		
		[signal, fs] = torchaudio.load(data_folder+wav_lst_te[i])  # reading with torchaudio
		# [signal, fs] = sf.read(data_folder+wav_lst_te[i])
		#signal = torch.from_numpy(signal).float().cuda().contiguous()
		
		
		#lab_batch = lab_dict[wav_lst_te[i]]  # label for batch?

		# split signals into chunks
		beg_samp = 0
		end_samp = wlen

		N_fr = int((signal.shape[1]-wlen)/(wshift))

		sig_arr=np.zeros([Batch_dev, 1, wlen])
		
		pout = Variable(torch.zeros(
			N_fr+1, class_lay[-1]).float().cuda().contiguous())
		count_fr = 0
		count_fr_tot = 0
		while end_samp < signal.shape[1]:
			sig_arr[count_fr, :] = signal[:,beg_samp:end_samp]
			beg_samp = beg_samp+wshift
			end_samp = beg_samp+wlen
			count_fr = count_fr+1
			count_fr_tot = count_fr_tot+1
			if count_fr == Batch_dev:
				inp = Variable(torch.from_numpy(sig_arr).float().cuda().contiguous())
				pout[count_fr_tot - Batch_dev:count_fr_tot,:] = MOBILENET_net(inp)
				count_fr = 0
				sig_arr = np.zeros([Batch_dev,1, wlen])

		if count_fr > 0:
			inp = Variable(torch.from_numpy(sig_arr[0:count_fr]).float().cuda().contiguous())
			
			pout[count_fr_tot - count_fr:count_fr_tot, :] = MOBILENET_net(inp)
		
		# averaging and normalizing all the d-vectors
		d_vect_out=torch.mean(pout/pout.norm(p=2, dim=1).view(-1,1),dim=0)
         
        	#print(d_vect)
		nan_sum=torch.sum(torch.isnan(d_vect_out))

		if nan_sum>0:
			print(wav_lst_te[i])
			sys.exit(0)
		print(d_vect_out.shape) # saving the d-vector in a numpy dictionary
		dict_key=wav_lst_te[i].split('/')[-2]+'/'+wav_lst_te[i].split('/')[-1]
		d_vect_dict[dict_key]=d_vect_out.cpu().numpy()
		print(dict_key)

# Save the dictionary
np.save(out_dict_file, d_vect_dict)
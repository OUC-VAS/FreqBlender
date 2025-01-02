from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd

# your_path = 'your dataset path'

your_path = "/meida/lhz/Data"

def init_ff(dataset,comp,phase, fake_dataset='Deepfakes',level='frame',n_frames=8):
	if dataset=='Original':
		dataset_path= your_path + '/FF++/original_sequences/youtube/{}/videos/'.format(comp)
	else:
		dataset_path= your_path + '/FF++/manipulated_sequences/{}/{}/videos/'.format(fake_dataset,comp)

	image_list=[]
	label_list=[]
	
	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(your_path + f'/FF++/{phase}.json','r'))
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if phase=='test':
		fake_path= your_path + '/FF++/manipulated_sequences/{}/{}/videos/'.format(fake_dataset,comp)
		fake_folder_list = sorted(glob(fake_path+'*'))
		fake_folder_list = [i for i in fake_folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		if phase=='test':
			label_list=label_list+[1]*len(fake_folder_list)
			folder_list=folder_list+fake_folder_list
		return folder_list,label_list
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		if dataset=='Original':
			label_list+=[0]*len(images_temp)
		else:
			label_list+=[1]*len(images_temp)

	return image_list,label_list



def init_dfdc():
	path = your_path + '/DFDC'

	folder_list = []
	label_list = []
	label = pd.read_csv(path + '/labels.csv', delimiter=',')
	for idx, content in label.iterrows():
		if os.path.exists(path + '/videos/' + content['filename']):
			folder_list.append(path + '/videos/' + content['filename'])
			label_list.append(content['label'])
	return folder_list, label_list




def init_dfdcp(phase='test'):
	phase_integrated = {'train': 'train', 'val': 'train', 'test': 'test'}

	path = your_path + '/DFDCP'

	with open(path+'/dataset.json') as f:
		df = json.load(f)
	fol_lab_list_all = [[path+'/'+ k, df[k]['label'] == 'fake'] for k in df if df[k]['set'] == phase_integrated[phase]]
	name2lab = {os.path.basename(fol_lab_list_all[i][0]): fol_lab_list_all[i][1] for i in range(len(fol_lab_list_all))}
	fol_list_all = [f[0] for f in fol_lab_list_all]
	fol_list_all = [os.path.basename(p) for p in fol_list_all]
	folder_list = glob(path+'/method_*/*/*/*.mp4') + glob(path+'/original_videos/*/*.mp4')
	folder_list = [p for p in folder_list if os.path.basename(p) in fol_list_all]
	label_list = [name2lab[os.path.basename(p)] for p in folder_list]

	return folder_list, label_list

def init_ffiw():
	# assert dataset in ['real','fake']
	path= your_path + '/FFIW10K-v1/FFIW10K-v1-release/'

	folder_list=sorted(glob(path+'source/val/*.mp4'))+sorted(glob(path+'target/val/*.mp4'))
	label_list=[0]*250+[1]*250
	return folder_list,label_list


def init_cdf():
	video_list_txt= your_path + '/Celeb-DF-v2/List_of_testing_videos.txt'

	image_list=[]
	label_list=[]
	with open(video_list_txt) as f:

		folder_list=[]
		for data in f:
			#print(data)
			line=data.split()
			#print(line)
			path=line[1].split('/')
			folder_list+=[your_path + '/Celeb-DF-v2/'+path[0]+'/'+path[1]]
			label_list+=[1-int(line[0])]
		return folder_list,label_list


def init_stylegan2():
	path= your_path + '/StyleGAN2/test/'

	data=pd.read_csv(path+'test.csv')
	folder_list=[]
	label_list = []
	for index, file in data.iterrows():
		folder_list += [your_path + '/StyleGAN2/test/'+str(file['label'])+'/'+file['filename']]
		label_list += [abs(file['label'] - 1)]

	return folder_list, label_list



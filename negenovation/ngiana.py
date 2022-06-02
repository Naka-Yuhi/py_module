import os
import glob
import pickle
import numpy as np
import copy
from natsort import natsorted

import py_module.negenovation as ngi


#############################################################################
#
#				myFFT
#
#############################################################################
def myFFT(data,T,split_rate=0.05,overrap=0.5,window_F="hanning"):
	"""
	
	
	"""

	data_split = []
	len_data = data.shape[0]
	len_split = int(len_data*split_rate)
	len_overrap = int(len_split*overrap)


	#time = np.arange(0,100,T)
	#time = time[0:len_data]


	indx_start = 0
	indx_end = len_split


	#Overrapping process
	while True:
		#print("start:" + str(indx_start) + ", end:" + str(indx_end))
		data_split.append(data[indx_start:indx_end])
		indx_start = indx_start + ( len_split - len_overrap )
		indx_end = indx_start + len_split


		if indx_end > len_data:
			break


	#window function and FFT
	num_of_data = len(data_split)
	N = data_split[0].shape[0]


	#A variable for results of FFT
	results_FFT = np.empty((N,num_of_data))


	#creating a window function
	if window_F == "hanning":
		window = np.hanning(N)		  # ハニング窓
	elif window_F == "hamming":
		window = np.hamming(N)		  # ハミング窓
	elif window_F == "blackman":
		window = np.blackman(N)		 # ブラックマン窓
	else:
		print("Error: input window function name is not sapported. Your input: ", window_F)
		print("Hanning window function is used.")
		window = np.hanning(N)		  # ハニング窓


	for i, data_each in enumerate(data_split):
		data_applied_window = data_each*window
		results_FFT[:,i] = (2/N)*np.abs(np.fft.fft(data_applied_window))
		
		
	freq_data = np.fft.fftfreq(N,T)


	results_FFT_ave = np.average(results_FFT,axis=1)
	return_fft_data = np.empty((int(N/2)-1,2))
	return_fft_data[:,0] = freq_data[1:int(N/2)]
	return_fft_data[:,1] = results_FFT_ave[1:int(N/2)]


	return return_fft_data

def getFFTMap( data_list,T,time):
	data_size = len(data_list)

	for i,acc_data in enumerate(data_list):
		
		fft_x = myFFT(acc_data[0],T)
		fft_y = myFFT(acc_data[1],T)
		fft_z = myFFT(acc_data[2],T)

		if i == 0:
			fft_map = np.empty(fft_x.shape[0],data_size,3)
			freq = fft_x[:,0]
		
		fft_map[:,i,0] = fft_x[:,1]
		fft_map[:,i,1] = fft_x[:,1]
		fft_map[:,i,2] = fft_x[:,1]

	return fft_map,freq,time

#############################################################################
#
#				getPeak
#
#############################################################################
def getPeak(data,range_list,sort_index=1,picked_data=20,pattern='max'):
	"""
	sorted:
	
	Parameters
	---
	data : ndarray
	
	
	range_list : list
	
	Structure of range_list
	[range_width, range_ofs ]
	
	
	sort_index : int
	
	picked_data : int
	
	pattern : str
	
	
	"""
	size_data = data.shape[0]
	extracted_index = []

	range_ofs = range_list[1]
	range_width = range_list[0]
	
	if pattern == 'max':
		data_return = np.empty((len(range_width),2))
		
		for i,range_val in enumerate( range_width ):
			logi = (data[:,0] >= range_val-range_ofs) & (data[:,0] <= range_val+range_ofs)
			data_partly = data[logi,:]
			
			max_index = np.argmax(data_partly[:,1])
			
			data_return[i,:] = data_partly[max_index,:]
			
		
		
	elif pattern == 'compare' | pattern == 'c':
		for range_val in range_width:

			flag_range = False
			for indx in range(picked_data,size_data-picked_data-1):
				value_x = data[indx,0]

				if ( range_val-range_ofs <= value_x ) & ( range_val+range_ofs >= value_x ):

					if flag_range == False:
						flag_range = True

					value_cur = data[indx,sort_index]
					flag = True

					for indx2 in range(1,picked_data):
						value_comp01 = data[indx-indx2,sort_index]
						value_comp02 = data[indx+indx2,sort_index]

						if (value_cur - value_comp01 <= 0) | ( value_cur - value_comp02 <= 0):
							flag = False
							break

					if flag:
						extracted_index.append(indx)
						break
				else:

					if flag_range == True:
						break
					continue
		data_return = data[np.array(extracted_index),:]
	#print(extracted_index)
	
	return data_return


def moduletest():
	ngi.testcall()
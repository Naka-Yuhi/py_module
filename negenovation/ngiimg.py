import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image

import os
import glob
from natsort import natsorted

def show_images(images,title_str, columns = 3):
	
	"""
	aiutef
	
	Parameters
	---
	images : list
		Imageクラスのリスト
		example : images = [Image.open("test.jpg") Image.open("test2.jpg")]
		
	title_str : list
	
	figsize
	
	columns : int
	
	"""
	sub_row = math.ceil(len(images)/columns)
	
	
	fig = plt.figure(figsize=(columns*4,sub_row*4),dpi=80)
	
	
	ax = fig.subplots(sub_row,columns)
	
	if type(ax) != np.ndarray:
		ax.set_title(title_str,fontsize=20)
		ax.axis('off')
		ax.imshow(images[0])
	else:
		ar = ax.ravel()
		for i, image in enumerate(images,0):

			#print(sub_row)
			ar[i].set_title(title_str[i],fontsize=20)
			ar[i].axis('off')
			ar[i].imshow(image)
	

	fig.tight_layout()
	plt.show()

########################################################################

def getIMG(path):
	
	img_work = []
	img_tool_side = []
	img_tool_frontS = []
	img_tool_frontL = []
	img_tool_naname = []
	
	
	folders_lv1 = natsorted(glob.glob(  os.path.join( path, '[0-9-_]*')   ))
	
	for folder_lv1 in folders_lv1:

		files_img_work = natsorted(glob.glob(  os.path.join( folder_lv1, 'image','work','*.jpg')))
		files_img_tool = natsorted(glob.glob(  os.path.join( folder_lv1, 'image','tool','*.jpg')))
		
		print("img_work: %d, img_tool: %d" % ( len(files_img_work),len(files_img_tool) ))

		if len(files_img_tool) != 0:
			img_tool_side.append(Image.open(files_img_tool[0],"r"))
			img_tool_frontS.append(Image.open(files_img_tool[1],"r"))
			img_tool_frontL.append(Image.open(files_img_tool[2],"r"))
			img_tool_naname.append(Image.open(files_img_tool[3],"r"))


		if len(files_img_work) != 0:
			img_tmp = []
			for img_path in files_img_work:
				img_tmp.append(Image.open(img_path,"r"))
		
			img_work.append(img_tmp)


		else:
			folders_lv2 = natsorted(glob.glob(  os.path.join( folder_lv1, '[0-9h_.]*')   ))

			for folder_lv2 in folders_lv2:
			
				files_img_work = natsorted(glob.glob(  os.path.join( folder_lv2, 'image','work','*.jpg')))
				files_img_tool = natsorted(glob.glob(  os.path.join( folder_lv2, 'image','tool','*.jpg')))
				
				##-----------------------
				if len(files_img_work) != 0:
					img_tmp = []
					for img_path in files_img_work:
						img_tmp.append(Image.open(img_path,"r"))
				
					img_work.append(img_tmp)
				
				##-----------------------
				if len(files_img_tool) != 0:
					img_tool_side.append(Image.open(files_img_tool[0],"r"))
					img_tool_frontS.append(Image.open(files_img_tool[1],"r"))
					img_tool_frontL.append(Image.open(files_img_tool[2],"r"))
					img_tool_naname.append(Image.open(files_img_tool[3],"r"))
	
	return (img_work,img_tool_side,img_tool_frontS,img_tool_frontL,img_tool_naname)
	
	
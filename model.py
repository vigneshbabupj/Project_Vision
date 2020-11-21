

import torch
import torch.nn as nn

from encoder import _make_resnet_encoder

from depth_decoder.depth_decoder import MidasNet_decoder

from plane_decoder.plane_decoder import MaskRCNN

from bbox_decoder.yolov3_bbox_decoder import *

class VisionNet(nn.Module):

	'''
		Network for detecting objects, generate depth map and identify plane surfaces
	'''

	def __init__(self,yolo_cfg,midas_cfg,planercnn_cfg,path=None):
		super(VisionNet, self).__init__()
		"""
			Get required configuration for all the 3 models
		
		"""
		self.yolo_params = yolo_cfg
		self.midas_params = midas_cfg
		self.planercnn_params = planercnn_cfg
		self.path = path

		use_pretrained = False if path is None else True

		print('use_pretrained',use_pretrained)
		print('path',path)
		
		self.encoder = _make_resnet_encoder(use_pretrained)


		self.depth_decoder = MidasNet_decoder(path)

		self.plane_decoder = MaskRCNN(self.planercnn_params,self.encoder) #options, config, modelType='final'

		self.bbox_decoder =  Darknet(self.yolo_params)

		self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), padding=0, bias=False)
		self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=0, bias=False)
		self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=0, bias=False)

	def forward(self,yolo_ip,midas_ip,plane_ip):

		#x = yolo_ip
		x = midas_ip
		print('yolo_ip',yolo_ip.shape,yolo_ip[0][0][0][0])
		print('midas_ip',midas_ip.shape,midas_ip[0][0][0][0])

		# Encoder blocks
		layer_1 = self.encoder.layer1(x)
		layer_2 = self.encoder.layer2(layer_1)
		layer_3 = self.encoder.layer3(layer_2)
		layer_4 = self.encoder.layer4(layer_3)

		#print('%'*30,'layer_1',layer_1[0][0][0][0])
		print('layer_4',layer_4)

		Yolo_75 = self.conv1(layer_4)
		Yolo_61 = self.conv2(layer_3)
		Yolo_36 = self.conv3(layer_2)
		

		# MiDaS depth decoder
		depth_out = self.depth_decoder([layer_1, layer_2, layer_3, layer_4])

		# PlaneRCNN decoder
		plane_out = self.plane_decoder.forward(plane_ip,[layer_1, layer_2, layer_3, layer_4])

		#print('en Yolo_75 :',Yolo_75.shape)
		#print('en Yolo_61 :',Yolo_61.shape)
		#print('en Yolo_36 :',Yolo_36.shape)

		#print('^'*66,'Yolo_75',Yolo_75[0][0])
		#YOLOv3 bbox decoder
		bbox_out = self.bbox_decoder(Yolo_75,Yolo_61,Yolo_36)

		print('depth_out',depth_out)

		return bbox_out, depth_out, plane_out
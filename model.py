

import torch
import torch.nn as nn

from encoder import *

from depth_decoder.depth_decoder import MidasNet_decoder

from plane_decoder.plane_decoder import MaskRCNN

class VisionNet(nn.Module):
	'''
		Network for detecting objects, generate depth map and identify plane surfaces
	'''

	def __init__(self,yolo_cfg,midas_cfg,planercnn_cfg):
		super(VisionNet, self).__init__()
		"""
			Get required configuration for all the 3 models
		
		"""
		self.yolo_params = parse(yolo_cfg)
		self.midas_params = parse(midas_cfg)
		self.planercnn_params = parse(planercnn_cfg)
		self.path =  path

		use_pretrained = False if path is None else True
		
		self.encoder = _make_resnet_encoder(use_pretrained)

		

		self.depth_decoder = MidasNet_decoder(path)

		self.plane_decoder = MaskRCNN(self.planercnn_params,self.encoder)

		self.bbox_decoder =  Darknet(self.yolo_params)


	def forward(self,x):

		# Encoder blocks
		layer_1 = self.encoder.layer1(x)
		layer_2 = self.encoder.layer2(layer_1)
		layer_3 = self.encoder.layer3(layer_2)
		layer_4 = self.encoder.layer4(layer_3)
		

		# MiDaS depth decoder
		depth_out = self.depth_decoder([layer_1,layer_2,layer_3,layer_4])

		# PlaneRCNN decoder
		plane_out = self.plane_decoder.predict() #have to decide passing params

		#YOLOv3 bbox decoder
		bbox_out = self.bbox_decoder(layer_4)

		return bbox_out,depth_out,plane_out
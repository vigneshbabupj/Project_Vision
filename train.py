#Yolo import
import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

#import test  # import test.py to get mAP after each epoch

from bbox_decoder import bbox_test


#from models import *
#from utils.datasets import *
from bbox_decoder.utils.utils import *
from bbox_decoder.utils import torch_utils

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    # print('Apex recommended for mixed precision and faster training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

#Yolo import

#Planercnn
import torch
from torch import optim
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import numpy as np
import cv2
import sys

#from models.model import *
#from models.refinement_net import *
from plane_decoder.modules import *
#from datasets.plane_stereo_dataset import *

from plane_decoder.utils import *
from plane_decoder.visualize_utils import *
from plane_decoder.evaluate_utils import *
#from options import parse_args
from plane_decoder.config import InferenceConfig# PlaneConfig

#planercnn
import glob
import numpy as np
import math


from model import *

from dataset import *

from plane_decoder.plane_decoder import *

import pytorch_ssim
from pytorch_msssim import msssim


from torch.autograd import Variable





def train(plane_args,yolo_args,midas_args,add_plane_loss,add_yolo_loss,add_midas_loss,resume_train=False):

    #Plane config
    options = plane_args
    config = InferenceConfig(options)

    ## Start yolo train setup 
    opt = yolo_args

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)

    # Image Sizes
    gs = 64  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    #model = Darknet(cfg).to(device)
    model = VisionNet(yolo_cfg=cfg,midas_cfg=None,planercnn_cfg=config,path=midas_args.weights).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    start_epoch = 0
    best_loss = 1000
    

    if resume_train:

        Vsn_chkpt = torch.load('visionet_checkpoint.pt', map_location=device)

        model.load_state_dict(Vsn_chkpt['state_dict'])
        optimizer.load_state_dict(Vsn_chkpt['optimizer'])
        best_loss = Vsn_chkpt['best_loss']
        start_epoch = Vsn_chkpt['epoch'] + 1

        del Vsn_chkpt

    else:
        attempt_download(weights)

        if weights.endswith('.pt'):  # pytorch format
            # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            chkpt = torch.load(weights, map_location=device)

            # load model
            try:
                #chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(chkpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                    "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
                raise KeyError(s) from e

            # load optimizer
            if chkpt['optimizer'] is not None:
                print('loading Optimizer')
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            start_epoch = chkpt['epoch'] + 1
            del chkpt

        elif len(weights) > 0:  # darknet format
            # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            load_darknet_weights(model, weights)


        #load planercnn weighta
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'),strict=False)

        #load midas pretrained weighta

        midas_parameters = torch.load(midas_args.weights)

        if "optimizer" in midas_parameters:
            midas_parameters = midas_parameters["model"]

        model.load_state_dict(midas_parameters,strict=False)



    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [round(epochs * x) for x in [0.8, 0.9]], 0.1, start_epoch - 1)

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level


    yolo_params = dict(path = train_path, img_size = img_size, batch_size = batch_size, augment=True,hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)


    planercnn_params = dict(options=options, config=config,random=False)

    midas_params=None

    ## Intit dataset
    train_dataset = create_data(yolo_params,planercnn_params,midas_params)

    yolo_params_test = dict(path = test_path, img_size = imgsz_test, batch_size = batch_size, augment=False,hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)


    test_dataset = create_data(yolo_params_test,planercnn_params,midas_params)


    #yolo data loader

    # Dataloader
    batch_size = min(batch_size, len(train_dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=train_dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=test_dataset.collate_fn)
 

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(trainloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    loss_list=[]


    for epoch in range(start_epoch, start_epoch+epochs):  # epoch ------------------------------------------------------------------
        #model.train()
    ## END yolo train setup 

        model.train()

        #print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        print(('\n' + '%10s' * 8) % ('Epoch', 'Dp_SSIM', 'Dp_Rmse', 'Dp_loss', 'bbx_loss', 'pln_loss', 'All_loss', 'img_size'))

        pbar = tqdm(enumerate(trainloader))

        #optimizer.zero_grad()

        mloss = torch.zeros(4).to(device)  # mean losses

        for i,(plane_data,yolo_data,depth_data) in pbar:

            optimizer.zero_grad()

            #print('i:',i)
            # print('plane :',len(plane_data))
            # print('yolo :',len(yolo_data))
            # print('depth :',len(depth_data))

            #yolov3 init start
            imgs, targets, paths, _ = yolo_data

            #print('path',paths,'shape',_)

            #if paths[0] in ['./data/customdata/images/Mimg_077.jpg','./data/customdata/images/Himage_102.jpg','./data/customdata/images/majdoor_23.jpg']:
            #    continue
            

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn * 2:
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou yolo_loss ratio (obj_loss = 1.0 or giou)
                if ni == n_burn:  # burnin complete
                    print_model_biases(model)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            yolo_inp = imgs
            
            # Run model
            #pred = model(imgs)

            ## Yolov3 init end


            #depth init start
            dp_img_size,depth_img,depth_target = depth_data #######

            #print('depth:',type(depth_img),len(depth_img))
            #print('np size:',np.asarray(depth_img).shape)

            dp_sample = torch.from_numpy(depth_img).to(device).unsqueeze(0) ######
            #print('dp_sample',dp_sample)

            midas_inp = dp_sample ####
            #dp_prediction = model.forward()
            #depth init end

            #planercnn init start
            data_pair,plane_img,plane_np = plane_data
            sampleIndex = i
            sample = data_pair

            #print('sample',len(sample))

            plane_losses = []            

            input_pair = []
            detection_pair = []
            dicts_pair = []

            camera = sample[30][0].cuda()

            #for indexOffset in [0, ]:
            indexOffset=0
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
            # print('gt_boxes',gt_boxes.size())
            # print('images',images.size())
            # print('image_metas',image_metas)
            # print('rpn_match',rpn_match.size())
            # print('rpn_bbox',rpn_bbox.size())
            # print('gt_class_ids',gt_class_ids.size())
            # print('gt_masks',gt_masks)
            # print('gt_parameters',gt_parameters.size())
            # print('gt_depth',gt_depth.size())
            # print('extrinsics',extrinsics.size())
            # print('planes',planes.size())
            # print('gt_segmentation',gt_segmentation.size())

            masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()
            input_pair.append({'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0], 'masks': masks, 'mask': gt_masks})
            
            plane_inp = dict(input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True, return_feature_map=False)

            #planercnn init end


            # All model prediction start

            plane_out,yolo_out,midas_out = model.forward(yolo_inp,midas_inp,plane_inp)

            pred = yolo_out
            dp_prediction = midas_out           

            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = plane_out


            # All model prediction End

            #Vignesh : block planercnn
            #Planercnn start
            # print('detections   :',len(detections))
            # print('rpn_class_logits',len(rpn_class_logits),rpn_class_logits.shape)
            # print('rpn_pred_bbox',len(rpn_pred_bbox))
            # print('target_class_ids',len(target_class_ids))
            # print('mrcnn_class_logits',len(mrcnn_class_logits))
            # print('target_deltas',len(target_deltas))
            # print('mrcnn_bbox',len(mrcnn_bbox))
            # print('target_mask',len(target_mask))
            # print('mrcnn_mask',len(mrcnn_mask))
            # print('target_parameters',len(target_parameters))
            # print('mrcnn_parameters',len(mrcnn_parameters))
            # print('detections',len(detections))
            # print('detection_masks',len(detection_masks))
            # print('detection_gt_parameters',len(detection_gt_parameters))
            # print('detection_gt_masks',len(detection_gt_masks))
            # print('rpn_rois',len(rpn_rois))
            # print('roi_features',len(roi_features))
            # print('roi_indices',len(roi_indices))
            # print('depth_np_pred',len(depth_np_pred),depth_np_pred.size())


            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss = compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters)

            plane_losses = [rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_parameter_loss]


            if depth_np_pred.shape != gt_depth.shape:
                depth_np_pred = torch.nn.functional.interpolate(depth_np_pred.unsqueeze(1), size=(512, 512), mode='bilinear',align_corners=False).squeeze(1)
                pass

            if config.PREDICT_NORMAL_NP:
                normal_np_pred = depth_np_pred[0, 1:]                    
                depth_np_pred = depth_np_pred[:, 0]
                gt_normal = gt_depth[0, 1:]                    
                gt_depth = gt_depth[:, 0]
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                normal_np_loss = l2LossMask(normal_np_pred[:, 80:560], gt_normal[:, 80:560], (torch.norm(gt_normal[:, 80:560], dim=0) > 1e-4).float())
                plane_losses.append(depth_np_loss)
                plane_losses.append(normal_np_loss)
            else:
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                plane_losses.append(depth_np_loss)
                normal_np_pred = None
                pass

            if len(detections) > 0:
                detections, detection_masks = unmoldDetections(config, camera, detections, detection_masks, depth_np_pred, normal_np_pred, debug=False)
                if 'refine_only' in options.suffix:
                    detections, detection_masks = detections.detach(), detection_masks.detach()
                    pass
                XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
                detection_mask = detection_mask.unsqueeze(0)                        
            else:
                XYZ_pred = torch.zeros((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                detection_mask = torch.zeros((1, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                plane_XYZ = torch.zeros((1, 3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()                        
                pass


            #input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'parameters': detection_gt_parameters, 'plane': planes, 'camera': camera})
            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'plane_XYZ': plane_XYZ, 'depth_np': depth_np_pred})

            loss_fn = nn.MSELoss()

            plane_parameters = torch.from_numpy(plane_np['plane_parameters']).cuda()
            plane_masks = torch.from_numpy(plane_np['plane_masks']).cuda()
            plane_parameters_pred = detection_pair[0]['detection'][:, 6:9]
            plane_masks_pred = detection_pair[0]['masks'][:, 80:560]

            if plane_parameters_pred.shape != plane_parameters.shape:
                plane_parameters_pred = torch.nn.functional.interpolate(plane_parameters_pred.unsqueeze(1).unsqueeze(0), size=plane_parameters.shape, mode='bilinear',align_corners=True).squeeze()
                pass
            if plane_masks_pred.shape != plane_masks.shape:
                plane_masks_pred = torch.nn.functional.interpolate(plane_masks_pred.unsqueeze(1).unsqueeze(0), size=plane_masks.shape, mode='trilinear',align_corners=True).squeeze()
                pass

           
            
            plane_params_loss = loss_fn(plane_parameters_pred,plane_parameters) + loss_fn(plane_masks_pred,plane_masks)

            print('plane_params_loss',plane_params_loss)


            predicted_detection = visualizeBatchPair(options, config, input_pair, detection_pair, indexOffset=i)
            predicted_detection = torch.from_numpy(predicted_detection)

            #print('predicted_detection',len(predicted_detection))
            #print(predicted_detection)
            print(predicted_detection.shape)
            print(plane_img.shape)

            if predicted_detection.shape != plane_img.shape:
                predicted_detection = torch.nn.functional.interpolate(predicted_detection.permute(2,0,1).unsqueeze(0).unsqueeze(1), size=plane_img.permute(2,0,1).shape, mode='trilinear',align_corners=True).squeeze()
                pass

            pln_rmse = torch.sqrt(loss_fn(predicted_detection, plane_img))



            # if 'depth' in options.suffix:
            #     ## Apply supervision on reconstructed depthmap (not used currently)
            #     if len(detections) > 0:
            #         background_mask = torch.clamp(1 - detection_masks.sum(0, keepdim=True), min=0)
            #         all_masks = torch.cat([background_mask, detection_masks], dim=0)

            #         all_masks = all_masks / all_masks.sum(0, keepdim=True)
            #         all_depths = torch.cat([depth_np_pred, plane_XYZ[:, 1]], dim=0)

            #         depth_loss = l1LossMask(torch.sum(torch.abs(all_depths[:, 80:560] - gt_depth[:, 80:560]) * all_masks[:, 80:560], dim=0), torch.zeros(config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM).cuda(), (gt_depth[0, 80:560] > 1e-4).float())
            #     else:
            #         depth_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
            #         pass
            #     plane_losses.append(depth_loss)                                                
            #     pass                    
            #continue

            # if (len(detection_pair[0]['detection']) > 0 and len(detection_pair[0]['detection']) < 30) and 'refine' in options.suffix:
            #     #use refinement network
            #     pass
            # else:
            #     plane_losses += [torch.zeros(1).cuda()]
            #     pass

            ## The warping yolo_loss
            # for c in range(1, 2):
            #     if 'warping' not in options.suffix:
            #         pass
            #         #break
            #     #continue

        

            #print('plane_losses :',plane_losses)
            print('pln_rmse',pln_rmse)
            plane_loss = sum(plane_losses) + pln_rmse
            plane_losses = [l.data.item() for l in plane_losses] #train_planercnn.py 331

            #statistics = [[], [], [], []]

            #for c in range(len(input_pair)):
            #    evaluateBatchDetection(options, config, input_pair[c], detection_pair[c], statistics=statistics, printInfo=True, evaluate_plane=options.dataset == '')
                        

            #print('plane_loss : ',plane_loss)
            #print('plane_losses : ',plane_losses)


            # epoch_losses.append(losses)
            # status = str(epoch + 1) + ' yolo_loss: '
            # for l in losses:
            #     status += '%0.5f '%l
            #     continue
            
            # data_iterator.set_description(status)

            # yolo_loss.backward()

            #planercnn END
            #Vignesh : block planercnn

            

            ## Midas start
            

            #dp_prediction = F.interpolate(dp_prediction.unsqueeze(1),size=dp_img_size[:2])



            dp_prediction = (
                            torch.nn.functional.interpolate(
                                dp_prediction.unsqueeze(1),
                                size=tuple(dp_img_size[:2]),
                                mode="bicubic",
                                align_corners=False,
                            )
                            #.unsqueeze(0)
                            #.cpu()
                            #.numpy()
                            )
            bits=2

            depth_min = dp_prediction.min()
            depth_max = dp_prediction.max()

            max_val = (2**(8*bits))-1

            if depth_max - depth_min > np.finfo("float").eps:
                depth_out = max_val * (dp_prediction - depth_min) / (depth_max - depth_min)
            else:
                depth_out = 0
            
            depth_target = torch.from_numpy(np.asarray(depth_target)).to(device).type(torch.cuda.FloatTensor).unsqueeze(0)
            #print('depth_target',depth_target.size())


            depth_target = (
                            torch.nn.functional.interpolate(
                                depth_target.unsqueeze(1),
                                size=dp_img_size[:2],
                                mode="bicubic",
                                align_corners=False
                            )
                            #.unsqueeze(0)
                            #.cpu()
                            #.numpy()
                            )



            depth_pred = Variable( depth_out,  requires_grad=True)
            depth_target = Variable( depth_target, requires_grad = False)

            # print('dp_prediction',depth_out.shape)
            # print('depth_target',depth_target.shape)
            # import matplotlib.pyplot as plt

            # plt.imshow(depth_out.squeeze().cpu().detach().numpy())
            # plt.show()
            # plt.imshow(depth_target.squeeze().cpu().detach().numpy())
            # plt.show()

            #print('depth',[[len(x),x.size()]  for x in [depth_pred,depth_target]])

            ssim_loss = pytorch_ssim.SSIM() #https://github.com/Po-Hsun-Su/pytorch-ssim
            #print('ssim_loss :',ssim_loss(depth_pred,depth_target))
            #print('msssim :',msssim(depth_pred,depth_target))
            ssim_out = torch.clamp(1-ssim_loss(depth_pred,depth_target),min=0,max=1) #https://github.com/jorge-pessoa/pytorch-msssim
            
            RMSE_loss = torch.sqrt(loss_fn(depth_pred, depth_target))


            depth_loss = (0.01*RMSE_loss) + ssim_out


            #print('Depth loss :', 'ssim',ssim_out,'RMSE', RMSE_loss)
            ## Midas End

            
            

            #Yolov3 Start
            
            # Compute yolo_loss
            yolo_loss, yolo_loss_items = compute_loss(pred, targets, model)
            #print('yolo_loss : ', yolo_loss.item())
            if not torch.isfinite(yolo_loss):
                print('path:',paths)
                print('YOLO',[len(x) for x in [pred, targets]])
                #print('pred',pred)
                print('target',targets)
                print('WARNING: non-finite yolo_loss, ending training ', yolo_loss_items)
                return results

            # Scale yolo_loss by nominal batch_size of 64
            #yolo_loss *= batch_size / 64



            all_loss = (add_plane_loss * plane_loss) + (add_yolo_loss * yolo_loss) + (add_midas_loss * depth_loss)
            #all_loss = (add_yolo_loss * yolo_loss) + (add_midas_loss * ssim_out)
            # print('plane_loss : ', plane_loss)
            # print('yolo_loss : ', yolo_loss)
            # print('ssim_out : ', depth_loss)
            # print('all_loss :',all_loss)

            #optimizer.zero_grad()

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(all_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                all_loss.backward()
                pass

            # Optimize accumulated gradient
            #if ni % accumulate == 0:
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            ema.update(model)


            # Print batch results
            mloss = (mloss * i + yolo_loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

            #s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            print(('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size))
            s = ('%10s'+'%10.3g' + '%10.3g' * 6) % ('%g/%g' % (epoch, (start_epoch+epochs) - 1), ssim_out.item(), RMSE_loss.item(), depth_loss.item(), yolo_loss.item(), plane_loss.item(), all_loss.item(), img_size)
            #print(('\n' + '%10s' * 8) % ('Epoch', 'Dp_SSIM', 'Dp_Rmse', 'Dp_loss', 'bbx_loss', 'plnrn_loss', 'All_loss', 'img_size'))
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()


        ## Yolov3 test start

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = bbox_test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      img_size=imgsz_test,
                                      model=ema.ema,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      dataloader=testloader)

        ## Yolov3 test end



        ## planercnn test start
        '''

        plane_test fun:

            if 'inference' not in options.dataset:
                    for c in range(len(input_pair)):
                        evaluateBatchDetection(options, config, input_pair[c], detection_pair[c], statistics=statistics, printInfo=options.debug, evaluate_plane=options.dataset == '')
                        continue
                else:
                    for c in range(len(detection_pair)):
                        np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_parameters_' + str(c) + '.npy', detection_pair[c]['detection'][:, 6:9])
                        np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_masks_' + str(c) + '.npy', detection_pair[c]['masks'][:, 80:560])
                        continue
                    pass

        ## planercnn test end

        ## midas start

        midas_test fun:
        '''    

        ## midas end

        ##Save model start

        visionet_checkpoint = {'best_loss':best_loss,
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }

        is_best = False
        if all_loss < best_loss:
            best_loss = all_loss
            is_best = True

        # Save last checkpoint
        torch.save(visionet_checkpoint, '/content/gdrive/My Drive/EVA/EVA5/capstone/visionet_checkpoint.pt')

        if is_best:
            torch.save(visionet_checkpoint, '/content/gdrive/My Drive/EVA/EVA5/capstone/visionet_best.pt')

        ##Save model end

        loss_list.append(all_loss.item())



    ##Yolov3 END
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()


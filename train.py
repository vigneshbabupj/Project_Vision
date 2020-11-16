#Yolo import
import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
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
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
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
#from visualize_utils import *
#from evaluate_utils import *
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

from torch.autograd import Variable


def train(plane_args,yolo_args,midas_args,add_plane_loss,add_yolo_loss,add_midas_loss):

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
    model = VisionNet(cfg,None,config).to(device)

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
    best_fitness = 0.0
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

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        #model.train()
    ## END yolo train setup 

        model.train()

        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(trainloader))

        #optimizer.zero_grad()

        mloss = torch.zeros(4).to(device)  # mean losses

        for i,(plane_data,yolo_data,depth_data) in pbar:

            # print('i:',i)
            # print('plane :',len(plane_data))
            # print('yolo :',len(yolo_data))
            # print('depth :',len(depth_data))

            #yolov3 init start
            imgs, targets, paths, _ = yolo_data

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

            dp_sample = torch.from_numpy(np.asarray(depth_img)).to(device).unsqueeze(0) ######

            midas_inp = dp_sample ####
            #dp_prediction = model.forward()
            #depth init end

            #planercnn init start
            sampleIndex = i
            sample = plane_data

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
            # print('gt_masks',gt_masks.size())
            # print('gt_parameters',gt_parameters.size())
            # print('gt_depth',gt_depth.size())
            # print('extrinsics',extrinsics.size())
            # print('planes',planes.size())
            # print('gt_segmentation',gt_segmentation.size())

            masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()
            input_pair.append({'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0], 'masks': masks, 'mask': gt_masks})
            
            plane_inp = dict(input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='training_detection', use_nms=2, use_refinement='refinement' in options.suffix, return_feature_map=True)

            #planercnn init end

            # All model prediction start

            yolo_out,midas_out,plane_out = model.forward(yolo_inp,midas_inp,plane_inp)

            pred = yolo_out

            dp_prediction = midas_out

            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, feature_map, depth_np_pred = plane_out

            # All model prediction End

            ''' Vignesh : block planercnn
            #Planercnn start
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss = compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters)

            plane_losses += [rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_parameter_loss]

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


            input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'parameters': detection_gt_parameters, 'plane': gt_plane, 'camera': camera})
            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'feature_map': feature_map[0], 'plane_XYZ': plane_XYZ, 'depth_np': depth_np_pred})

            if 'depth' in options.suffix:
                ## Apply supervision on reconstructed depthmap (not used currently)
                if len(detections) > 0:
                    background_mask = torch.clamp(1 - detection_masks.sum(0, keepdim=True), min=0)
                    all_masks = torch.cat([background_mask, detection_masks], dim=0)

                    all_masks = all_masks / all_masks.sum(0, keepdim=True)
                    all_depths = torch.cat([depth_np_pred, plane_XYZ[:, 1]], dim=0)

                    depth_loss = l1LossMask(torch.sum(torch.abs(all_depths[:, 80:560] - gt_depth[:, 80:560]) * all_masks[:, 80:560], dim=0), torch.zeros(config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM).cuda(), (gt_depth[0, 80:560] > 1e-4).float())
                else:
                    depth_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                    pass
                plane_losses.append(depth_loss)                                                
                pass                    
            continue

            if (len(detection_pair[0]['detection']) > 0 and len(detection_pair[0]['detection']) < 30) and 'refine' in options.suffix:
                #use refinement network
                pass
            else:
                plane_losses += [torch.zeros(1).cuda()]
                pass

            ## The warping yolo_loss
            for c in range(1, 2):
                if 'warping' not in options.suffix:
                    break
                continue            

            plane_loss = sum(plane_losses)
            plane_losses = [l.data.item() for l in plane_losses] #train_planercnn.py 331

            # epoch_losses.append(losses)
            # status = str(epoch + 1) + ' yolo_loss: '
            # for l in losses:
            #     status += '%0.5f '%l
            #     continue
            
            # data_iterator.set_description(status)

            # yolo_loss.backward()

            #planercnn END
            Vignesh : block planercnn '''

            ## Midas start


            dp_prediction = (
                            torch.nn.functional.interpolate(
                                dp_prediction.unsqueeze(1),
                                size=dp_img_size[:2],
                                mode="bicubic",
                                align_corners=False,
                            )
                            #.unsqueeze(0)
                            #.cpu()
                            #.numpy()
                            )
            # bits=2

            # depth_min = dp_prediction.min()
            # depth_max = dp_prediction.max()

            # max_val = (2**(8*bits))-1

            # if depth_max - depth_min > np.finfo("float").eps:
            #     depth_pred = max_val * (dp_prediction - depth_min) / (depth_max - depth_min)
            # else:
            #     depth_pred = 0
            #dp_prediction = dp_prediction.unsqueeze(0)
            print('depth_target',depth_target.size)
            depth_target = torch.from_numpy(np.asarray(depth_target)).to(device).type(torch.cuda.FloatTensor)
            print('depth_target',depth_target.size)
            depth_target = (
                            torch.nn.functional.interpolate(
                                depth_target.unsqueeze(1),
                                size=dp_img_size[:2],
                                mode="bicubic",
                                align_corners=False,
                            )
                            #.unsqueeze(0)
                            #.cpu()
                            #.numpy()
                            )

            # print('dp_prediction',dp_prediction.shape)
            # print('depth_target',depth_target.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(dp_prediction.squeeze().cpu().detach().numpy())
            # plt.show()
            # plt.imshow(depth_target.squeeze().cpu().detach().numpy())
            # plt.show()

            depth_pred = Variable( dp_prediction,  requires_grad=True)
            depth_target = Variable( depth_target, requires_grad = False)

            ssim_loss = pytorch_ssim.SSIM() #https://github.com/Po-Hsun-Su/pytorch-ssim
            ssim_out = -ssim_loss(depth_pred,depth_target)


            #print('Depth loss :', ssim_out)
            ## Midas End

            #Yolov3 Start

            # Compute yolo_loss
            yolo_loss, yolo_loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(yolo_loss):
                print('WARNING: non-finite yolo_loss, ending training ', yolo_loss_items)
                return results

            # Scale yolo_loss by nominal batch_size of 64
            yolo_loss *= batch_size / 64


            #all_loss = (add_plane_loss * plane_loss) + (add_yolo_loss * yolo_loss) + (add_midas_loss * ssim_out)
            all_loss = (add_yolo_loss * yolo_loss) + (add_midas_loss * ssim_out)

            print('yolo_loss : ', yolo_loss.item())
            print('ssim_out : ', ssim_out.item())
            print('all_loss :',all_loss.item())

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(all_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                all_loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print batch results
            mloss = (mloss * i + yolo_loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

            ##Yolov3 END
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()


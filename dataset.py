

## InferenceDataset
import numpy as np
import glob
import cv2
import os

from plane_decoder.utils import *
from plane_decoder.plane_dataset import * #from datasets.plane_dataset import *


## Yolo LoadImagesAndLabels

import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from bbox_decoder.utils.utils import xyxy2xywh, xywh2xyxy

from bbox_decoder.utils.datasets import * #Vignesh



help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break



## Midas dataset
import os
import glob
import torch
import depth_decoder.utils as utils
import cv2

from torchvision.transforms import Compose
#from midas.transforms import Resize, NormalizeImage, PrepareForNet

from depth_decoder.transforms import Resize, NormalizeImage, PrepareForNet



class create_data(Dataset):
    #merge LoadImagesAndLabels from yolo and InferenceDataset from planercnn 
    def __init__(self,yolo_params,planercnn_params,midas_params):
        #planercnn_params : self, options, config, image_list, camera, random=False
        #yolo_params : self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
        #        cache_labels=True, cache_images=False, single_cls=False

        ## InferenceDataset start
        

        self.options = planercnn_params['options']
        self.config = planercnn_params['config']
        self.random = planercnn_params['random']
        #self.camera = camera
        #self.imagePaths = image_list
        self.anchors = generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                      self.config.RPN_ANCHOR_RATIOS,
                                                      self.config.BACKBONE_SHAPES,
                                                      self.config.BACKBONE_STRIDES,
                                                      self.config.RPN_ANCHOR_STRIDE)

        if os.path.exists(self.options.customDataFolder + '/camera.txt'):
            self.camera = np.zeros(6)
            with open(self.options.customDataFolder + '/camera.txt', 'r') as f:
                for line in f:
                    values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                    for c in range(6):
                        self.camera[c] = values[c]
                        continue
                    break
                pass
        else:
            self.camera = [filename.replace('.png', '.txt').replace('.jpg', '.txt') for filename in image_list]
            pass
        #return
        ## InferenceDataset END

        ## Yolo LoadImagesAndLabels Start
        path = yolo_params['path']
        img_size = yolo_params.get('img_size',416)
        batch_size = yolo_params.get('batch_size',16)
        augment = yolo_params.get('augment',False)
        hyp = yolo_params.get('hyp',None)
        rect = yolo_params.get('rect',False)
        image_weights = yolo_params.get('image_weights',False)
        cache_labels = yolo_params.get('cache_labels',True)
        cache_images = yolo_params.get('cache_images',False)
        single_cls = yolo_params.get('single_cls',False)

        path = str(Path(path))  # os-agnostic
        assert os.path.isfile(path), 'File not found %s. See %s' % (path, help_url)
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in img_formats]

        self.imagePaths = self.img_files

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = False #self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Read image shapes (wh)
            sp = path.replace('.txt', '.shapes')  # shapefile path
            try:
                with open(sp, 'r') as f:  # read existing shapefile
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == n, 'Shapefile out of sync'
            except:
                s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
                np.savetxt(sp, s, fmt='%g')  # overwrites existing (if any)

            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.shapes = s[i]  # wh
            ar = ar[i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 64.).astype(np.int) * 64

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n
        if cache_labels or image_weights:  # cache labels for faster training
            self.labels = [np.zeros((0, 5))] * n
            extract_bounding_boxes = False
            create_datasubset = False
            pbar = tqdm(self.label_files, desc='Caching labels')
            nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
            for i, file in enumerate(pbar):
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue

                if l.shape[0]:
                    assert l.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (l >= 0).all(), 'negative labels: %s' % file
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                        nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                    if single_cls:
                        l[:, 0] = 0  # force dataset into single-class mode
                    self.labels[i] = l
                    nf += 1  # file found

                    # Create subdataset (a smaller dataset)
                    if create_datasubset and ns < 1E4:
                        if ns == 0:
                            create_folder(path='./datasubset')
                            os.makedirs('./datasubset/images')
                        exclude_classes = 43
                        if exclude_classes not in l[:, 0]:
                            ns += 1
                            # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                            with open('./datasubset/images.txt', 'a') as f:
                                f.write(self.img_files[i] + '\n')

                    # Extract object detection boxes for a second stage classifier
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(l):
                            f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)  # make new output folder

                            b = x[1:] * [w, h, w, h]  # box
                            b[2:] = b[2:].max()  # rectangle to square
                            b[2:] = b[2:] * 1.3 + 30  # pad
                            b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
                else:
                    ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                    # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

                pbar.desc = 'Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                    nf, nm, ne, nd, n)
            assert nf > 0, 'No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

        ## Yolo LoadImagesAndLabels END


        # midas params : inp_path,depth_path
        # midas dataset start

        self.depth_names = [x.replace('images', 'depth_images').replace(os.path.splitext(x)[-1], '.png') for x in self.img_files]
        #self.img_path = inp_path
        #self.depth_path = depth_path
        self.transform = Compose(
                            [
                                Resize(
                                    384,
                                    384,
                                    resize_target=None,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="upper_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC,
                                ),
                                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                PrepareForNet(),
                            ]
                                )

        # midas dataset end


    def __getitem__(self,index):

        ## plane InferenceDataset start
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        if self.random:
            index = np.random.randint(len(self.imagePaths))
        else:
            index = index % len(self.imagePaths)
            pass

        imagePath = self.imagePaths[index]
        image = cv2.imread(imagePath)
        orig_image = image.copy()
        extrinsics = np.eye(4, dtype=np.float32)

        if isinstance(self.camera, list):
            if isinstance(self.camera[index], str):
                camera = np.zeros(6)
                with open(self.camera[index], 'r') as f:
                    for line in f:
                        values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                        for c in range(6):
                            camera[c] = values[c]
                            continue
                        break
                    pass
            else:
                camera = self.camera[index]
                pass
        elif len(self.camera) == 6:
            camera = self.camera
        else:
            assert(False)
            pass

        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        camera[[0, 2, 4]] *= 640.0 / camera[4]        
        camera[[1, 3, 5]] *= 480.0 / camera[5]

        ## The below codes just fill in dummy values for all other data entries which are not used for inference. You can ignore everything except some preprocessing operations on "image".
        depth = np.zeros((self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM), dtype=np.float32)
        segmentation = np.zeros((self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM), dtype=np.int32)


        planes = np.zeros((segmentation.max() + 1, 3))

        instance_masks = []
        class_ids = []
        parameters = []

        if len(planes) > 0:
            if 'joint' in self.config.ANCHOR_TYPE:
                distances = np.linalg.norm(np.expand_dims(planes, 1) - self.config.ANCHOR_PLANES, axis=-1)
                plane_anchors = distances.argmin(-1)
            elif self.config.ANCHOR_TYPE == 'Nd':
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
                normal_anchors = distances_N.argmin(-1)
                distances_d = np.abs(np.expand_dims(plane_offsets, -1) - self.config.ANCHOR_OFFSETS)
                offset_anchors = distances_d.argmin(-1)
            elif self.config.ANCHOR_TYPE in ['normal', 'patch']:
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.maximum(np.expand_dims(plane_offsets, axis=-1), 1e-4)
                distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
                normal_anchors = distances_N.argmin(-1)
            elif self.config.ANCHOR_TYPE == 'normal_none':
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                pass
            pass

        for planeIndex, plane in enumerate(planes):
            m = segmentation == planeIndex
            if m.sum() < 1:
                continue
            instance_masks.append(m)
            if self.config.ANCHOR_TYPE == 'none':
                class_ids.append(1)
                parameters.append(np.concatenate([plane, np.zeros(1)], axis=0))
            elif 'joint' in self.config.ANCHOR_TYPE:
                class_ids.append(plane_anchors[planeIndex] + 1)
                residual = plane - self.config.ANCHOR_PLANES[plane_anchors[planeIndex]]
                parameters.append(np.concatenate([residual, np.array([0, plane_info[planeIndex][-1]])], axis=0))
            elif self.config.ANCHOR_TYPE == 'Nd':
                class_ids.append(normal_anchors[planeIndex] * len(self.config.ANCHOR_OFFSETS) + offset_anchors[planeIndex] + 1)
                normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
                offset = plane_offsets[planeIndex] - self.config.ANCHOR_OFFSETS[offset_anchors[planeIndex]]
                parameters.append(np.concatenate([normal, np.array([offset])], axis=0))
            elif self.config.ANCHOR_TYPE == 'normal':
                class_ids.append(normal_anchors[planeIndex] + 1)
                normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
                parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))
            elif self.config.ANCHOR_TYPE == 'normal_none':
                class_ids.append(1)
                normal = plane_normals[planeIndex]
                parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))
            else:
                assert(False)
                pass
            continue

        parameters = np.array(parameters)
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)

        image, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters = load_image_gt(self.config, index, image, depth, mask, class_ids, parameters, augment=False)
        ## RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        ## If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
            gt_parameters = gt_parameters[ids]
            pass

        ## Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        image = mold_image(image.astype(np.float32), self.config)

        depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0).astype(np.float32)
        segmentation = np.concatenate([np.full((80, 640), fill_value=-1), segmentation, np.full((80, 640), fill_value=-1)], axis=0).astype(np.float32)

        data_pair = [image.transpose((2, 0, 1)).astype(np.float32), image_metas, rpn_match.astype(np.int32), rpn_bbox.astype(np.float32), gt_class_ids.astype(np.int32), gt_boxes.astype(np.float32), gt_masks.transpose((2, 0, 1)).astype(np.float32), gt_parameters[:, :-1].astype(np.float32), depth.astype(np.float32), extrinsics.astype(np.float32), planes.astype(np.float32), segmentation.astype(np.int64), gt_parameters[:, -1].astype(np.int32)]
        data_pair = data_pair + data_pair

        data_pair.append(np.zeros(7, np.float32))

        data_pair.append(planes)
        data_pair.append(planes)
        data_pair.append(np.zeros((len(planes), len(planes))))
        data_pair.append(camera.astype(np.float32))

        #return data_pair
        ## InferenceDataset END

        ## Yolo LoadImagesAndLabels START

        #if self.image_weights:
        #    index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x is not None and x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        yolo_item=[torch.from_numpy(img), labels_out, self.img_files[index], shapes]

        ## Yolo LoadImagesAndLabels END

        # midas dataset start

        img_name = self.img_files[index] #vig
        depth_name = self.depth_names[index]

        img_ip = utils.read_image(img_name)
        img_input = self.transform({"image": img_ip})["image"]


        #print('depth_name',depth_name)
        depth_img = cv2.imread(depth_name)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        
        #print('depth_img',depth_img.shape)

        data = [img_input,depth_img]

        # midas dataset end
        #print('plane:',len(data_pair))
        #print('yolo:',len(yolo_item))
        #print('depth:',len(data))

        return data_pair,yolo_item,data

    @staticmethod
    def collate_fn(batch):

        # print('len batch',len(batch))

        plane_item,yolo_item,dp_item = zip(*batch)
        up_plane=[]
        up_depth=[]
        if len(batch) > 1:
            for i in range(len(batch)):
                up_plane.append(plane_item[i][0])
                up_depth.append(dp_item[i][0])
        else:
            up_plane = plane_item[0]
            up_depth = dp_item[0]


        for p in range(31):
            up_plane[p] = torch.from_numpy(up_plane[p]).unsqueeze(0)


        # print('plane item:',len(plane_item[0]))
        # print('yolo item:',len(yolo_item[0]))
        # print('depth item:',len(dp_item[0]))

        img, label, path, shapes = zip(*yolo_item)  # transposed

        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        yolo_item = [torch.stack(img, 0), torch.cat(label, 0), path, shapes]


        return up_plane,yolo_item,up_depth

    def __len__(self):
        return len(self.img_files)
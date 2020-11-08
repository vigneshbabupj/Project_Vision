
from model import *

from dataset import *


def dataloader():
	
	train_dataset = create_data(yolo_params,planercnn_params,midas_params)

	test_dataset = create_data(yolo_params,planercnn_params,midas_params)


	#yolo data loader

	# Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True )
                                             #collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(test_dataset
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True )
                                             #collate_fn=dataset.collate_fn)

    return trainloader,testloader

def train():

	

	model = VisionNet().to(device)

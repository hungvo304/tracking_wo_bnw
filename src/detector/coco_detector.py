import torchvision
import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    torch.save(model.state_dict(),
               '../../output/faster_rcnn_fpn_training_mot_17/coco_model.model')

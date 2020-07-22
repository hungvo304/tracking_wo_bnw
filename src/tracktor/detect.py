import torch
import cv2
import os
import glob
import numpy as np

from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from frcnn_fpn import FRCNN_FPN
from PIL import Image
from natsort import natsorted

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def main():
    obj_detect = FRCNN_FPN(num_classes=91)
    # obj_detect.load_state_dict(torch.load(
    #     '../../output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model'))
    obj_detect.load_state_dict(torch.load(
        '../../output/faster_rcnn_fpn_training_mot_17/coco_model.model'))
    obj_detect.eval()

    FRAMES_ROOT = '/mmlabstorage/workingspace/InstaceSearch/hungvq/source/src/uit/mmlab/ins/objecttracking/tracking_wo_bnw/data/9_video_test/test/'
    SAVE_ROOT = '/mmlabstorage/workingspace/InstaceSearch/hungvq/source/src/uit/mmlab/ins/objecttracking/tracking_wo_bnw/data/9_video_test_label/test'

    # video_dir = glob.glob(os.path.join(FRAMES_ROOT, '*'))
    # video_dir = [os.path.join(
    #     FRAMES_ROOT, 'NKKN-VoThiSau 2017-07-18_08_00_00_000')]
    video_dir = [os.path.join(
        FRAMES_ROOT, 'test_sample')]
    for subdir in video_dir:

        dirname = os.path.basename(subdir) + '_coco'
        frames_root = os.path.join(subdir, 'img1')

        save_path = os.path.join(SAVE_ROOT, dirname)
        os.makedirs(save_path, exist_ok=True)
        det_dir = os.path.join(save_path, 'det')
        os.makedirs(det_dir, exist_ok=True)

        det_txt = os.path.join(det_dir, 'det.txt')
        if os.path.exists(det_txt):
            continue
        with open(det_txt, 'w') as f:

            img_list = []
            for frame_path in natsorted(glob.glob(os.path.join(frames_root, '*.jpg'))):
                frame_name, _ = os.path.splitext(os.path.basename(frame_path))
                frame_id = int(frame_name)

                img = Image.open(frame_path).convert('RGB')
                img = T.ToTensor()(img)
                img = img.unsqueeze(1)
                vis_img = cv2.imread(frame_path)

                bboxes, scores = obj_detect.detect(img)

                for bbox, score in zip(bboxes, scores):
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1 + 1
                    height = y2 - y1 + 1

                    line = ','.join((str(frame_id), '-1', str(x1.item()), str(y1.item()),
                                     str(width.item()), str(height.item()), str(score.item())))
                    print(line)
                    f.write(line + '\n')

                # For visualization purpose
                # for bbox in bboxes:
                # 	x1, y1, x2, y2 = [int(c) for c in bbox]

                # 	cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # cv2.imshow('fig', vis_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

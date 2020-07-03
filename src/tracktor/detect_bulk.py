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


class SequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.imgs_path = natsorted(glob.glob(os.path.join(root_dir, '*.jpg')))

    def __len__(self):
        return len(imgs_path)

    def __getitem__(self, idx):
        img = Image.open(frame_path).convert('RGB')
        img = T.ToTensor()(img)


def main():
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(
        '../../output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model'))
    obj_detect.eval()

    FRAMES_ROOT = '/mmlabstorage/workingspace/InstaceSearch/hungvq/source/src/uit/mmlab/ins/objecttracking/tracking_wo_bnw/data/9_video_test/test/'
    SAVE_ROOT = '/mmlabstorage/workingspace/InstaceSearch/hungvq/source/src/uit/mmlab/ins/objecttracking/tracking_wo_bnw/data/9_video_test_label/test'

    for subdir in glob.glob(os.path.join(FRAMES_ROOT, '*')):
        dirname = os.path.basename(subdir)
        frames_root = os.path.join(subdir, 'img1')

        save_path = os.path.join(SAVE_ROOT, dirname)
        os.makedirs(save_path, exist_ok=True)
        det_dir = os.path.join(save_path, 'det')
        os.makedirs(det_dir, exist_ok=True)

        det_txt = os.path.join(det_dir, 'det_test.txt')
        if os.path.exists(det_txt):
            continue
        with open(det_txt, 'w') as f:

            img_list = []
            for frame_path in natsorted(glob.glob(os.path.join(frames_root, '*.jpg')))[:5]:
                frame_name, _ = os.path.splitext(os.path.basename(frame_path))
                frame_id = int(frame_name) + 1  # adding one temporarily

                img = Image.open(frame_path).convert('RGB')
                img = np.array(img)

                print(type(img))
                # img = T.ToTensor()(img)
                # img = img.unsqueeze(1)
                img_list.append(img)
                # vis_img = cv2.imread(frame_path)

                # bboxes, scores = obj_detect.detect(img)

            img_list = np.array(img_list)
            img_list = img_list.transpose((0, 3, 1, 2))
            print(img_list.shape)
            imgs = torch.from_numpy(img_list)
            bboxes, scores = obj_detect.detect(imgs)
            print(bboxes, scores)

            # 	for bbox, score in zip(bboxes, scores):
            # 		x1, y1, x2, y2 = bbox
            # 		width = x2 - x1 + 1
            # 		height = y2 - y1 + 1

            # 		line = ','.join((str(frame_id), '-1', str(x1.item()), str(y1.item()), str(width.item()), str(height.item()), str(score.item())))
            # 		print(line)
            # 		f.write(line + '\n')

            # For visualization purpose
            # for bbox in bboxes:
            # 	x1, y1, x2, y2 = [int(c) for c in bbox]

            # 	cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # cv2.imshow('fig', vis_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

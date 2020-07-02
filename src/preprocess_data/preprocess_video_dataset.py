import configparser
import os
import cv2
import glob
import sys
import subprocess


def getTotalFrame(video_path, save_path, is_save=True):
    # calculate the total frames of a video
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        if is_save:
            cv2.imwrite(
                os.path.join(save_path, str(cnt).zfill(6) + '.jpg'), frame)
        cnt += 1
    return cnt


def getDuration(cap):
    # duration = total_frame / FPS
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)


def main():
    DATASET_ROOT = '/mmlabstorage/workingspace/InstaceSearch/hungvq/source/src/uit/mmlab/ins/objecttracking/tracking_wo_bnw/data/9_video_test'
    videos_root = os.path.join(DATASET_ROOT, 'raw_videos')

    video_exts = ['mkv', 'mov', 'asf']
    videos_path = []
    for ext in video_exts:
        video_path = os.path.join(videos_root, f'*.{ext}')
        videos_path.extend(glob.glob(video_path))

    SAVE_ROOT = os.path.join(DATASET_ROOT, 'test')
    os.makedirs(SAVE_ROOT, exist_ok=True)
    for video_path in videos_path:
        print(f'Processing video: {video_path}')
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        video_save_root = os.path.join(SAVE_ROOT, video_name)
        os.makedirs(video_save_root, exist_ok=True)

        img_dir = 'img1'
        frames_save_root = os.path.join(video_save_root, img_dir)
        os.makedirs(frames_save_root, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        num_frames = getTotalFrame(
            video_path, save_path=frames_save_root, is_save=True)
        duration = getDuration(cap)
        fps = num_frames/duration
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        im_ext = '.jpg'

        config = configparser.ConfigParser()
        config.add_section('Sequence')
        config.set('Sequence', 'name', video_name)
        config.set('Sequence', 'imDir', img_dir)
        config.set('Sequence', 'frameRate', f'{round(fps)}')
        config.set('Sequence', 'seqLength', f'{num_frames}')
        config.set('Sequence', 'imWidth', f'{round(width)}')
        config.set('Sequence', 'imHeight', f'{round(height)}')
        config.set('Sequence', 'imExt', im_ext)

        # print(f'Num Frames: {num_frames}')
        # print(f'Duration: {duration}')
        # print(f'FPS: {fps}')
        # print(f'Height: {height}')
        # print(f'Width: {width}')

        cfg_file = os.path.join(video_save_root, 'seqinfo.ini')
        with open(cfg_file, 'w') as f:
            config.write(f)

        cap.release()


if __name__ == '__main__':
    main()

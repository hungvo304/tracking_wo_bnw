import os
import glob

from natsort import natsorted


if __name__ == '__main__':
	frame_root = '/mmlabstorage/workingspace/InstaceSearch/hungvq/source/src/uit/mmlab/ins/objecttracking/tracking_wo_bnw/data/9_video_test/test/NKKN-VoThiSau 2017-07-18_08_00_00_000/img1'

	for frame_path in natsorted(glob.glob(os.path.join(frame_root, '*.jpg')), reverse=True):
		old_frame_name, ext = os.path.splitext(os.path.basename(frame_path))

		new_frame_name = str(int(old_frame_name)+1).zfill(6)

		print(f'{old_frame_name}{ext}', f'{new_frame_name}{ext}')
		os.rename(os.path.join(frame_root, f'{old_frame_name}{ext}'), os.path.join(frame_root, f'{new_frame_name}{ext}'))


import os
# import scipy.io
import mat73
import numpy as np
import cv2
from tqdm import tqdm
import argparse

global_path = '/mnt/fast/nobackup/scratch4weeks/jz01019/Teacher-Student/'
feature_path = '/mnt/fast/nobackup/scratch4weeks/jz01019/synthetic/feature/'

def gccphat_convert(seq_name):
    path = os.path.join(global_path, 'tracker/gcf/output_sum')

    file_path = os.path.join(path, seq_name)
    mat_contents = mat73.loadmat(file_path)
    gccphat = mat_contents['output']
    X_g, Y_g = 141, 201
    timestep = mat_contents['output'].shape[0]
    gccphat_2d = np.zeros((timestep, X_g, Y_g))
    for t in range(timestep):
        for x in range(X_g):
            for y in range(Y_g):
                gccphat_2d[t, x, y] = gccphat[t, x * Y_g + y]
    np.save(os.path.join(global_path, 'tracker/preprocess/features', seq_name.split('.')[0] + '.npy'), gccphat_2d)


def gccphat_gt(data_split):
    file_path = os.path.join(global_path, 'single_people_sequence_set/')
    data_split_path = os.path.join(file_path, data_split)
    data_split_folders = os.listdir(data_split_path)
    
    for data_split_file in data_split_folders:
        if '_seg' in data_split_file:
            continue
     
        print('Processing ' + data_split_file)
        cam = data_split_file.split('cam')[1][0]
        seq = data_split_file.split('_')[0]
        audio_file = data_split_file.split('_')[0]
        gccphat_path = os.path.join(global_path, 'tracker/preprocess/features', audio_file + '.npy')
        gccphat = np.load(gccphat_path)
        nb_images = len(os.listdir(os.path.join(file_path, data_split, data_split_file)))
        a = min(nb_images, gccphat.shape[0])
        for t in tqdm(range(a)):
            output = []
            img = cv2.imread(os.path.join(file_path, data_split, data_split_file, str(t + 1) + '.jpg'))
            seg = np.load(os.path.join(file_path, data_split, data_split_file + '_seg', str(t + 1) + '.npy'))
            with open(os.path.join(global_path, 'tracker/gt/', data_split, 'facebb_AV16.3_' + data_split_file.replace('cam', 'C') + '_DSFD.txt')) as f:    
                gt_or_not = False
                gt_x, gt_y = 0, 0
                for line in f.readlines():
                    contents = line.split(' ')
                    if int(contents[0]) == t + 1:
                        gt_x = float(contents[2]) + 0.5 * (float(contents[4]) - float(contents[2]))
                        gt_y = float(contents[3]) + 0.75 * (float(contents[5]) - float(contents[3]))
                        gt_or_not = True
                        break
                if gt_or_not:
                    output = (gccphat[t], np.array(img), seg, (gt_x, gt_y), data_split_file)
                else:
                    continue
            # create a folder named with seq, time and cam,
            if not os.path.exists(os.path.join(feature_path, data_split, seq + '_cam' + str(cam) + '_t_' + str(t + 1))):
                os.makedirs(os.path.join(feature_path, data_split, seq + '_cam' + str(cam) + '_t_' + str(t + 1)))
            # save the output to the created folder
            np.save(os.path.join(feature_path, data_split, seq + '_cam' + str(cam) + '_t_' + str(t + 1), 'output.npy'), output)

            
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', default='train',
                        help='which dataset: train, eval and test')
   
    args = parser.parse_args()
    data_split = args.data_split
   
    gccphat_gt(data_split)
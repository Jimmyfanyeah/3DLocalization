# Import modules and libraries
import torch
from shutil import copy2
import csv
import re
import pickle5 as pickle
import numpy as np
import time
import os
from collections import defaultdict
import torch.distributed as dist
# self-defined functions
from utils.data import dataloader
from utils.postprocess import Postprocess, Postprocess_v0
from utils.loss import calculate_loss_v2 as calculate_loss
from utils.helper import print_log, load_labels


def test_model(opt, cnn, postprocess_params, path_model, path_save, exp_imgs_path=None, log=None):

    path_results = re.compile('(.+?)/'+path_model.split('/')[-1]).findall(path_model)[0]

    # load assumed setup parameters
    path_params_pickle = os.path.join(path_results,'setup_params.pickle')
    with open(path_params_pickle, 'rb') as handle:
        setup_params = pickle.load(handle)

    if opt.rank==0:
        print_log('setup_params - when train:',log)
        for key in setup_params.keys():
            if not key == 'partition':
                print_log('{}: {}'.format(key,setup_params[key]),log)

    device, setup_params['device'] = 'cuda', 'cuda'
    calc_loss = calculate_loss(setup_params)

    try:
        # read imgs in opt.test_id_loc
        img_names = []
        with open(opt.test_id_loc,'r') as file:
            for line in file:
                img_names.append(line[:-1])
        if opt.rank==0:
            print_log('test chosen {} images in {}'.format(len(img_names),exp_imgs_path), log)
            print_log(img_names,log)
    except:
        # read all imgs in exp_imgs_path
        img_names = [x[2:-4] for x in os.listdir(exp_imgs_path) if 'mat' in x and 'im' in x]

    # if label exist, load labels
    label_existence = os.path.exists(os.path.join(exp_imgs_path,'label.txt'))
    if label_existence:
        copy2(os.path.join(exp_imgs_path,'label.txt'),os.path.join(path_save,'label.txt'))
        if opt.rank == 0:
            print_log('label exists!',log)
        labels = load_labels(os.path.join(exp_imgs_path,'label.txt'))

        params_test = {'batch_size': 1, 'shuffle': False, 'partition':img_names}
        exp_generator = dataloader(exp_imgs_path, labels, params_test, setup_params, opt)

    if opt.post_pro == 1:
        # post-processing, restore points loc to original size, do cluster, get final result
        thresh, radius = postprocess_params['thresh'], postprocess_params['radius']
        postprocessing_module = Postprocess(thresh, radius, setup_params)
    else:
        # only restore points loc to original size, but not cluster
        postprocessing_module = Postprocess_v0(setup_params)

    # time the entire dataset analysis
    tall_start = time.time()

    # process all experimental images
    cnn.eval()
    metric = defaultdict(float)
    metrics = defaultdict(float)
    results = np.array(['frame', 'x', 'y', 'z', 'intensity'])
    results_bol = np.array(['frame', 'x', 'y', 'z', 'intensity'])
    pt = open(os.path.join(path_save,'loss_{}.txt'.format(opt.rank)),'w')
    with torch.set_grad_enabled(False):
        for im_ind, (exp_im_tensor, target, target_2d, fileid) in enumerate(exp_generator):

            exp_im_tensor = exp_im_tensor.to(device)
            target = target.to(device)
            target_2d = target_2d.to(device)

            pred_volume = cnn(exp_im_tensor)

            loss = calc_loss(pred_volume, target, target_2d, metric, metrics)
            # loss = calc_loss(pred_volume, normgrid, target, target_2d, metric, metrics)
            pt.write('{},{:.4f},{:.4f},{:.4f}\n'.format(fileid[0], metric['dice'], metric['kde'], metric['mse_2d']))

            # post-process result to get the xyz coordinates and their confidence
            xyz_rec, conf_rec, xyz_bool = postprocessing_module(pred_volume)

            # if prediction is empty then set number for found emitters to 0
            # otherwise generate the frame column and append results for saving
            if xyz_rec is None:
                nemitters = 0
            else:
                nemitters = xyz_rec.shape[0]
                frm_rec = (int(img_names[im_ind]))*np.ones(nemitters)
                results = np.vstack((results, np.column_stack((frm_rec, xyz_rec, conf_rec))))
                results_bol = np.vstack((results_bol, np.column_stack((frm_rec, xyz_bool, conf_rec))))

            if opt.rank==0:
                if label_existence:
                    xyz_gt = np.squeeze(labels[fileid[0]])
                    print_log('{} Img{} found {:d}/{} emitters'.format(im_ind, fileid[0],nemitters,len(xyz_gt)),log)
                else:
                    print_log('{} Img{} found {:d} emitters'.format(im_ind, fileid[0], nemitters),log)

    # print the time it took for the entire analysis
    tall_end = time.time() - tall_start
    if opt.rank==0:
        print_log('=' * 50,log, arrow=False)
        print_log('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60), log, arrow=False)
        print_log('=' * 50,log, arrow=False)

    # write the results to a csv file named "localizations.csv" under the exp img folder
    row_list = results.tolist()
    with open(os.path.join(path_save,'loc.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)
    pt.close()

    row_list = results_bol.tolist()
    with open(os.path.join(path_save,'loc_bool.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)
    pt.close()

    dist.barrier()
    if opt.rank == 0:
        pt = open(os.path.join(path_save,'loss.txt'),'w')
        for i in range(opt.world_size):
            with open(os.path.join(path_save,'loss_{}.txt'.format(i)),'r') as tmpPt:
                pt.writelines(tmpPt.readlines())
        pt.close()

    return xyz_rec, conf_rec

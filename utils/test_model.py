# official modules
from shutil import copy2
import numpy as np
import time
import os
from collections import defaultdict
import csv
import pandas as pd
import torch
import torch.distributed as dist
# self-defined modules
from utils.loss import calculate_loss
from utils.data import dataloader
from utils.postprocess import Postprocess
from utils.helper import print_log, load_labels, print_time, print_metric_format


def test_model(opt, cnn, log=None):

    imgs_path = opt.data_path
    save_path = opt.save_path

    device = 'cuda'

    # loss function
    loss_type = ['loss']
    if opt.extra_loss: # None or string
        extra_loss = opt.extra_loss.split('_')
        extra_weight = [float(n) for n in opt.weight.split('_')]
        loss_type = loss_type + extra_loss
        if not len(extra_loss) == len(extra_weight):
            raise Exception(f'Input {len(extra_loss)} weight with {len(extra_weight)} extra loss')
    print_log(f'[INFO==>] Loss types: {loss_type}',log)
    calc_loss = calculate_loss(opt,loss_type,extra_weight)


    if opt.test_id_loc:
        # read imgs with id in opt.test_id_loc if specific img list for test are chosen
        imgList = []
        with open(opt.test_id_loc,'r') as file:
            for line in file:
                imgList.append(line[:-1])
        if opt.rank==0:
            all_imgs = [x[2:-4] for x in os.listdir(os.path.join(imgs_path,'noise')) if 'mat' in x and 'im' in x]
            print_log(f'[INFO==>] test on {len(imgList)}/{len(all_imgs)} samples in {imgs_path}', log)
            print_log(imgList,log)
    else:
        # imgList = [x[2:-4] for x in os.listdir(os.path.join(imgs_path,'noise')) if 'mat' in x and 'im' in x]
        imgList = [x[2:-4] for x in os.listdir(os.path.join(imgs_path,'noise')) if 'mat' in x and 'im' in x]
        imgList.sort()
        if opt.rank==0:
            print_log('[INFO==>] Test {} samples in {}'.format(len(imgList),imgs_path), log)

    # load label file if label exist
    label_existence = os.path.exists(os.path.join(imgs_path,'label.txt'))
    if label_existence:
        copy2(os.path.join(imgs_path,'label.txt'),os.path.join(save_path,'label.txt'))
        if opt.rank == 0:
            print_log('[INFO==>] Label exists',log)
        labels = load_labels(os.path.join(imgs_path,'label.txt'))

        params_test = {'batch_size':1, 'shuffle':False, 'partition':imgList}
        data_generator = dataloader(imgs_path, labels, params_test, opt)

    postpro_module = Postprocess(opt)

    # time the entire dataset analysis
    tall_start = time.time()

    # process all experimental images
    cnn.eval()
    metric, metrics = defaultdict(float), defaultdict(float)
    results = np.array(['frame', 'x', 'y', 'z', 'intensity'])
    results_bol = np.array(['frame', 'x', 'y', 'z', 'intensity'])
    pt = open(os.path.join(save_path,'loss_{}.txt'.format(opt.rank)),'w')
    with torch.set_grad_enabled(False):
        for im_ind, (im_tensor, target, target_2d, fileid) in enumerate(data_generator):

            im_tensor = im_tensor.to(device)
            target = target.to(device)
            target_2d = target_2d.to(device)

            pred_volume = cnn(im_tensor)

            loss = calc_loss(pred_volume, target, target_2d, metric, metrics)
            pt.write('{},{}\n'.format(fileid[0], print_metric_format(metric)))

            # post-process result to get the xyz coordinates and their confidence
            xyz_rec, conf_rec, xyz_bool = postpro_module(pred_volume)

            # if prediction is empty then set number for found emitters to 0
            # otherwise generate the frame column and append results for saving
            if xyz_rec is None:
                nemitters = 0
            else:
                nemitters = xyz_rec.shape[0]
                frm_rec = (int(fileid[0]))*np.ones(nemitters)
                results = np.vstack((results, np.column_stack((frm_rec, xyz_rec, conf_rec))))
                results_bol = np.vstack((results_bol, np.column_stack((frm_rec, xyz_bool, conf_rec))))

            if opt.rank==0:
                if label_existence:
                    xyz_gt = np.squeeze(labels[fileid[0]])
                    print_log('Test sample {} Img{} found {:d}/{} emitters'.format(im_ind, fileid[0],nemitters,len(xyz_gt)),log)
                else:
                    print_log('Test sample {} Img{} found {:d} emitters'.format(im_ind, fileid[0], nemitters),log)

    # print the time it took for the entire analysis
    tall_end = time.time() - tall_start
    if opt.rank==0:
        print_log('=' * 50,log, arrow=False)
        print_log('Analysis complete in {}'.format(print_time(tall_end)), log)
        print_log('=' * 50,log, arrow=False)

    # write the results to a csv file named "loc.csv" under the infer result save folder
    row_list = results.tolist()
    with open(os.path.join(save_path,f'loc_{opt.rank}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)
    pt.close()

    row_list = results_bol.tolist()
    with open(os.path.join(save_path,f'loc_bool_{opt.rank}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)
    pt.close()

    # dist.barrier()
    # Merge loc files
    if opt.rank == 0:
        arr = []
        for i in range(opt.world_size):
            file = os.path.join(save_path,f'loc_{i}.csv')
            arr.append(pd.read_csv(file))
        # writer = pd.ExcelWriter(os.path.join(save_path,'loc.csv'))
        pd.concat(arr, ignore_index=True).to_csv(os.path.join(save_path,'loc.csv'))
        # writer.save()

    if opt.rank == 0:
        pt = open(os.path.join(save_path,'loss.txt'),'w')
        for i in range(opt.world_size):
            with open(os.path.join(save_path,'loss_{}.txt'.format(i)),'r') as tmpPt:
                pt.writelines(tmpPt.readlines())
        pt.close()

    return xyz_rec, conf_rec

import os
import numpy as np

def match_result(gt_path,pred_path,id_txt=None,criterion_xy=2,criterion_z=1):

    # gt_path = 'D:/OneDrive - City University of Hong Kong/microscope/results/test_images_5'
    # pred_path = 'D:/OneDrive - City University of Hong Kong/microscope/results/test_images_5'
    # criterion_xy = 2
    # criterion_z = 1

    # load groundtruth localization
    gt_raw = open(os.path.join(gt_path,'label.txt'),'r')
    gt_all = []
    for line in gt_raw:
        tmp = line[:-1].split(' ')
        while '' in tmp: 
            tmp.remove('')
        gt_all.append(tmp)
    gt_all = np.asarray(gt_all)
    gt_all = gt_all.astype(np.float)

    # load predicted 3d localization
    csvFile = open(os.path.join(pred_path,'loc.csv'), "r")
    pred_all = []
    for line in csvFile:
        tmp = line[:-1].split(',')
        pred_all.append(tmp)
    pred_all = np.asarray(pred_all)
    pred_all = pred_all[1:].astype(np.float)

    # compute Precision and Recall
    TP = 0
    FP = 0
    FN = 0

    try:
        loop_list = []
        with open(id_txt,'r') as file:
            for line in file:
                loop_list.append(int(line[:-1]))
    except:
        loop_list = range(int(min(gt_all[:,0])),int(max(gt_all[:,0]))+1)

    result_file = open(os.path.join(pred_path,'pr.txt'),'w')
    for ii in loop_list:
        used_pred = []
        gt = gt_all[gt_all[:,0]==ii]
        pred = pred_all[pred_all[:,0]==ii]

        TP_now = 0
        FP_now = 0
        FN_now = 0
        for jj in range(len(gt)):
            # xy_distance = []
            # z_distance = []
            s = []
            for kk in range(len(pred)):
                x_d = abs(gt[jj,1] - pred[kk,1])
                y_d = abs(gt[jj,2] - pred[kk,2])
                z_d = abs(gt[jj,3] - pred[kk,3])
                if x_d < criterion_xy and y_d < criterion_xy and z_d < criterion_z:
                    s.append(str(kk))

            if len(s)>1:
                # print('{} --> {}'.format(jj,s))
                # result_file.write('{} --> {}'.format(jj,s))
                # result_file.write('\n')
                for item in s:
                    if item in used_pred:
                        print('pred {} used more than 1 times!'.format(item))
                        result_file.write('pred {} used more than 1 times!'.format(item))
                        result_file.write('\n')
                    else:
                        used_pred.append(item)

                TP = TP + 1
                TP_now = TP_now + 1
            elif len(s)==1: 
                # print('{} = {}'.format(jj,s[0]))
                # result_file.write('{} = {}'.format(jj,s[0]))
                # result_file.write('\n')
                if s[0] in used_pred:
                    print('pred {} used more than 1 times!'.format(s[0]))
                    result_file.write('pred {} used more than 1 times!'.format(s[0]))
                    result_file.write('\n')
                else:
                    used_pred.append(s[0])

                TP = TP + 1
                TP_now = TP_now + 1
            elif len(s)==0:
                FN = FN + 1
                FN_now = FN_now + 1
        FP = FP + max(len(pred) - len(set(used_pred)), 0)
        FP_now = max(len(pred) - len(set(used_pred)), 0)
        print('img{} TP={}, FP={}, FN={}'.format(ii,TP_now,FP_now,FN_now))
        result_file.write('img{} TP={}, FP={}, FN={}'.format(ii,TP_now,FP_now,FN_now))
        result_file.write('\n')

    print('total TP={}, FP={}, FN={}'.format(TP,FP,FN))
    result_file.write('total TP={}, FP={}, FN={}'.format(TP,FP,FN))
    result_file.write('\n')
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    # print('precision: {}, recall: {}'.format(precision, recall))
    result_file.write('precision: {}, recall: {}'.format(precision, recall))
    result_file.close()
    return precision, recall


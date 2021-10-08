# import pickle
import pickle5 as pickle
import numpy as np
import os
import time
import math
import re
from collections import defaultdict
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from utils.loss import calculate_loss_v2 as calculate_loss
from utils.helper import save_checkpoint, print_log, print_metrics, print_time

def train_model(cnn,optimizer,scheduler,device,training_generator,validation_generator,log,logger,opt,setup_params):

    learning_results = defaultdict(list)
    max_epoch = setup_params['maxEpoch']
    steps_train = math.ceil(setup_params['ntrain']/setup_params['batch_size']/opt.world_size)
    steps_val = math.ceil(setup_params['nvalid']/setup_params['batch_size']/opt.world_size)
    params_valid = {'batch_size': setup_params['batch_size'], 'shuffle': False}
    path_save = setup_params['result_save_path']

    # loss function
    calc_loss = calculate_loss(setup_params)

    scaler = GradScaler()

    if setup_params['resume_training']:
        checkpoint = setup_params['checkpoint_path']
        start_epoch = torch.load(checkpoint)['epoch']
        end_epoch = max_epoch

        # load all recorded metrics in checkpoint
        tmp_path = re.compile('(.+?)/'+checkpoint.split('/')[-1]).findall(checkpoint)[0]
        # re.compile.findall remove checkpoint.split('/')[-1], eg. checkpoint='a/b/c', tmp_path='a/b'
        path_learning_results = os.path.join(tmp_path, 'learning_results.pickle')
        with open(path_learning_results, 'rb') as handle:
            learning_results = pickle.load(handle)
        if opt.rank==0:
            print_log('total epoch in checkpoint: {}, continue from epoch {} and load loss\n'.format(len(learning_results['train_loss']),start_epoch),log)
        for key in learning_results.keys():
            try:
                learning_results[key] = learning_results[key][:start_epoch]
            except:
                pass
        # visualize results of checkpoint in tensorboard
        if opt.rank==0:
            for ii in range(len(learning_results['train_loss'])):
                logger.scalars_summary('All/Loss',{'train':learning_results['train_loss'][ii], 'val':learning_results['valid_loss'][ii]}, ii+1)
                try:
                    logger.scalars_summary('All/MSE3D',{'train':learning_results['train_mse3d'][ii], 'val':learning_results['valid_mse3d'][ii]}, ii+1)
                    # logger.scalars_summary('All/MSE2D',{'train':learning_results['train_mse2d'][ii], 'val':learning_results['valid_mse2d'][ii]}, ii+1)
                    logger.scalars_summary('All/Dice',{'train':learning_results['train_dice'][ii], 'val':learning_results['valid_dice'][ii]}, ii+1)
                except: pass
                logger.scalar_summary('Other/MaxOut', learning_results['max_valid'][ii], ii+1)
                logger.scalar_summary('Other/MaxOutSum', learning_results['sum_valid'][ii], ii+1)

        # initialize validation set best loss and jaccard
        valid_loss_prev = np.min(learning_results['valid_loss'])

    else:
        # start from scratch
        start_epoch, end_epoch = 0, max_epoch
        learning_results = {'train_loss': [], 'valid_loss': [], 'train_dice': [], 'train_mse3d': [], \
                             'valid_dice': [], 'valid_mse3d': [], 'max_valid': [], 'sum_valid': [],  \
                            'steps_per_epoch': steps_train}
        valid_loss_prev = float('Inf')

    # starting time of training
    train_start_time = time.time()
    not_improve = 0

    for epoch in np.arange(start_epoch, end_epoch):

        # starting time of current epoch
        epoch_start_time = time.time()

        if opt.rank == 0:
            print_log('Epoch {}/{}'.format(epoch+1, end_epoch), log)
            print_log('-' * 10, log)
            print_log(time.ctime(time.time()), log)
            for param_group in optimizer.param_groups:
                print_log('lr '+str(param_group['lr']),log)

        # training phase
        cnn.train()
        metric = defaultdict(float)
        metrics = defaultdict(float)

        with torch.set_grad_enabled(True):
            for batch_ind, (inputs, targets, target_ims, fileids) in enumerate(training_generator):

                inputs = inputs.to(device)
                targets = targets.to(device)
                target_ims = target_ims.to(device)

                optimizer.zero_grad()
                with autocast():
                    upgrid = cnn(inputs)
                loss = calc_loss(upgrid.float(), targets, target_ims, metric, metrics)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if opt.rank==0:
                    print('Epoch {}/{} Train {}/{} dice {:.4f} reg {:.4f} mse3d {:.4f}  mse2d {:.4f}  MaxOut {:.2f}  ids {}'.format( \
                        epoch+1,end_epoch, batch_ind+1,steps_train, metric['dice'],metric['reg'],metric['mse3d'],metric['mse2d'], upgrid.max(), fileids))
                    # img_mse {:.4f}  metric['img_mse'], fileid:%s scheduler.get_lr()[0], optimizer.param_groups[0]['lr']

        if opt.scheduler_type == 'StepLR':
            scheduler.step()

        for key in metrics.keys():
            dist.all_reduce_multigpu([metrics[key]])

        # calculate and print all keys in metrics
        if opt.rank == 0:
            print_metrics(metrics, setup_params['ntrain'], 'Train',log)

        # record training loss and jaccard results
        mean_train_loss = (metrics['Loss']/setup_params['ntrain'])
        learning_results['train_loss'].append(mean_train_loss)

        mean_train_dice = (metrics['Dice']/setup_params['ntrain'])
        learning_results['train_dice'].append(mean_train_dice)
        mean_train_mse3d = (metrics['MSE3D']/setup_params['ntrain'])
        learning_results['train_mse3d'].append(mean_train_mse3d)

        # validation
        cnn.eval()
        metrics_val = defaultdict(float)

        with torch.set_grad_enabled(False):
            for batch_ind, (inputs, targets, target_ims, fileids) in enumerate(validation_generator):

                inputs = inputs.to(device)
                targets = targets.to(device)
                target_ims = target_ims.to(device)

                # forward
                optimizer.zero_grad()
                with autocast():
                    upgrid = cnn(inputs)
                val_loss = calc_loss(upgrid.float(), targets,target_ims, metric, metrics_val)

                if opt.rank==0:
                    print('Epoch {}/{} Val {}/{} dice {:.4f}  mse3d {:.4f}  mse2d {:.4f}  MaxOut {:.2f}  ids {}'.format( \
                        epoch+1,end_epoch, batch_ind+1,steps_val, metric['dice'],metric['reg'],metric['mse3d'],metric['mse2d'], upgrid.max(), fileids))
                # img_mse {:.4f}  metric['img_mse']

        for key in metrics_val.keys():
            dist.all_reduce_multigpu([metrics_val[key]])

        # calculate and print all keys in metrics_val
        if opt.rank==0:
            print_metrics(metrics_val, setup_params['nvalid'], 'Valid',log)

        # record validation loss and jaccard results
        mean_valid_loss = (metrics_val['Loss']/setup_params['nvalid'])
        learning_results['valid_loss'].append(mean_valid_loss)

        mean_valid_dice = (metrics_val['Dice']/setup_params['nvalid'])
        learning_results['valid_dice'].append(mean_valid_dice)
        mean_valid_mse3d = (metrics_val['MSE3D']/setup_params['nvalid'])
        learning_results['valid_mse3d'].append(mean_valid_mse3d)

        if not opt.scheduler_type == 'StepLR':
            scheduler.step(mean_valid_loss)

        # sanity check: record maximal value and sum of last validation sample
        max_last = upgrid.max()
        sum_last = upgrid.sum()/params_valid['batch_size']
        learning_results['max_valid'].append(max_last)
        learning_results['sum_valid'].append(sum_last)

        # visualize in tensorboard
        if opt.rank==0:
            for key in metrics.keys():
                logger.scalars_summary('All/{}'.format(key) ,{'train':metrics[key]/setup_params['ntrain'], 'val':metrics_val[key]/setup_params['nvalid']}, epoch+1)
            logger.scalar_summary('Other/MaxOut', max_last, epoch+1)
            logger.scalar_summary('Other/MaxOutSum', sum_last, epoch+1)
            logger.scalar_summary('Other/Learning Rate', optimizer.param_groups[0]['lr'], epoch+1)


        # save checkpoint
        if opt.rank==0:
            # save latest checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cnn.state_dict(),
                'optimizer': optimizer.state_dict()}, os.path.join(path_save,'checkpoint_latest'))
            # save checkpoint per saveEpoch
            if epoch%opt.saveEpoch == opt.saveEpoch-1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cnn.state_dict(),
                    'optimizer': optimizer.state_dict()}, os.path.join(path_save,'checkpoint_'+str(epoch+1)))

        # save checkpoint for best val loss
        if mean_valid_loss < (valid_loss_prev - 1e-4):
            if opt.rank==0:
                # print an update and save the model weights
                print_log('Val Loss Improved from %.4f to %.4f, Saving Model Weights...'% (valid_loss_prev, mean_valid_loss), log)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cnn.state_dict(),
                    'optimizer': optimizer.state_dict()}, os.path.join(path_save,'checkpoint_best_loss'))
            # change minimal loss and init stagnation indicator
            valid_loss_prev = mean_valid_loss
            not_improve = 0
        else:
            # update stagnation indicator
            not_improve += 1
            if opt.rank==0:
                print_log('Val loss not improve by %d epochs, best val loss: %.6f'% (not_improve,valid_loss_prev), log)


        epoch_time_elapsed = time.time() - epoch_start_time
        if opt.rank==0:
            print_log('Max test last: %.4f, Sum test last: %.4f' %(max_last, sum_last), log)
            print_log('{}, Epoch complete in {}\n'.format(time.ctime(time.time()),print_time(epoch_time_elapsed)), log)

            # save all records for latter visualization
            path_learning_results = os.path.join(path_save, 'learning_results.pickle')
            with open(path_learning_results, 'wb') as handle:
                pickle.dump(learning_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if no improvement for more than 7 epochs break training
        if not_improve >= 15 or optimizer.param_groups[0]['lr']<1e-7:
            break

    # measure time that took the model to train
    train_time_elapsed = time.time() - train_start_time
    if opt.rank==0:
        print_log('Training complete in {}'.format(print_time(train_time_elapsed)), log)
        print_log('Best Validation Loss: {:6f}'.format(valid_loss_prev), log)
        learning_results['last_epoch_time'], learning_results['training_time'] = epoch_time_elapsed, train_time_elapsed

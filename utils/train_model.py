# official modules
import os
import time
from time import localtime, strftime
from math import ceil
import pickle5 as pickle
import numpy as np
from collections import defaultdict
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
# self-defined modules
from utils.loss import calculate_loss
from utils.helper import print_log, print_metrics, print_time, print_metric_format


def train_model(model,optimizer,scheduler,device,training_generator,validation_generator,log,logger,opt):

    learning_results = defaultdict(list)
    max_epoch = opt.max_epoch

    steps_train = ceil(opt.ntrain / opt.batch_size / opt.world_size)
    steps_val = ceil(opt.nval / opt.batch_size / opt.world_size)
    params_val = {'batch_size': opt.batch_size, 'shuffle': False}

    # loss function
    loss_type = ['loss']
    extra_weight = []
    if opt.extra_loss: # None or string
        extra_loss = opt.extra_loss.split('_')
        extra_weight = [float(n) for n in opt.weight.split('_')]
        loss_type = loss_type + extra_loss
        if not len(extra_loss) == len(extra_weight):
            raise Exception(f'Input {len(extra_loss)} weight with {len(extra_weight)} extra loss')
    print_log(f'[INFO==>] Loss types: {loss_type}',log)
    calc_loss = calculate_loss(opt,loss_type,extra_weight)

    scaler = GradScaler()

    if opt.resume:
        checkpoint = opt.checkpoint_path
        start_epoch = torch.load(checkpoint)['epoch']
        end_epoch = max_epoch

        # load all recorded metrics in checkpoint
        tmp_path = os.path.dirname(checkpoint)
        with open(os.path.join(tmp_path, 'learning_results.pickle'), 'rb') as handle:
            learning_results = pickle.load(handle)

        if opt.rank==0:
            print_log('[INFO==>] Resume from epoch {}/{}, load loss...\n'.format(start_epoch,len(learning_results['train_loss'])),log)
        for key in learning_results.keys():
            try:
                learning_results[key] = learning_results[key][:start_epoch]
            except:
                pass

        # visualize results of checkpoint in tensorboard
        """ Haven't modify yet """
        if opt.rank==0:
            for ii in range(len(learning_results['train_loss'])):
                logger.scalars_summary('All/Loss',{'train':learning_results['train_loss'][ii], 'val':learning_results['val_loss'][ii]}, ii+1)
                try:
                    logger.scalars_summary('All/MSE3D',{'train':learning_results['train_mse3d'][ii], 'val':learning_results['val_mse3d'][ii]}, ii+1)
                    logger.scalars_summary('All/MSE2D',{'train':learning_results['train_mse2d'][ii], 'val':learning_results['val_mse2d'][ii]}, ii+1)
                    logger.scalars_summary('All/Dice',{'train':learning_results['train_dice'][ii], 'val':learning_results['val_dice'][ii]}, ii+1)
                except: pass

                logger.scalar_summary('Other/MaxOut', learning_results['val_max'][ii], ii+1)
                logger.scalar_summary('Other/MaxOutSum', learning_results['val_sum'][ii], ii+1)


        # initialize validation set best loss and jaccard
        best_val_loss = np.min(learning_results['val_loss'])

    else:
        # start from scratch
        start_epoch, end_epoch = 0, max_epoch
        learning_results = {'val_max': [], 'val_sum': [], 'steps_per_epoch': steps_train}
        for loss in loss_type:
            learning_results['train_'+loss] = []
            learning_results['val_'+loss] = []
        best_val_loss = float('Inf')

    # starting time of training
    train_start_time = time.time()
    not_improve = 0

    print_log(f'[INFO==>] Start training from {start_epoch} to {end_epoch} rank {opt.rank}\n',log)
    for epoch in np.arange(start_epoch, end_epoch):
        # starting time of current epoch
        epoch_start_time = time.time()

        if opt.rank == 0:
            print_log(f'Epoch {epoch+1}/{end_epoch} | {strftime("%Y-%m-%d %H:%M:%S", localtime())} | lr {optimizer.param_groups[0]["lr"]}', log, arrow=True)

        # training phase
        model.train()
        metric, metrics = defaultdict(float), defaultdict(float)
        with torch.set_grad_enabled(True):
            for batch_ind, (inputs, targets, target_ims, fileids) in enumerate(training_generator):

                inputs = inputs.to(device)
                targets = targets.to(device)
                target_ims = target_ims.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                loss = calc_loss(upgrid=outputs.float(), gt_upgrid=targets, gt_im=target_ims, metric=metric, metrics=metrics)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if opt.rank==0:
                    print(f'Epoch {epoch}/{end_epoch-1} Train {batch_ind}/{steps_train-1} MaxOut {outputs.max():.2f} {print_metric_format(metric)}')


        # if opt.scheduler_type == 'StepLR':
            # scheduler.step()

        if opt.gpu_number:
            for key in metrics.keys():
                dist.all_reduce_multigpu([metrics[key]])

        # calculate and print all keys in metrics
        if opt.rank == 0:
            print_metrics(metrics, opt.ntrain, 'Train', log)

        # record training loss
        for key in loss_type:
            mean_train_tmp = (metrics[key]/opt.ntrain).cpu().numpy()
            learning_results['train_'+key].append(mean_train_tmp)




        """ validation """
        model.eval()
        metrics_val = defaultdict(float)

        with torch.set_grad_enabled(False):
            for batch_ind, (inputs, targets, target_ims, fileids) in enumerate(validation_generator):

                inputs = inputs.to(device)
                targets = targets.to(device)
                target_ims = target_ims.to(device)

                # forward
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                val_loss = calc_loss(upgrid=outputs.float(), gt_upgrid=targets, gt_im=target_ims, metric=metric, metrics=metrics_val)

                if opt.rank==0:
                    print(f'Epoch {epoch+1}/{end_epoch} Val {batch_ind}/{steps_val-1} MaxOut {outputs.max():.2f} {print_metric_format(metric)}')

        if opt.gpu_number:
            for key in metrics_val.keys():
                dist.all_reduce_multigpu([metrics_val[key]])

        # calculate and print all keys in metrics_val
        if opt.rank == 0:
            print_metrics(metrics_val, opt.nval, 'Valid',log)

        # record validation loss
        for key in loss_type:
            mean_train_tmp = (metrics_val[key]/opt.ntrain).cpu().numpy()
            learning_results['val_'+key].append(mean_train_tmp)

        # if not opt.scheduler_type == 'StepLR':
        mean_val_loss = (metrics_val['loss']/opt.nval).cpu().numpy()
        scheduler.step(mean_val_loss)

        # sanity check: record maximal value and sum of last validation sample
        max_last = outputs.max().cpu().numpy()
        sum_last = (outputs.sum()/params_val['batch_size']).cpu().numpy()
        learning_results['val_max'].append(max_last)
        learning_results['val_sum'].append(sum_last)



        # visualize in tensorboard
        if opt.rank==0:
            for key in metrics.keys():
                logger.scalars_summary('LOSS/{}'.format(key) ,{'train':metrics[key]/opt.ntrain, 'val':metrics_val[key]/opt.nval}, epoch)
            logger.scalar_summary('Other/MaxOut', max_last/opt.scaling_factor, epoch)
            logger.scalar_summary('Other/MaxOutSum', sum_last, epoch)
            logger.scalar_summary('Other/Learning Rate', optimizer.param_groups[0]['lr'], epoch)


        # save checkpoint
        if opt.rank==0:
            # save latest checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, os.path.join(opt.save_path,'ckpt_latest'))
            # save checkpoint per save_epoch
            if epoch%opt.save_epoch == opt.save_epoch-1:
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, os.path.join(opt.save_path,'ckpt_'+str(epoch+1)))

        # save checkpoint for best val loss
        if mean_val_loss < (best_val_loss - 1e-4):
            if opt.rank==0:
                # print an update and save the model weights
                print_log('Val loss improved from %.4f to %.4f, saving best model...'% (best_val_loss, mean_val_loss), log, arrow=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, os.path.join(opt.save_path,'ckpt_best_loss'))
            # change minimal loss and init stagnation indicator
            best_val_loss = mean_val_loss
            not_improve = 0
        else:
            # update stagnation indicator
            not_improve += 1
            if opt.rank==0:
                print_log('Val loss not improve by %d epochs, best val loss: %.4f'% (not_improve,best_val_loss), log, arrow=True)

        epoch_time_elapsed = time.time() - epoch_start_time
        if opt.rank==0:
            print_log('Max test last: %.2f, Sum test last: %.2f' %(max_last, sum_last), log, arrow=True)
            print_log('{}, Epoch complete in {}\n'.format(time.ctime(time.time()),print_time(epoch_time_elapsed)), log, arrow=True)

            # save all records for latter visualization
            with open(os.path.join(opt.save_path, 'learning_results.pickle'), 'wb') as handle:
                pickle.dump(learning_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if no improvement for more than 15 epochs, break training
        if not_improve >= 30 or optimizer.param_groups[0]['lr']<1e-7:
            break

    # measure time that took the model to train
    train_time_elapsed = time.time() - train_start_time
    if opt.rank==0:
        print_log('[INFO==>] Training complete in {}'.format(print_time(train_time_elapsed)), log)
        print_log('[INFO==>] Best Validation Loss: {:6f}'.format(best_val_loss), log)

        learning_results['last_epoch_time'], learning_results['training_time'] = epoch_time_elapsed, train_time_elapsed
        with open(os.path.join(opt.save_path, 'learning_results.pickle'), 'wb') as handle:
            pickle.dump(learning_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

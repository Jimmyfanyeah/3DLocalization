import argparse
from cmath import inf
import datetime
import os
import shutil
import socket
import sys
from pathlib import Path
import numpy
import time
import torch

import decode.evaluation
import decode.neuralfitter
import decode.neuralfitter.coord_transform
import decode.neuralfitter.utils
import decode.simulation
import decode.utils
# from decode.neuralfitter.train.random_simulation import setup_random_simulation
from decode.neuralfitter.utils import log_train_val_progress
from decode.utils.checkpoint import CheckPoint


def parse_args():
    parser = argparse.ArgumentParser(description='Training Args')

    parser.add_argument('-i', '--device', default=None, 
                        help='Specify the device string (cpu, cuda, cuda:0) and overwrite param.',
                        type=str)

    parser.add_argument('-p', '--param_file', default=None,
                        help='Specify your parameter file (.yml or .json).', type=str)

    parser.add_argument('-w', '--num_worker_override',default=None,
                        help='Override the number of workers for the dataloaders.',
                        type=int)

    parser.add_argument('-n', '--no_log', default=False, action='store_true',
                        help='Set no log if you do not want to log the current run.')

    parser.add_argument('-c', '--log_comment', default=None,
                        help='Add a log_comment to the run.')

    parser.add_argument('-d', '--data_path_override', default=None,
                        help='Specify your path to data', type=str)

    parser.add_argument('-is', '--img_size_override', default=None,
                        help='Override img size', type=int)

    args = parser.parse_args()
    return args


def live_engine_setup(args):
    """
    Sets up the engine to train DECODE. Includes sample simulation and the actual training.

    Args:
        param_file: parameter file path
        device_overwrite: overwrite cuda index specified by param file
        debug: activate debug mode (i.e. less samples) for fast testing
        num_worker_override: overwrite number of workers for dataloader
        no_log: disable logging
        log_folder: folder for logging (where tensorboard puts its stuff)
        log_comment: comment to the experiment
    """

    """Load Parameters and back them up to the network output directory"""
    param_file = Path(args.param_file)
    param = decode.utils.param_io.ParamHandling().load_params(param_file)

    # add meta information - Meta=namespace(version='0.10.0'),
    param.Meta.version = decode.utils.bookkeeping.decode_state()

    """Experiment ID"""
    if param.InOut.checkpoint_init is None:
        experiment_id = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") + '_' + socket.gethostname()
        from_ckpt = False
        if args.log_comment:
            experiment_id = experiment_id + '_' + args.log_comment
    else:
        from_ckpt = True
        experiment_id = Path(param.InOut.checkpoint_init).parent.name

    """Set up unique folder for experiment"""
    if not from_ckpt:
        experiment_path = Path(param.InOut.experiment_out) / Path(experiment_id)
    else:
        experiment_path = Path(param.InOut.checkpoint_init).parent

    if not experiment_path.parent.exists():
        experiment_path.parent.mkdir()

    if not from_ckpt:
        experiment_path.mkdir(exist_ok=False)

    model_out = experiment_path / Path('model.pt')
    ckpt_path = experiment_path / Path('ckpt.pt')

    # Modify parameters
    if args.num_worker_override is not None:
        param.Hardware.num_worker_train = args.num_worker_override

    """Hardware / Server stuff."""
    if args.device is not None:
        device = args.device
        # param.Hardware.device_simulation = device_overwrite  # lazy assumption
    else:
        device = param.Hardware.device

    if args.data_path_override is not None:
        param.InOut.data_path = args.data_path_override

    if args.img_size_override is not None:
        param.Simulation.img_size = [args.img_size_override,args.img_size_override]
        param.Simulation.psf_extent = [[-0.5, args.img_size_override-0.5],
                                       [-0.5, args.img_size_override-0.5], None]

        param.TestSet.frame_extent = param.Simulation.psf_extent
        param.TestSet.img_size = param.Simulation.img_size

    # Backup the parameter file under the network output path with the experiments ID
    param_backup_in = experiment_path / Path('param_run_in').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_backup_in)

    param_backup = experiment_path / Path('param_run').with_suffix(param_file.suffix)
    decode.utils.param_io.ParamHandling().write_params(param_backup, param)

    if sys.platform in ('linux', 'darwin'):
        os.nice(param.Hardware.unix_niceness)
    elif param.Hardware.unix_niceness is not None:
        print(f"Cannot set niceness on platform {sys.platform}. You probably do not need to worry.")

    torch.set_num_threads(param.Hardware.torch_threads)

    """Setup Log System"""
    if args.no_log:
        logger = decode.neuralfitter.utils.logger.NoLog()

    else:
        log_folder = experiment_path

        logger = decode.neuralfitter.utils.logger.MultiLogger(
            [decode.neuralfitter.utils.logger.SummaryWriter(log_dir=log_folder,
                                                            filter_keys=["dx_red_mu", "dx_red_sig",
                                                                         "dy_red_mu",
                                                                         "dy_red_sig", "dz_red_mu",
                                                                         "dz_red_sig",
                                                                         "dphot_red_mu",
                                                                         "dphot_red_sig"]),
             decode.neuralfitter.utils.logger.DictLogger()])

    # sim_train, sim_test = setup_random_simulation(param)
    ds_train, ds_test, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, ckpt = setup_trainer(logger, model_out, ckpt_path, device, param)
    dl_train, dl_test = setup_dataloader(param, ds_train, ds_test)

    if from_ckpt:
        ckpt = decode.utils.checkpoint.CheckPoint.load(param.InOut.checkpoint_init)
        model.load_state_dict(ckpt.model_state)
        optimizer.load_state_dict(ckpt.optimizer_state)
        lr_scheduler.load_state_dict(ckpt.lr_sched_state)
        first_epoch = ckpt.step + 1
        model = model.train()
        print(f'Resuming training from checkpoint ' + experiment_id)
    else:
        first_epoch = 0

    best_val_loss = inf
    for i in range(first_epoch, param.HyperParameter.epochs):
        logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)

        print(f'Epoch{i}')

        if i >= 1:
            _ = decode.neuralfitter.train_val_impl.train(
                model=model,
                optimizer=optimizer,
                loss=criterion,
                dataloader=dl_train,
                grad_rescale=param.HyperParameter.moeller_gradient_rescale,
                grad_mod=grad_mod,
                epoch=i,
                device=torch.device(device),
                logger=logger
            )

        # val_loss=avg of loss for all batches
        # test_out=list of network_output: ["loss", "x", "y_out", "y_tar", "weight", "em_tar"]
        val_loss, test_out = decode.neuralfitter.train_val_impl.test(
            model=model,
            loss=criterion,
            dataloader=dl_test,
            epoch=i,
            device=torch.device(device))
        # print(val_loss)

        if best_val_loss - val_loss >1e-4:
            best_val_loss = val_loss
            model_ls.save(model, None, epoch_idx='best')

        t0 = time.time()
        if i%3 == 0:
            """Post-Process and Evaluate"""
            log_train_val_progress.post_process_log_test(loss_cmp=test_out.loss,
                                                        loss_scalar=val_loss,
                                                        x=test_out.x, y_out=test_out.y_out,
                                                        y_tar=test_out.y_tar,
                                                        weight=test_out.weight,
                                                        em_tar=ds_test.emitter(),
                                                        px_border=-0.5, px_size=1.,
                                                        post_processor=post_processor,
                                                        matcher=matcher, logger=logger,
                                                        step=i)
        else:
            log_train_val_progress.log_kpi_simplified(loss_scalar=val_loss,
                                                    loss_cmp=test_out.loss,
                                                    logger=logger,
                                                    step=i)

        t_log = time.time() - t0
        print(f'log time:{t_log}')

        if i >= 1:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

        if i%10 == 0:
            model_ls.save(model, None, epoch_idx=str(i))

        if args.no_log:
            ckpt.dump(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict(),
                        step=i)
        else:
            ckpt.dump(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict(),
                        log=logger.logger[1].log_dict, step=i)

    print("Training finished after reaching maximum number of epochs.")



def setup_trainer(logger, model_out, ckpt_path, device, param):
    """Set model, optimiser, loss and schedulers"""
    models_available = {
        'SigmaMUNet': decode.neuralfitter.models.SigmaMUNet,
        'DoubleMUnet': decode.neuralfitter.models.model_param.DoubleMUnet,
        'SimpleSMLMNet': decode.neuralfitter.models.model_param.SimpleSMLMNet,
    }

    model = models_available[param.HyperParameter.architecture]
    model = model.parse(param)

    model_ls = decode.utils.model_io.LoadSaveModel(model, output_file=model_out)

    model = model_ls.load_init()
    model = model.to(torch.device(device))

    # Small collection of optimisers
    optimizer_available = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    }

    optimizer = optimizer_available[param.HyperParameter.optimizer]
    optimizer = optimizer(model.parameters(), **param.HyperParameter.opt_param)

    """Loss function."""
    criterion = decode.neuralfitter.loss.GaussianMMLoss(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size,
        device=device,
        chweight_stat=param.HyperParameter.chweight_stat)

    """Learning Rate and Simulation Scheduling"""
    lr_scheduler_available = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'StepLR': torch.optim.lr_scheduler.StepLR
    }
    lr_scheduler = lr_scheduler_available[param.HyperParameter.learning_rate_scheduler]
    lr_scheduler = lr_scheduler(optimizer, **param.HyperParameter.learning_rate_scheduler_param)

    """Checkpointing"""
    checkpoint = CheckPoint(path=ckpt_path)

    """Setup gradient modification"""
    grad_mod = param.HyperParameter.grad_mod

    """Log the model (Graph) """
    try:
        dummy = torch.rand((2, param.HyperParameter.channels_in,
                            *param.Simulation.img_size), requires_grad=False).to(torch.device(device))
        logger.add_graph(model, dummy)

    except:
        print("Did not log graph.")
        # raise RuntimeError("Your dummy input is wrong. Please update it.")

    """Setup Target generator consisting possibly multiple steps in a transformation sequence."""
    tar_proc = decode.neuralfitter.utils.processing.TransformSequence(
        [
            # param_tar --> phot/max, z/z_max, bg/bg_max
            decode.neuralfitter.scale_transform.ParameterListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max)
        ])

    train_IDs = numpy.arange(1,18001,1).tolist()
    val_IDs = numpy.arange(18001,20001,1).tolist()
    # train_IDs = numpy.arange(1,27001,1).tolist()
    # val_IDs = numpy.arange(27001,30001,1).tolist()
    # train_IDs = numpy.arange(0,9000,1).tolist()
    # val_IDs = numpy.arange(9000,10000,1).tolist()
    # train_IDs = numpy.arange(1,9001,1).tolist()
    # val_IDs = numpy.arange(9001,10001,1).tolist()

    train_ds = decode.neuralfitter.dataset.rPSFDataset(root_dir=param.InOut.data_path,
                                                       list_IDs=train_IDs, label_path=None, 
                                                       n_max=param.HyperParameter.max_number_targets,
                                                       tar_proc=tar_proc,
                                                       img_shape=param.Simulation.img_size)

    test_ds = decode.neuralfitter.dataset.rPSFDataset(root_dir=param.InOut.data_path,
                                                       list_IDs=val_IDs, label_path=None, 
                                                       n_max=param.HyperParameter.max_number_targets,
                                                       tar_proc=tar_proc,
                                                       img_shape=param.Simulation.img_size)

    # print(test_ds.label_gen())

    """Set up post processor"""
    if param.PostProcessing is None:
        post_processor = decode.neuralfitter.post_processing.NoPostProcessing(xy_unit='px',
                                                                              px_size=param.Camera.px_size)

    elif param.PostProcessing == 'LookUp':
        post_processor = decode.neuralfitter.utils.processing.TransformSequence([

            decode.neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max),

            decode.neuralfitter.coord_transform.Offset2Coordinate.parse(param),

            decode.neuralfitter.post_processing.LookUpPostProcessing(
                raw_th=param.PostProcessingParam.raw_th,
                pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],
                xy_unit='px',
                px_size=param.Camera.px_size)
        ])

    elif param.PostProcessing in ('SpatialIntegration', 'NMS'):  # NMS as legacy support
        post_processor = decode.neuralfitter.utils.processing.TransformSequence([
            # out_tar --> out_tar: photo*photo_max, z*z_max, bg*bg_max
            decode.neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max),
            # offset --> coordinate e.g., 0.2 --> 10.2 
            decode.neuralfitter.coord_transform.Offset2Coordinate.parse(param),

            decode.neuralfitter.post_processing.SpatialIntegration(
                raw_th=param.PostProcessingParam.raw_th, # 0.5
                xy_unit='px')
        ])

    else:
        raise NotImplementedError

    """Evaluation Specification"""
    matcher = decode.evaluation.match_emittersets.GreedyHungarianMatching.parse(param)
    # matcher = None

    return train_ds, test_ds, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, checkpoint


def setup_dataloader(param, train_ds, test_ds=None):
    """Set up dataloader"""

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameter.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=param.Hardware.num_worker_train,
        pin_memory=True,
        collate_fn=decode.neuralfitter.utils.dataloader_customs.smlm_collate)

    if test_ds is not None:

        test_dl = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=param.HyperParameter.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=param.Hardware.num_worker_train,
            pin_memory=False,
            collate_fn=decode.neuralfitter.utils.dataloader_customs.smlm_collate)
    else:

        test_dl = None

    return train_dl, test_dl


def main():
    args = parse_args()
    live_engine_setup(args)


if __name__ == '__main__':
    main()

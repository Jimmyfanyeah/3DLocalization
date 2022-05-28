import argparse
import yaml
import shutil
import os
import csv

import decode.utils
import decode.neuralfitter.utils

if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description="Inference. This uses the default, suggested implementation. "
                    "For anything else, consult the fitting notebook and make your changes there.")
    parse.add_argument('--fit_meta_path', '-p', 
                          help='Path to the fit meta file that specifies all following options in a yaml file',
                          default='/home/lingjia/Documents/rPSF/NN/infer_param.yaml')
    args = parse.parse_args()

    """Meta file"""
    with open(args.fit_meta_path) as f:
        meta = yaml.safe_load(f)

    device = meta['Hardware']['device']
    worker = meta['Hardware']['worker'] if meta['Hardware']['worker'] is not None else 4

    model_path = meta['Model']['path']
    model_param_path = meta['Model']['param_path']

    online = False

    # ToDo: This is a massive code duplication of the Fitting Notebook. PLEASE CLEAN UP!

    """Load the model"""
    param = decode.utils.param_io.load_params(model_param_path)

    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = decode.utils.model_io.LoadSaveModel(
        model, input_file=model_path, output_file=None).load_init(device)

    """Load the frame"""
    tar_proc = decode.neuralfitter.utils.processing.TransformSequence(
        [
            # param_tar --> phot/max, z/z_max, bg/bg_max
            decode.neuralfitter.scale_transform.ParameterListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max)
        ])

    # if list of folders
    frame_path_base = meta['Frames']['path']
    frame_path_list = os.listdir(frame_path_base)

    output_base = meta['Output']['path']
    os.makedirs(output_base,exist_ok=True)

    for frame_path in frame_path_list:
        output = os.path.join(output_base, frame_path)
        frame_path = os.path.join(frame_path_base, frame_path)
        os.makedirs(output,exist_ok=True)
        print(f'data:{frame_path}\nsave:{output}')

        # infer_ds = decode.neuralfitter.dataset.rPSF_InferenceDataset(root_dir=frame_path,
        #                                                 img_shape=param.Simulation.img_size)

        shutil.copy2(os.path.join(frame_path,'label.txt'),os.path.join(output,'label.txt'))

        # determine extent of frame and its dimension after frame_processing
        # size_procced = decode.neuralfitter.frame_processing.get_frame_extent(frames.unsqueeze(1).size(),
        #                                                                      frame_proc.forward)  # frame size after processing
        # frame_extent = ((-0.5, size_procced[-2] - 0.5), (-0.5, size_procced[-1] - 0.5))
        frame_extent = param.TestSet.frame_extent
        img_shape = param.TestSet.img_size

        # Setup post-processing
        # It's a sequence of backscaling, relative to abs. coord conversion and frame2emitter conversion
        post_proc = decode.neuralfitter.utils.processing.TransformSequence([

            decode.neuralfitter.scale_transform.InverseParamListRescale.parse(param),

            decode.neuralfitter.coord_transform.Offset2Coordinate(xextent=frame_extent[0],
                                                                yextent=frame_extent[1],
                                                                img_shape=img_shape),

            decode.neuralfitter.post_processing.SpatialIntegration(raw_th=0.1,
                                                                xy_unit='px',
                                                                px_size=None)
        ])

        """Fit"""
        # infer = decode.neuralfitter.Infer(model=model, ch_in=param.HyperParameter.channels_in,
        #                                 frame_proc=None, post_proc=post_proc,
        #                                 device=device, num_workers=worker, batch_size=1)
        infer = decode.neuralfitter.Infer(model=model, ch_in=param.HyperParameter.channels_in,
                                        frame_proc=None, post_proc=post_proc,
                                        device=device, num_workers=worker, batch_size=1,param=param)

        infer_ds = decode.neuralfitter.dataset.rPSFDataset(root_dir=frame_path,
                                                        img_shape=param.Simulation.img_size)

        emitter, loss_info = infer.forward(infer_ds)
        emitter.save(os.path.join(output,'infer.csv'))
        print(f"Fit done and emitters saved to {output}")

        row_list = loss_info.tolist()
        with open(os.path.join(output,'loss.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        # param_backup = os.path.join(output, 'param_run_model.yaml')
        # decode.utils.param_io.ParamHandling().write_params(param_backup, param)
        shutil.copy2(args.fit_meta_path,os.path.join(output,'infer_param.yaml'))

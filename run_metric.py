import argparse, logging
import os
import time
import sys

import pandas as pd

from data import *
import torch
from tqdm import trange
import pycvvdp
import pycvvdp.utils
from pycvvdp.vq_metric import vq_metric
from pycvvdp.run_cvvdp import np2vid, np2img
from pycvvdp.dm_preview import dm_preview_metric


# Create a directory if it does not exist
def mkdir_safe(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except:
            pass # Do not fail - other process could have created that dir


def main():
    all_datasets = ['AR-DAVID'] ## Add your available datasets here
    all_metrics = ['cvvdp','dm-preview'] ## Add your available metrics here / dm-preview for debugging purpose
    all_fusion_methods = ['none','mean','pinhole','pinhole-stereo','blur','blur-stereo'] ## Implemented in optics_model.py

    parser = argparse.ArgumentParser('Run a metric on the selected datasets')
    parser.add_argument('-d', '--datasets', default=['AR-DAVID'], choices=all_datasets, nargs='+',
                        help='dataset must exist as a symlink in folder "datasets/"')
    parser.add_argument('-m', '--metric', default='cvvdp', choices=all_metrics, help='Which metric to run.')
    parser.add_argument('-f','--fusion-method', default='pinhole', choices=all_fusion_methods, help='Which fusion method to run.')
    parser.add_argument('--discount_factor', type=float, default=1, help='Discount factor of the background (e.g. 0% --> 1, 100% --> 0)')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume running the metric (skip the conditions that have been already processed).')
    parser.add_argument("--gpu", type=int,  default=0, help="select which GPU to use (e.g. 0), default is GPU 0. Pass -1 to run on the CPU.")
    args = parser.parse_args()
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    logging.info( f"Running on device: {device}"  )

    ## Add your vq_metric here
    if args.metric == 'cvvdp':
        metric = pycvvdp.cvvdp(device=device, temp_padding='replicate')
    elif args.metric == 'dm-preview':
        metric = dm_preview_metric(output_exr =False, device =device)
    else:
        raise RuntimeError(f"Metric {args.metric} not supported.")


    res_path = 'metric_results' # Path to save the results
    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    ## Add your dataset here
    for i, d in enumerate(args.datasets):
        location = f'datasets/{d}'
        if d =='AR-DAVID':
            dataset = ARDAVID(location, 'all', cache_yuv = False, mf_method = args.fusion_method, discount_factor = args.discount_factor)
            dataset.n = -1
        else:
            raise RuntimeError(f"Dataset {d} not supported.")

        logging.info( f"Processing {dataset.get_ds_name()} - {len(dataset)} conditions")

        metric_name = metric.short_name()
        if d == 'AR-DAVID':
            res_file = os.path.join( res_path, f"{dataset.get_ds_name()}_{metric_name}.csv")
        else:
            res_file = os.path.join( res_path, f"{d}_{metric_name}.csv")


        if isinstance( metric, dm_preview_metric ):
            if d == 'AR-DAVID':
                preview_location =f'datasets/{dataset.get_ds_name()}'
            else:
                preview_location =f'datasets/{d}'
            preview_path = os.path.join( preview_location, "preview" )
            mkdir_safe(preview_path)
        else:
            preview_path = None

        ## Resume if the result file exists
        if args.resume and os.path.isfile(res_file):
            logging.info( f'Resuming using file "{res_file}"')
            df = pd.read_csv(res_file)
        else:
            df = pd.DataFrame( columns=['condition_id', 'test_file', 'reference_file', 'heatmap_file', 'Q', 'proc_time', 'Q_subj', 'part_id_scene'])
        
        try:
            last_save = time.time()
            step_range =1
            for kk in trange(0, len(dataset), step_range):
                condition_id = dataset.get_condition_id(kk)
                print( condition_id )

                if args.resume and (df["condition_id"] == condition_id).any():
                    logging.info( f"Skipping condition {condition_id}" )
                    # Skip processed conditions
                    continue

                vs, disp_photo, disp_geom, foveated, q_subj = dataset[kk]

                try:
                    metric.set_display_model(display_photometry=disp_photo, display_geometry=disp_geom)
                    metric.foveated = foveated

                    if preview_path:
                        base_fname = os.path.join(preview_path, condition_id)
                        metric.set_base_fname(base_fname)

                    with torch.no_grad():   # Do not accumulate gradients
                        s_time = time.time()
                        q_jod, stats = metric.predict_video_source(vs)
                        proc_time = time.time()-s_time
                except:
                    logging.error( f'Failed on condition {condition_id}' )
                    raise

                ht_file = ""

                test_fn, ref_fn = dataset.get_file_names(kk)
                part_id_scene = dataset.get_part(kk)
                
                new_row = pd.Series({ "condition_id": condition_id, "Q": q_jod.item(), "proc_time": proc_time, "Q_subj": q_subj, "test_file": test_fn, "reference_file": ref_fn, "heatmap_file": ht_file, "part_id_scene": part_id_scene })
                
                df = pd.concat( [df, new_row.to_frame().T], ignore_index=True )
                if (time.time()-last_save)>60: # Save every 60 seconds (important if the dataset has a lot of small images)
                    df.to_csv( res_file, index=False ) 
                    last_save = time.time()
        finally:
            # Save results even if the script fails in the middle
            df.to_csv( res_file, index=False )
            logging.info( f"Results saved to {res_file}" )

if __name__ == '__main__':
    main()
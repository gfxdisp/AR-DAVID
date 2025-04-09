import abc
import os
import glob
import os.path as osp
import numpy as np
import pandas as pd
import logging
import json
from tqdm import tqdm, trange
import ffmpeg
import torch
import torch.utils.data as D
import pytorch_lightning as pl
import pytorch_lightning.utilities.combined_loader as cl
from torch.functional import Tensor

import pycvvdp
import pycvvdp.utils
import pycvvdp.video_source_yuv as yuv
from pycvvdp.video_source_file import video_reader_yuv_pytorch, load_image_as_array
from pycvvdp.video_source import video_source_array, reshuffle_dims
import pycvvdp.display_model as display_model

import optics_model
import math

class VideoDataset(D.Dataset, metaclass=abc.ABCMeta):
    """
    Abstract class for all video datasets.
    """
    def __init__(self, path, mode, cache_loc=None, cache_yuv=True, random_parts=False):
        super().__init__()
        # self.n = 2    # Debugging
        logging.info(f'Loading dataset "{self.__class__.__name__}"')
        assert all(hasattr(self, attr) for attr in ('h', 'w', 'n')), 'Please set h, w, n'
        self.path = path
        do_split = self.read_quality_csv(mode)

        if do_split:
            parts = self.quality_table.part_id_scene.unique()
            if random_parts:
                with pl.utilities.seed.isolate_rng():
                    # Isolate RNG to ensure same permutation for train and val
                    parts = np.random.permutation(parts)

            # 80-20 train-val split
            if mode == 'train':
                parts = parts[:int(len(parts)*0.8)]
            elif mode == 'val':
                parts = parts[int(len(parts)*0.8):]
                print(f'Validation scene IDs for {self.__class__.__name__}: {parts}')
            else:
                assert mode == 'all', f'Invalid mode "{mode}", pick 1 of ["train", "val", "all"]'
            self.quality_table = self.quality_table[self.quality_table.part_id_scene.isin(parts)]

        self.cache_yuv=cache_yuv
        if cache_yuv:
            self.cache_loc = osp.join( path, "cache" ) if cache_loc is None else cache_loc
            logging.info(f'Cache location: {self.cache_loc}')
            if not osp.isdir(self.cache_loc) and not osp.islink(self.cache_loc):
                os.makedirs(self.cache_loc)

        self.cache_entries = [None] * self.__len__()

    def __getitem__(self, index):
        """
        Returns:
            vs:         video source object
            disp_photo: photometric display model
            disp_geom:  geometric display model
            foveated:   boolean foveated flag
            quality:    subjective quality (in JOD)
        """
        assert index in range(self.__len__()), f'{index} is out of range, len={self.__len__()}'

        row = self.quality_table.iloc[index]
        test_fname_base, ref_fname_base, quality = self.parse_row(row)

        test_fname = osp.join( self.path, f"{test_fname_base}.mp4" )
        ref_fname = osp.join( self.path, f"{ref_fname_base}.mp4" )

        if self.cache_yuv:
            test_fname_cache = osp.join( self.cache_loc, test_fname_base )
            ref_fname_cache = osp.join( self.cache_loc, ref_fname_base )

            if not self.cache_entries[index] is None:  # Item in our cache
                (test_fname_yuv, test_vprops, ref_fname_yuv, ref_vprops, vs) = self.cache_entries[index]
            else:
                # Check if the cache files exist (e.g. from the previous run)
                test_fname_yuv = glob.glob( f"{test_fname_cache}*.yuv" )
                ref_fname_yuv = glob.glob( f"{ref_fname_cache}*.yuv" )

                if len(test_fname_yuv)==0: # We need to create a test YUV file
                    test_fname_yuv, test_vprops = self._convert_to_yuv(test_fname, test_fname_cache)
                else:
                    assert( len(test_fname_yuv)==1 )
                    test_fname_yuv = test_fname_yuv[0]
                    # Decode video props from the file name
                    test_vprops = yuv.decode_video_props(test_fname_yuv)

                if len(ref_fname_yuv)==0: # No, we need to create a reference YUV file
                    ref_fname_yuv, ref_vprops = self._convert_to_yuv(ref_fname, ref_fname_cache)
                else: 
                    assert( len(ref_fname_yuv)==1 )
                    ref_fname_yuv = ref_fname_yuv[0]
                    # Decode video props from the file name
                    ref_vprops = yuv.decode_video_props(ref_fname_yuv)

                vs = self._get_video_source_yuv(test_fname_yuv, ref_fname_yuv, display_photometry=self.disp_photo, max_frames=self.n, row=row)
                h, w, n = vs.get_video_size()
                assert (self.h, self.w) == (h, w), f'Inconsistent cache for file "{test_fname_yuv}", please delete "{self.cache_loc}" or provide a different path to rebuild'                
                # and (self.n == n or self.n == -1)
                self.cache_entries[index] = (test_fname_yuv, test_vprops, ref_fname_yuv, ref_vprops, vs)            
        else:
            vs = self._get_video_source_mp4(test_fname, ref_fname, display_photometry=self.disp_photo, max_frames=self.n, row=row)

        return vs, self.disp_photo, self.disp_geom, False, np.float32(quality)

    @abc.abstractmethod
    def _get_video_source_yuv(self, test_fname_yuv, ref_fname_yuv, display_photometry, max_frames, row):
        """
        """
    @abc.abstractmethod
    def _get_video_source_mp4(self, test_fname_yuv, ref_fname_yuv, display_photometry, max_frames, row):
        """
        """

    def __len__(self):
        return self.quality_table.shape[0]

    @abc.abstractmethod
    def read_quality_csv(self, split):
        """
        Read and store the CSV file containing subjective quality scores.
        Store the data in a pandas dataframe called "self.quality_table".
        :split: can be "train" or "val" to determine which scenes to retain
        """

    @abc.abstractmethod
    def parse_row(self, row):
        """
        Parse one row from the quality scores table.
        Returns: test_file_name, reference_file_name, quality (in JOD)
        """

    def get_part( self, index ):
        return -1 # if part (for train/test splits) is not available

    # Get file names of the test and reference files
    def get_file_names( self, index ):
        row = self.quality_table.iloc[index]
        if len(self.parse_row(row))==4:
            test_fname_base, ref_fname_base, quality, resolution = self.parse_row(row)
        elif len(self.parse_row(row))==3:
            test_fname_base, ref_fname_base, quality = self.parse_row(row)

        test_fname = osp.join( self.path, f"{test_fname_base}.mp4" )
        ref_fname = osp.join( self.path, f"{ref_fname_base}.mp4" )
        return test_fname, ref_fname

    def _convert_to_yuv( self, in_mp4, out_basename ):
        with video_reader_yuv_pytorch( in_mp4 ) as vr:            
            color_space = "2020" if vr.color_space == "bt2020nc" else "709"
            vprops = { "width": vr.width, "height": vr.height, "bit_depth": vr.bit_depth, "fps": vr.avg_fps, "color_space": color_space, "chroma_ss": vr.chroma_ss }
            
            # Create a directory as needed
            dname = os.path.dirname(out_basename)
            if not os.path.isdir(dname):
                os.makedirs(dname)

            out_yuv = yuv.create_yuv_fname(out_basename, vprops)
            with open(out_yuv, "wb") as of:
                n_frames = vr.frames if self.n == -1 or (hasattr(self, 'store_full_yuv') and self.store_full_yuv) else self.n
                n_frames = min(n_frames, vr.frames)
                for _ in range(n_frames):
                    frame = vr.get_frame()
                    of.write( frame.tobytes() )
        return out_yuv, vprops

    @abc.abstractmethod
    def get_ds_name(self):
        """
        Unique name for every class
        """

    # A custom video source for AR-DAVID that adds background image
class video_source_ardavid(pycvvdp.video_source_video_file):

    image_cache = {}
    frg_size = (1080, 1920)

    def get_background_image( self, bkg_fname, device ):
        if bkg_fname in video_source_ardavid.image_cache:
            return video_source_ardavid.image_cache[bkg_fname]

        path = 'datasets/AR-DAVID/Background images'
        im = load_image_as_array(os.path.join(path, bkg_fname))
        assert im.dtype==np.uint8
        im_tensor = self.bkg_dp.source_2_target_colourspace( reshuffle_dims(torch.tensor(im, device=device)/255., in_dims="HWC", out_dims="BCFHW" ), target_colorspace="RGB2020" )

        ## For debugging, erase before run
        # tmp = torch.tensor(load_image_as_array(os.path.join(path, 'bright_leaves.png')))
        # tmp_tensor = self.bkg_dp.source_2_target_colourspace( reshuffle_dims(torch.tensor(tmp, device=device)/255., in_dims="HWC", out_dims="BCFHW" ), target_colorspace="RGB2020" )
        # im_tensor = im_tensor+tmp_tensor
        
        # Rescale the background tensor to the size of the foreground
        bkg_sz = im_tensor.shape[-2:]
        
        center = (int(bkg_sz[0]/2), int(bkg_sz[1]/2))
              
        
        wh2 = (int(round(video_source_ardavid.frg_size[0]/2*self.bkg_to_frg_scale)), int(round(video_source_ardavid.frg_size[1]/2*self.bkg_to_frg_scale)))
        assert wh2[0]<=center[0] and wh2[1]<=center[1]

        im_result_tensor = torch.zeros(size=(1,im_tensor.shape[1],1,video_source_ardavid.frg_size[0],video_source_ardavid.frg_size[1]),device=im_tensor.device)
        if self.stereo == True: ## Stereo mode assuming gaze at the center of foreground
            angle_stereo = math.atan(self.ipd*1e-3/2/self.test_dg.distance_m)
            dist_diff = abs(self.bkg_dg.distance_m - self.test_dg.distance_m)
            pix_stereo = math.floor(dist_diff*math.tan(angle_stereo) / self.bkg_dg.display_size_m[0] *self.bkg_dg.resolution[0])
            for i in [-1, 1]: ## LEFT / RIGHT in horizontal disparity
                im_cropped = im_tensor[:,:,:,(center[0]-wh2[0]):(center[0]+wh2[0]+1),(center[1]+i*pix_stereo-wh2[1]):(center[1]+i*pix_stereo+wh2[1]+1)]
                im_result_tensor = im_result_tensor + 1/2*torch.nn.functional.interpolate( im_cropped.view(1,1,im_cropped.shape[-2],im_cropped.shape[-1]), 
                        size=video_source_ardavid.frg_size, mode='bilinear', antialias=True).view(1,im_cropped.shape[1],1,video_source_ardavid.frg_size[0],video_source_ardavid.frg_size[1])

        else:
            im_cropped = im_tensor[:,:,:,(center[0]-wh2[0]):(center[0]+wh2[0]+1),(center[1]-wh2[1]):(center[1]+wh2[1]+1)]
            im_result_tensor = im_result_tensor+ torch.nn.functional.interpolate( im_cropped.view(1,1,im_cropped.shape[-2],im_cropped.shape[-1]), 
                        size=video_source_ardavid.frg_size, mode='bilinear', antialias=True).view(1,im_cropped.shape[1],1,video_source_ardavid.frg_size[0],video_source_ardavid.frg_size[1])

        video_source_ardavid.image_cache[bkg_fname] = im_result_tensor

        return im_result_tensor

                
    def __init__( self, test_fname, reference_fname, bkg_pattern, bkg_luminance, mf_method, discount_factor, display_photometry, config_paths=[], frames=-1, bkg_to_frg_scale=1 ):
        super().__init__( test_fname, reference_fname, display_photometry=display_photometry, config_paths=config_paths, frames=frames )

        assert bkg_luminance in [10, 100]
        assert bkg_pattern in ["flat", "leaves", "noise"]
        
        self.bkg_image = f"{'bright' if bkg_luminance==100 else 'dim'}_{bkg_pattern}.png"
        self.mf_method = mf_method

        self.test_fname = test_fname
        self.reference_fname = reference_fname
        
        # Display model used for the final color space conversion BT.2020 to target. Note that we use 500 nit peak: 300 nit for Eizo + up to 200 for the background
        self.comp_dp = pycvvdp.vvdp_display_photo_eotf(Y_peak=500, contrast = 1000000, source_colorspace='BT.2020-linear', EOTF="linear")

        self.test_dp = pycvvdp.vvdp_display_photometry.load( "eizo_CG3146-AR-DAVID", config_paths=config_paths )
        self.test_dg = pycvvdp.vvdp_display_geometry.load( "eizo_CG3146-AR-DAVID", config_paths=config_paths )

        self.bkg_dp = pycvvdp.vvdp_display_photometry.load( "Dynascan", config_paths=config_paths )
        self.bkg_dg = pycvvdp.vvdp_display_geometry.load( "Dynascan", config_paths=config_paths )

        self.bkg_to_frg_scale = bkg_to_frg_scale
        
        self.optics_model = None # 
        self.ref_fs_weight = None
        self.test_fs_weight = None
        self.UPDATE_PSF = False # updating psf (computation load x frames)
        
        ## TODO: implement stereo vision 
        self.ipd = 60 # Set interpupilarity distance in millemeter
        if "stereo" in self.mf_method:
            self.stereo = True
        else:
            self.stereo = False

        self.Num_layer = 2 # Number of layers (foreground and background)
        self.discount_factor = discount_factor # Discount factor for the background image (0-1)

    def add_background(self, frame, device):
        bkg_image = self.get_background_image(self.bkg_image, device)
        if self.optics_model is None :
            self.optics_model = optics_model.optics_model(Num_layer=self.Num_layer, 
                                                          test_frame = frame, 
                                                          bkg_frame= bkg_image, 
                                                          test_dp=self.test_dp, 
                                                          test_dg = self.test_dg, 
                                                          bkg_dp =self.bkg_dp, 
                                                          bkg_dg = self.bkg_dg, 
                                                          device=device)
            if "blur" in self.mf_method:
                self.psf = self.optics_model.gen_psf()
            else:
                self.psf = None
            
        bw = torch.ones(size=(self.Num_layer, ))
        df = self.discount_factor
        if isinstance(df, float):
            df = torch.tensor([df], device=device)

        if "none" in self.mf_method:
            bw[1:] = 0 ## Remove the blending weight 

        if "blur" in self.mf_method:
            fs_weight = torch.tensor([1, 0], device=device)
            return self.optics_model.get_image(bw[0]*frame, bw[1]*df*bkg_image, psf=self.psf ,mode='sum',fs_weight=fs_weight)
        elif "mean" in self.mf_method:
            return bw[0]*frame + bw[1]*df*torch.mean(bkg_image)*torch.ones_like(bkg_image)
        else:
            return bw[0]*frame + bw[1]*df*bkg_image
        
    def get_test_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        self.frame_idx = frame
        frame = self.add_background( super().get_test_frame(frame,device,colorspace="RGB2020"), device)
        return self.comp_dp.source_2_target_colourspace(frame, target_colorspace=colorspace)

    def get_reference_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        self.frame_idx = frame
        frame = self.add_background( super().get_reference_frame(frame,device,colorspace="RGB2020"), device)
        return self.comp_dp.source_2_target_colourspace(frame, target_colorspace=colorspace)


class ARDAVID(VideoDataset):
    """
    """
    n, h, w = -1, 1080, 1920    # Number of frames (maximum), height, width

    def __init__(self, path, mode, cache_loc=None, cache_yuv=True, random_parts=False, mf_method="pinhole", discount_factor = 1):
        super().__init__(path, mode, cache_loc, cache_yuv, random_parts)
        assert mf_method in ['none','mean','pinhole','pinhole-stereo','blur','blur-stereo'], f"Invalid fusion method {mf_method}" 
        self.mf_method = mf_method
        self.discount_factor = discount_factor


    def read_quality_csv(self, split):
        do_split = True

        df = pd.read_csv(osp.join(self.path, 'results_with_parts.csv'))

        self.quality_table = df

        self.disp_photo = pycvvdp.vvdp_display_photometry.load('eizo_CG3146-AR-DAVID', [self.path])
        self.disp_geom = pycvvdp.vvdp_display_geometry.load('eizo_CG3146-AR-DAVID', [self.path])
        self.foveated = False

        bkg_dg = pycvvdp.vvdp_display_geometry.load( "Dynascan", config_paths=[self.path] )
        # The ratio of the foreground pixel density to the background pixel density
        self.bkg_to_frg_scale = self.disp_geom.get_ppd()/bkg_dg.get_ppd()

        return do_split

    def get_condition_id(self, index):
        assert index in range(self.__len__()), f'{index} is out of range, len={self.__len__()}'
        scene, distortion, level, luminance, background = self.quality_table.iloc[index][['scene', 'distortion', 'level', 'luminance', 'background']]
        return f"{scene}_{distortion}_l{level}_{luminance}_{background}"

    def parse_row(self, row):
        scene, distortion, level, quality = row[['scene', 'distortion', 'level', 'jod']]
        test_fname = f'{scene}_{distortion}_Level00{level}'
        ref_fname = f'{scene}_reference_Level001'
        return test_fname, ref_fname, quality

    def get_part( self, index ):
        assert index in range(self.__len__()), f'{index} is out of range, len={self.__len__()}'
        return self.quality_table.iloc[index]['part_id_scene']

    def _get_video_source_yuv(self, test_fname_yuv, ref_fname_yuv, display_photometry, max_frames, row):
        assert False #not implemented
        return yuv.video_source_yuv_file(test_fname_yuv, ref_fname_yuv, display_photometry=display_photometry, frames=max_frames)

    def _get_video_source_mp4(self, test_fname_mp4, ref_fname_mp4, display_photometry, max_frames, row):
        try:
            probe = ffmpeg.probe(test_fname_mp4)
        except:
            raise RuntimeError("ffmpeg failed to open file \"" + test_fname_mp4 + "\"")

        # select the first video stream
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        total_frames = int(video_stream['nb_frames'])
        # avg_fps_num, avg_fps_denom = [float(x) for x in video_stream['r_frame_rate'].split("/")]
        # avg_fps = avg_fps_num/avg_fps_denom

        if max_frames == -1:
            max_frames = total_frames
        n_frames = min(max_frames, total_frames)

        bkg_pattern, bkg_luminance = row[['background', 'luminance']]
        return video_source_ardavid(test_fname_mp4, ref_fname_mp4, bkg_pattern=bkg_pattern, 
                                    bkg_luminance=bkg_luminance, display_photometry=display_photometry, 
                                    frames=n_frames, config_paths=[self.path], 
                                    bkg_to_frg_scale=self.bkg_to_frg_scale, 
                                    mf_method=self.mf_method, discount_factor = self.discount_factor)

    def get_ds_name(self):
        if math.isclose(self.discount_factor, 1.0, rel_tol=1e-3):
            return f"AR-DAVID_{self.mf_method}"
        else:
            return f"AR-DAVID_{self.mf_method}_discount_{self.discount_factor:.2f}"
        # return f"AR-DAVID"

#!/usr/bin/env python

#@markdown **Python Definitions**
import json
from IPython import display

import gc, math, os, pathlib, subprocess, sys, time
import cv2
import numpy as np
import pandas as pd
import random
import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from types import SimpleNamespace
from torch import autocast
import re
from scipy.ndimage import gaussian_filter
import yaml
from pathlib import Path
import os


sys.path.extend([
    'src/taming-transformers',
    'src/clip',
    'stable-diffusion/',
    'k-diffusion',
    'pytorch3d-lite',
    'AdaBins',
    'MiDaS',
])

import py3d_tools as p3d

from helpers import DepthModel, sampler_fn
from k_diffusion.external import CompVisDenoiser
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def DeforumAnimArgs():

    #@markdown ####**Animation:**
    animation_mode = '2D' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 0 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

    #@markdown ####**Motion Parameters:**
    angle = "0:(0.2)"#@param {type:"string"}
    zoom = "0:(1.005)"#@param {type:"string"}
    #translation_x = "0:(10*sin(2*3.14*t/10))"#@param {type:"string"}
    translation_x = "0:(0)"#@param {type:"string"}
    translation_y = "0:(1)"#@param {type:"string"}
    translation_z = "0:(0)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(0)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}
    flip_2d_perspective = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.02)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path ='/content/video_in.mp4'#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

    #@markdown ####**Interpolation:**
    interpolate_key_frames = False #@param {type:"boolean"}
    interpolate_x_frames = 4 #@param {type:"number"}
    
    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()


def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    import numexpr
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    
    for i in range(0, max_frames):
        if i in key_frames:
            value = key_frames[i]
            value_is_number = check_is_number(value)
            # if it's only a number, leave the rest for the default interpolation
            if value_is_number:
                t = i
                key_frame_series[i] = value
        if not value_is_number:
            t = i
            key_frame_series[i] = numexpr.evaluate(value)
    key_frame_series = key_frame_series.astype(float)
    
    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
      interp_method = 'Quadratic'    
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
      interp_method = 'Linear'
          
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def parse_key_frames(string, prompt_parser=None):
    # because math functions (i.e. sin(t)) can utilize brackets 
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    pattern = r'((?P<frame>[0-9]+):[\s]*\((?P<param>[\S\s]*?)\)([,][\s]?|[\s]?$))'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))
    return tmp.replace(' ', '_')

from functools import reduce
def construct_RotationMatrixHomogenous(rotation_angles):
    assert(type(rotation_angles)==list and len(rotation_angles)==3)
    RH = np.eye(4,4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

# https://en.wikipedia.org/wiki/Rotation_matrix
def getRotationMatrixManual(rotation_angles):
	
    rotation_angles = [np.deg2rad(x) for x in rotation_angles]
    
    phi         = rotation_angles[0] # around x
    gamma       = rotation_angles[1] # around y
    theta       = rotation_angles[2] # around z
    
    # X rotation
    Rphi        = np.eye(4,4)
    sp          = np.sin(phi)
    cp          = np.cos(phi)
    Rphi[1,1]   = cp
    Rphi[2,2]   = Rphi[1,1]
    Rphi[1,2]   = -sp
    Rphi[2,1]   = sp
    
    # Y rotation
    Rgamma        = np.eye(4,4)
    sg            = np.sin(gamma)
    cg            = np.cos(gamma)
    Rgamma[0,0]   = cg
    Rgamma[2,2]   = Rgamma[0,0]
    Rgamma[0,2]   = sg
    Rgamma[2,0]   = -sg
    
    # Z rotation (in-image-plane)
    Rtheta      = np.eye(4,4)
    st          = np.sin(theta)
    ct          = np.cos(theta)
    Rtheta[0,0] = ct
    Rtheta[1,1] = Rtheta[0,0]
    Rtheta[0,1] = -st
    Rtheta[1,0] = st
    
    R           = reduce(lambda x,y : np.matmul(x,y), [Rphi, Rgamma, Rtheta]) 
    
    return R


def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):
    
    ptsIn2D      =  ptsIn[0,:]
    ptsOut2D     =  ptsOut[0,:]
    ptsOut2Dlist =  []
    ptsIn2Dlist  =  []
    
    for i in range(0,4):
        ptsOut2Dlist.append([ptsOut2D[i,0], ptsOut2D[i,1]])
        ptsIn2Dlist.append([ptsIn2D[i,0], ptsIn2D[i,1]])
    
    pin  =  np.array(ptsIn2Dlist)   +  [W/2.,H/2.]
    pout = (np.array(ptsOut2Dlist)  +  [1.,1.]) * (0.5*sidelength)
    pin  = pin.astype(np.float32)
    pout = pout.astype(np.float32)
    
    return pin, pout

def warpMatrix(W, H, theta, phi, gamma, scale, fV):
    
    # M is to be estimated
    M          = np.eye(4, 4)
    
    fVhalf     = np.deg2rad(fV/2.)
    d          = np.sqrt(W*W+H*H)
    sideLength = scale*d/np.cos(fVhalf)
    h          = d/(2.0*np.sin(fVhalf))
    n          = h-(d/2.0);
    f          = h+(d/2.0);
    
    # Translation along Z-axis by -h
    T       = np.eye(4,4)
    T[2,3]  = -h
    
    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])
    
    
    # Projection Matrix 
    P       = np.eye(4,4)
    P[0,0]  = 1.0/np.tan(fVhalf)
    P[1,1]  = P[0,0]
    P[2,2]  = -(f+n)/(f-n)
    P[2,3]  = -(2.0*f*n)/(f-n)
    P[3,2]  = -1.0
    
    # pythonic matrix multiplication
    F       = reduce(lambda x,y : np.matmul(x,y), [P, T, R]) 
    
    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way. 
    # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
    ptsIn = np.array([[
                 [-W/2., H/2., 0.],[ W/2., H/2., 0.],[ W/2.,-H/2., 0.],[-W/2.,-H/2., 0.]
                 ]])
    ptsOut  = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut  = cv2.perspectiveTransform(ptsIn, F)
    
    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)
    
    # check float32 otherwise OpenCV throws an error
    assert(ptsInPt2f.dtype  == np.float32)
    assert(ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)

    return M33, sideLength

def anim_frame_warp_2d(prev_img_cv2, args, anim_args, frame_idx):
    p_dict = get_prompt_for_frame(frame_idx)
    angle = p_dict["angle"]
    zoom = p_dict["scale"]
    translation_x = p_dict["translation_x"]
    translation_y = p_dict["translation_y"]

    center = (args.W // 2, args.H // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def anim_frame_warp_3d(prev_img_cv2, depth, anim_args, keys, frame_idx):
    TRANSLATION_SCALE = 1.0/200.0 # matches Disco
    translate_xyz = [
        -keys.translation_x_series[frame_idx] * TRANSLATION_SCALE, 
        keys.translation_y_series[frame_idx] * TRANSLATION_SCALE, 
        -keys.translation_z_series[frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(keys.rotation_3d_x_series[frame_idx]), 
        math.radians(keys.rotation_3d_y_series[frame_idx]), 
        math.radians(keys.rotation_3d_z_series[frame_idx])
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    result = transform_image_3d(prev_img_cv2, depth, rot_mat, translate_xyz, anim_args)
    torch.cuda.empty_cache()
    return result

def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path,time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path

def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
      # Split alpha channel into a mask_image
      red, green, blue, alpha = Image.Image.split(image)
      mask_image = alpha.convert('L')
      image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.

    return image, mask_image

def load_mask_latent(mask_input, shape):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    
    if isinstance(mask_input, str): # mask input is probably a file name
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge, 
    #     0 is black, 1 is no adjustment, >1 is brighter
    # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image, 
    #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast
    
    mask = load_mask_latent(mask_input, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask,(4,1,1))
    mask = np.expand_dims(mask,axis=0)
    mask = torch.from_numpy(mask)

    if args.invert_mask:
        mask = ( (mask - 0.5) * -1) + 0.5
    
    mask = np.clip(mask,0,1)
    return mask

def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


#
# Callback functions
#
class SamplerCallback(object):
    # Creates the callback function to be passed into the samplers for each step
    def __init__(self, args, mask=None, init_latent=None, sigmas=None, sampler=None,
                  verbose=False):
        self.sampler_name = args.sampler
        self.dynamic_threshold = args.dynamic_threshold
        self.static_threshold = args.static_threshold
        self.mask = mask
        self.init_latent = init_latent 
        self.sigmas = sigmas
        self.sampler = sampler
        self.verbose = verbose

        self.batch_size = args.n_samples
        self.save_sample_per_step = args.save_sample_per_step
        self.show_sample_per_step = args.show_sample_per_step
        self.paths_to_image_steps = [os.path.join( args.outdir, f"{args.timestring}_{index:02}_{args.seed}") for index in range(args.n_samples) ]

        if self.save_sample_per_step:
            for path in self.paths_to_image_steps:
                os.makedirs(path, exist_ok=True)

        self.step_index = 0

        self.noise = None
        if init_latent is not None:
            self.noise = torch.randn_like(init_latent, device=device)

        self.mask_schedule = None
        if sigmas is not None and len(sigmas) > 0:
            self.mask_schedule, _ = torch.sort(sigmas/torch.max(sigmas))
        elif len(sigmas) == 0:
            self.mask = None # no mask needed if no steps (usually happens because strength==1.0)

        if self.sampler_name in ["plms","ddim"]: 
            if mask is not None:
                assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"

        if self.sampler_name in ["plms","ddim"]: 
            # Callback function formated for compvis latent diffusion samplers
            self.callback = self.img_callback_
        else: 
            # Default callback function uses k-diffusion sampler variables
            self.callback = self.k_callback_

        self.verbose_print = print if verbose else lambda *args, **kwargs: None

    def view_sample_step(self, latents, path_name_modifier=''):
        if self.save_sample_per_step or self.show_sample_per_step:
            samples = model.decode_first_stage(latents)
            if self.save_sample_per_step:
                fname = f'{path_name_modifier}_{self.step_index:05}.png'
                for i, sample in enumerate(samples):
                    sample = sample.double().cpu().add(1).div(2).clamp(0, 1)
                    sample = torch.tensor(np.array(sample))
                    grid = make_grid(sample, 4).cpu()
                    TF.to_pil_image(grid).save(os.path.join(self.paths_to_image_steps[i], fname))
            if self.show_sample_per_step:
                print(path_name_modifier)
                self.display_images(samples)
        return

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(np.array(images))
        grid = make_grid(images, 4).cpu()
        display.display(TF.to_pil_image(grid))
        return

    # The callback function is applied to the image at each step
    def dynamic_thresholding_(self, img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(self, args_dict):
        self.step_index = args_dict['i']
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(args_dict['x'], self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*self.static_threshold, self.static_threshold)
        if self.mask is not None:
            init_noise = self.init_latent + self.noise * args_dict['sigma']
            is_masked = torch.logical_and(self.mask >= self.mask_schedule[args_dict['i']], self.mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + args_dict['x'] * torch.where(is_masked,0,1)
            args_dict['x'].copy_(new_img)

        self.view_sample_step(args_dict['denoised'], "x0_pred")

    # Callback for Compvis samplers
    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(self, img, i):
        self.step_index = i
        # Thresholding functions
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(img, self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(img, -1*self.static_threshold, self.static_threshold)
        if self.mask is not None:
            i_inv = len(self.sigmas) - i - 1
            init_noise = self.sampler.stochastic_encode(self.init_latent, torch.tensor([i_inv]*self.batch_size).to(device), noise=self.noise)
            is_masked = torch.logical_and(self.mask >= self.mask_schedule[i], self.mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + img * torch.where(is_masked,0,1)
            img.copy_(new_img)

        self.view_sample_step(img, "x")

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def transform_image_3d(prev_img_cv2, depth_tensor, rot_mat, translate, anim_args):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion 
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    aspect_ratio = float(w)/float(h)
    near, far, fov_deg = anim_args.near_plane, anim_args.far_plane, anim_args.fov
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode=anim_args.sampling_mode, 
        padding_mode=anim_args.padding_mode, 
        align_corners=False
    )

    # convert back to cv2 style numpy array
    result = rearrange(
        new_image.squeeze().clamp(0,255), 
        'c h w -> h w c'
    ).cpu().numpy().astype(prev_img_cv2.dtype)
    return result

def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

# prompt weighting with colons and number coefficients (like 'bacon:0.75 eggs:0.25')
# borrowed from https://github.com/kylewlacy/stable-diffusion/blob/0a4397094eb6e875f98f9d71193e350d859c4220/ldm/dream/conditioning.py
# and https://github.com/raefu/stable-diffusion-automatic/blob/unstablediffusion/modules/processing.py
def get_uc_and_c(prompts, model, args, frame = 0):
    prompt = prompts[0] # they are the same in a batch anyway

    # get weighted sub-prompts
    negative_subprompts, positive_subprompts = split_weighted_subprompts(
        prompt, frame, not args.normalize_prompt_weights
    )

    uc = get_learned_conditioning(model, negative_subprompts, "", args, -1)
    c = get_learned_conditioning(model, positive_subprompts, prompt, args, 1)

    return (uc, c)

def get_learned_conditioning(model, weighted_subprompts, text, args, sign = 1):
    if len(weighted_subprompts) < 1:
        log_tokenization(text, model, args.log_weighted_subprompts, sign)
        c = model.get_learned_conditioning(args.n_samples * [text])
    else:
        c = None
        for subtext, subweight in weighted_subprompts:
            log_tokenization(subtext, model, args.log_weighted_subprompts, sign * subweight)
            if c is None:
                c = model.get_learned_conditioning(args.n_samples * [subtext])
                c *= subweight
            else:
                c.add_(model.get_learned_conditioning(args.n_samples * [subtext]), alpha=subweight)
        
    return c

def parse_weight(match, frame = 0)->float:
    import numexpr
    w_raw = match.group("weight")
    if w_raw == None:
        return 1
    if check_is_number(w_raw):
        return float(w_raw)
    else:
        t = frame
        if len(w_raw) < 3:
            print('the value inside `-characters cannot represent a math function')
            return 1
        return float(numexpr.evaluate(w_raw[1:-1]))

def normalize_prompt_weights(parsed_prompts):
    if len(parsed_prompts) == 0:
        return parsed_prompts
    weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    if weight_sum == 0:
        print(
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

def split_weighted_subprompts(text, frame = 0, skip_normalize=False):
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    prompt_parser = re.compile("""
            (?P<prompt>         # capture group for 'prompt'
            (?:\\\:|[^:])+      # match one or more non ':' characters or escaped colons '\:'
            )                   # end 'prompt'
            (?:                 # non-capture group
            :+                  # match one or more ':' characters
            (?P<weight>((        # capture group for 'weight'
            -?\d+(?:\.\d+)?     # match positive or negative integer or decimal number
            )|(                 # or
            `[\S\s]*?`# a math function
            )))?                 # end weight capture group, make optional
            \s*                 # strip spaces after weight
            |                   # OR
            $                   # else, if no ':' then match end of line
            )                   # end non-capture group
            """, re.VERBOSE)
    negative_prompts = []
    positive_prompts = []
    for match in re.finditer(prompt_parser, text):
        w = parse_weight(match, frame)
        if w < 0:
            # negating the sign as we'll feed this to uc
            negative_prompts.append((match.group("prompt").replace("\\:", ":"), -w))
        elif w > 0:
            positive_prompts.append((match.group("prompt").replace("\\:", ":"), w))

    if skip_normalize:
        return (negative_prompts, positive_prompts)
    return (normalize_prompt_weights(negative_prompts), normalize_prompt_weights(positive_prompts))

# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, log=False, weight=1):
    if not log:
        return
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(f"\n>> Tokens ({usedTokens}), Weight ({weight:.2f}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m"
        )

def generate(args, frame = 0, return_latent=False, return_sample=False, return_c=False):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sampler = PLMSSampler(model) if args.sampler == 'plms' else DDIMSampler(model)
    model_wrap = CompVisDenoiser(model)
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image, 
                                          shape=(args.W, args.H),  
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space        

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


        mask = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                            init_latent.shape, 
                            args.mask_contrast_adjust, 
                            args.mask_brightness_adjust)
        
        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")
        
        mask = mask.to(device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not ( (args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"
        
    t_enc = int((1.0-args.strength) * args.steps)

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(args.steps)
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if args.sampler in ['plms','ddim']:
        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, ddim_discretize='fill', verbose=False)

    callback = SamplerCallback(args=args,
                            mask=mask, 
                            init_latent=init_latent,
                            sigmas=k_sigmas,
                            sampler=sampler,
                            verbose=False).callback  

    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in data:
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    if args.prompt_weighting:
                        uc, c = get_uc_and_c(prompts, model, args, frame)
                    else:
                        uc = model.get_learned_conditioning(batch_size * [""])
                        c = model.get_learned_conditioning(prompts)


                    if args.scale == 1.0:
                        uc = None
                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                        samples = sampler_fn(
                            c=c, 
                            uc=uc, 
                            args=args, 
                            model_wrap=model_wrap, 
                            init_latent=init_latent, 
                            t_enc=t_enc, 
                            device=device, 
                            cb=callback)
                    else:
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        if init_latent is not None and args.strength > 0:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        else:
                            z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)
                        if args.sampler == 'ddim':
                            samples = sampler.decode(z_enc, 
                                                     c, 
                                                     t_enc, 
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=uc,
                                                     img_callback=callback)
                        elif args.sampler == 'plms': # no "decode" function in plms, so use "sample"
                            shape = [args.C, args.H // args.f, args.W // args.f]
                            samples, _ = sampler.sample(S=args.steps,
                                                            conditioning=c,
                                                            batch_size=args.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=args.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=args.ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=callback)
                        else:
                            raise Exception(f"Sampler {args.sampler} not recognised.")

                    
                    if return_latent:
                        results.append(samples.clone())

                    x_samples = model.decode_first_stage(samples)

                    if args.use_mask and args.overlay_mask:
                        # Overlay the masked image after the image is generated
                        if args.init_sample is not None:
                            img_original = args.init_sample
                        elif init_image is not None:
                            img_original = init_image
                        else:
                            raise Exception("Cannot overlay the masked image without an init image to overlay")

                        mask_fullres = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                                                    img_original.shape, 
                                                    args.mask_contrast_adjust, 
                                                    args.mask_brightness_adjust)
                        mask_fullres = mask_fullres[:,:3,:,:]
                        mask_fullres = repeat(mask_fullres, '1 ... -> b ...', b=batch_size)

                        mask_fullres[mask_fullres < mask_fullres.max()] = 0
                        mask_fullres = gaussian_filter(mask_fullres, args.mask_overlay_blur)
                        mask_fullres = torch.Tensor(mask_fullres).to(device)

                        x_samples = img_original * mask_fullres + x_samples * ((mask_fullres * -1.0) + 1)


                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    return results




def DeforumArgs():
    #@markdown **Image Settings**
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    #@markdown **Sampling Settings**
    seed = -1 #@param
    sampler = 'klms' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 50 #@param
    scale = 7 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None   

    #@markdown **Save & Display Settings**
    save_samples = True #@param {type:"boolean"}
    save_settings = True #@param {type:"boolean"}
    display_samples = True #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    prompt_weighting = False #@param {type:"boolean"}
    normalize_prompt_weights = True #@param {type:"boolean"}
    log_weighted_subprompts = False #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = 1 #@param
    batch_name = "StableFun" #@param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "random" #@param ["iter","fixed","random"]
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param 
    outdir = get_output_folder(output_path, batch_name)

    #@markdown **Init Settings**
    use_init = False #@param {type:"boolean"}
    strength = 0.0 #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False #@param {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 # {type:"number"}

    n_samples = 1 # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    return locals()



def next_seed(args):
    if args.seed_behavior == 'iter':
        args.seed += 1
    elif args.seed_behavior == 'fixed':
        pass # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed

def time2frame(str_time, fps):
    tokens = str_time.split(":")
    if len(tokens) > 3:
        print(f"time2frame(): ERROR: invalid time {str_time}")
        return None

    str_secs_frames = tokens[-1]
    sf_token = str_secs_frames.split(".")
    if len(sf_token) > 2 or len(sf_token) == 0:
        print(f"time2frame(): ERROR: invalid time {str_time}")
        return None
    elif len(sf_token) > 1:
        secs = int(sf_token[0])
        frames = int(sf_token[1])
    else:
        secs = int(sf_token[0])
        frames = 0

    if len(tokens) > 1:
        mins = int(tokens[-2])
    else:
        mins = 0

    if len(tokens) > 2:
        hours = int(tokens[-3])
    else:
        hours = 0

    total_frames = hours * 3600 * fps + mins * 60 * fps + secs * fps + frames
    #print(f"Total frames of {str_time}: {total_frames}")
    return total_frames


def multiforum_frame_dict(p):
    p_tokens = p.split("|")

    ccd_video = config_custom_data["default"]
    fps = config_custom_data["fps"]
    p_num_frame = time2frame(p_tokens[0], fps)
    p_prompt = p_tokens[-1]
    p_angle = float(p_tokens[1]) if len(p_tokens) > 2 else ccd_video["angle"]
    p_scale = float(p_tokens[2]) if len(p_tokens) > 3 else ccd_video["scale"]
    p_translation_x = float(p_tokens[3]) if len(p_tokens) > 4 else ccd_video["translation_x"]
    p_translation_y = float(p_tokens[4]) if len(p_tokens) > 5 else ccd_video["translation_y"]
    p_strength_schedule = float(p_tokens[5]) if len(p_tokens) > 6 else ccd_video["strength_schedule"]
    p_noise_schedule = ccd_video["noise_schedule"]
    p_contrast_schedule = ccd_video["contrast_schedule"]

    return {
        "num_frame": p_num_frame, 
        "prompt": p_prompt,
        "angle": p_angle, 
        "scale": p_scale,
        "translation_x": p_translation_x, 
        "translation_y": p_translation_y, 
        "strength_schedule": p_strength_schedule, 
        "noise_schedule": p_noise_schedule, 
        "contrast_schedule": p_contrast_schedule, 
    }


def multiforum_expand(prompts, config_cd):
    print("multiforum_expand(): starting")
    str_total_time = config_cd["total_time"]
    #print(f"multiforum_expand(): Total time: {str_total_time}")
    total_frames = time2frame(str_total_time, config_cd["fps"])
    #print(f"multiforum_expand(): Total frames: {total_frames}")
    prompt_series = list()
    for p in prompts:
        p_dict = multiforum_frame_dict(p)
        prompt_series.append(p_dict)

    return prompt_series


def get_prompt_for_frame(num_frame):
    prompts = config_custom_data["prompts"]
    for p in reversed(prompts):
        p_dict = multiforum_frame_dict(p)
        p_num_frame = p_dict["num_frame"]
        if num_frame >= p_num_frame:
            return p_dict

    print(f"get_prompt_for_frame(): ERROR: No frame for {num_frame}")
    return None
    

def render_animation(args, anim_args):
    # animations use key framed prompts
    args.prompts = animation_prompts

    # resume animation
    start_frame = 0
    if anim_args.resume_from_timestring:
        for tmp in os.listdir(args.outdir):
            if tmp.split("_")[0] == anim_args.resume_timestring:
                start_frame += 1
        start_frame = start_frame - 1

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")
    prompt_series = multiforum_expand(args.prompts, config_custom_data)

    args.n_samples = 1
    frame_idx = start_frame
    prev_sample = None
    color_match_sample = None
    while frame_idx < anim_args.max_frames:
        print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")
        prompt_cfg = get_prompt_for_frame(frame_idx)
        noise = prompt_cfg['noise_schedule']
        strength = prompt_cfg['strength_schedule']
        contrast = prompt_cfg['contrast_schedule']
        depth = None

        # apply transforms to previous frame
        if prev_sample is not None:
            if anim_args.animation_mode == '2D':
                prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), args, anim_args, frame_idx)

            # apply color matching
            if anim_args.color_coherence != 'None':
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

            # apply scaling
            contrast_sample = prev_img * contrast
            # apply frame noising
            noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

            # use transformed previous frame as init for current
            args.use_init = True
            if config_custom_data["precision"] == "fp16":
                args.init_sample = noised_sample.half().to(device)
            else:
                args.init_sample = noised_sample.to(device)
            args.strength = max(0.0, min(1.0, strength))

        # grab prompt for current frame
        prompt_cfg = get_prompt_for_frame(frame_idx)
        args.prompt = prompt_cfg["prompt"]
        print(f"{args.prompt} {args.seed}")

        print(f"Angle: {prompt_cfg['angle']} Zoom: {prompt_cfg['scale']}")
        print(f"Tx: {prompt_cfg['translation_x']} Ty: {prompt_cfg['translation_y']}")

        # sample the diffusion model
        sample, image = generate(args, frame_idx, return_latent=False, return_sample=True)
        prev_sample = sample

        filename = f"{args.timestring}_{frame_idx:05}.png"
        image.save(os.path.join(args.outdir, filename))
        frame_idx += 1

        display.clear_output(wait=True)
        display.display(image)

        args.seed = next_seed(args)

def vid2frames(video_path, frames_path, n=1, overwrite=True):      
    if not os.path.exists(frames_path) or overwrite: 
      try:
          for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
              f.unlink()
      except:
          pass
      assert os.path.exists(video_path), f"Video input {video_path} does not exist"
          
      vidcap = cv2.VideoCapture(video_path)
      success,image = vidcap.read()
      count = 0
      t=1
      success = True
      while success:
        if count % n == 0:
            cv2.imwrite(frames_path + os.path.sep + f"{t:05}.jpg" , image)     # save frame as JPEG file
            t += 1
        success,image = vidcap.read()
        count += 1
      print("Converted %d frames" % count)
    else: print("Frames already unpacked")


def get_model_device(custom_cd):
    #@markdown **Select and Load Model**

    models_path = custom_cd["models_path"]
    model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
    model_checkpoint =  "custom" #@param ["custom","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","waifu-diffusion-v1-3.ckpt"]
    custom_config_path = "" #@param {type:"string"}
    custom_checkpoint_path = custom_cd["custom_checkpoint_path"] #@param {type:"string"}

    load_on_run_all = True #@param {type: 'boolean'}
    half_precision = custom_cd["precision"] == "fp16" # check
    check_sha256 = True #@param {type:"boolean"}

    model_map = {
        "sd-v1-4-full-ema.ckpt": {
            'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-4-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-4.ckpt": {
            'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt',
            'requires_login': True,
            },
        "sd-v1-3-full-ema.ckpt": {
            'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/blob/main/sd-v1-3-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-3.ckpt": {
            'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt',
            'requires_login': True,
            },
        "sd-v1-2-full-ema.ckpt": {
            'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-2-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-2.ckpt": {
            'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2.ckpt',
            'requires_login': True,
            },
        "sd-v1-1-full-ema.ckpt": {
            'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829',
            'url':'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1-full-ema.ckpt',
            'requires_login': True,
            },
        "sd-v1-1.ckpt": {
            'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt',
            'requires_login': True,
            },
        "robo-diffusion-v1.ckpt": {
            'sha256': '244dbe0dcb55c761bde9c2ac0e9b46cc9705ebfe5f1f3a7cc46251573ea14e16',
            'url': 'https://huggingface.co/nousr/robo-diffusion/resolve/main/models/robo-diffusion-v1.ckpt',
            'requires_login': False,
            },
        "model-epoch05-float16.ckpt": {
            'sha256': '26cf2a2e30095926bb9fd9de0c83f47adc0b442dbfdc3d667d43778e8b70bece',
            'url': 'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/model-epoch05-float16.ckpt',
            'requires_login': False,
            },
    }

    # config path
    ckpt_config_path = custom_config_path if model_config == "custom" else os.path.join(models_path, model_config)
    if os.path.exists(ckpt_config_path):
        print(f"{ckpt_config_path} exists")
    else:
        print(f"{ckpt_config_path} does not exist")
        ckpt_config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    print(f"Using config: {ckpt_config_path}")

    # checkpoint path or download
    ckpt_path = custom_checkpoint_path if model_checkpoint == "custom" else os.path.join(models_path, custom_checkpoint_path)
    print(f"ckpt_path: {ckpt_path}")
    ckpt_valid = True
    if os.path.exists(ckpt_path):
        print(f"{ckpt_path} exists")
    elif 'url' in model_map[model_checkpoint]:
        url = model_map[model_checkpoint]['url']

        # CLI dialogue to authenticate download
        if model_map[model_checkpoint]['requires_login']:
            print("This model requires an authentication token")
            print("Please ensure you have accepted its terms of service before continuing.")

            username = input("What is your huggingface username?:")
            token = input("What is your huggingface token?:")

            _, path = url.split("https://")

            url = f"https://{username}:{token}@{path}"

        # contact server for model
        print(f"Attempting to download {model_checkpoint}...this may take a while")
        ckpt_request = requests.get(url)
        request_status = ckpt_request.status_code

        # inform user of errors
        if request_status == 403:
            raise ConnectionRefusedError("You have not accepted the license for this model.")
        elif request_status == 404:
            raise ConnectionError("Could not make contact with server")
        elif request_status != 200:
            raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

        # write to model path
        with open(os.path.join(models_path, model_checkpoint), 'wb') as model_file:
            model_file.write(ckpt_request.content)
    else:
        print(f"Please download model checkpoint and place in {os.path.join(models_path, model_checkpoint)}")
        ckpt_valid = False

    if check_sha256 and model_checkpoint != "custom" and ckpt_valid:
        import hashlib
        print("\n...checking sha256")
        with open(ckpt_path, "rb") as f:
            bytes = f.read() 
            hash = hashlib.sha256(bytes).hexdigest()
            del bytes
        if model_map[model_checkpoint]["sha256"] == hash:
            print("hash is correct\n")
        else:
            print("hash in not correct\n")
            ckpt_valid = False

    if ckpt_valid:
        print(f"Using ckpt: {ckpt_path}")

    def load_model_from_config(config, ckpt, verbose=False, device='cuda', half_precision=True):
        map_location = "cuda" #@param ["cpu", "cuda"]
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location=map_location)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        if half_precision:
            model = model.half().to(device)
        else:
            model = model.to(device)
        model.eval()
        return model

    if load_on_run_all and ckpt_valid:
        local_config = OmegaConf.load(f"{ckpt_config_path}")
        model = load_model_from_config(local_config, f"{ckpt_path}", half_precision=half_precision)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

    return model, device






























# BEGIN MAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# BEGIN MAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# BEGIN MAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print( " Initializing Multiforum")
if len(sys.argv) >= 2:
    override_settings_with_file = False
    custom_settings_file = sys.argv[1]
    print(f" *** Config filepath: {custom_settings_file}")
    config_custom_data = yaml.safe_load(Path('config.yml').read_text())
else:
    print(f"Usage: {sys.argv[0]} config.yml")
    config_custom_data = None

models_path = config_custom_data["models_path"]
output_path = config_custom_data["output_path"]

print(f" *** Models path: {models_path}")
print(f" *** Output path: {output_path}")

os.makedirs(models_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()
animation_prompts = config_custom_data["prompts"]

args_dict['H'] = config_custom_data["deforum_args"]["H"]
args_dict['W'] = config_custom_data["deforum_args"]["W"]

#@markdown **Load Settings**

#jdata = config_custom_data
#animation_prompts = jdata["prompts"]
#for i, k in enumerate(args_dict):
#    if k in jdata:
#        args_dict[k] = jdata[k]
#    else:
#        print(f"key {k} doesn't exist in the custom settings data! using the default value of {args_dict[k]}")
#        for i, k in enumerate(anim_args_dict):
#            if k in jdata:
#                anim_args_dict[k] = jdata[k]
#            else:
#                print(f"key {k} doesn't exist in the custom settings data! using the default value of {anim_args_dict[k]}")
        
#print(args_dict)
#print(anim_args_dict)
#print(f" *** config_custom_data: {config_custom_data}")



args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

args.timestring = time.strftime('%Y%m%d%H%M%S')
args.strength = max(0.0, min(1.0, args.strength))

if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
if not args.use_init:
    args.init_image = None
if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
    print(f"Init images aren't supported with PLMS yet, switching to KLMS")
    args.sampler = 'klms'
if args.sampler != 'ddim':
    args.ddim_eta = 0

if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

anim_args.max_frames = time2frame(config_custom_data["total_time"], config_custom_data["fps"])
# clean up unused memory
gc.collect()
torch.cuda.empty_cache()

model,device = get_model_device(config_custom_data)

# dispatch to appropriate renderer
if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
    render_animation(args, anim_args)

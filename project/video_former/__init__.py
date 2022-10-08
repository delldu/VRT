"""Video Matte Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos

from . import former

import pdb


def video_forward(model, lq, device, batch_size=10):
    batch_overlap_size = 2
    scale = model.upscale

    B, C, H, W = lq.size()  # (100, 3, 180, 320)
    C = C - 1 if model.nonblind_denoising else C

    stride = batch_size - batch_overlap_size  # 12 - 2 == 10
    d_list = list(range(0, B - batch_size, stride)) + [max(0, B - batch_size)]
    # d_list -- [0, 10, 20, 30, 40, 50, 60, 70, 80, 88]

    E = torch.zeros(B, C, H * scale, W * scale)
    W = torch.zeros(B, 1, 1, 1)

    progress_bar = tqdm(total=len(d_list))
    for i in d_list:
        progress_bar.update(1)

        lq_clip = lq[i : i + batch_size, :, :]
        out_clip = todos.model.tile_forward(
            model,
            device,
            lq_clip,
            h_tile_size=128,
            w_tile_size=128,
            overlap_size=20,
            scale=model.upscale,
        )

        chan = out_clip.shape[1]
        if chan > C:
            out_clip = out_clip[:, 0:C, :, :]

        out_clip_mask = torch.ones(min(batch_size, B), 1, 1, 1)

        E[i : i + batch_size, :, :, :].add_(out_clip)
        W[i : i + batch_size, :, :, :].add_(out_clip_mask)
    output = E.div_(W)

    return output.clamp(0.0, 1.0)


def get_zoom_model():
    """Create model."""

    device = todos.model.get_device()

    model_path = "models/video_zoom.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = former.zoom_model()
    todos.model.load(model, checkpoint, key="params")
    model = model.to(device)
    model.eval()

    # print(f"Running on {device} ...")
    # model = torch.jit.script(model)

    # todos.data.mkdir("output")
    # if not os.path.exists("output/video_zoom.torch"):
    #     model.save("output/video_zoom.torch")

    return model, device


def video_zoom_predict(input_file, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_zoom_model()

    print(f"  zoom {input_file}, save to {output_file} ...")
    lq_list = []

    def zoom_video_frame(no, data):
        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        lq_list.append(input_tensor)

    video.forward(callback=zoom_video_frame)

    lq = torch.cat(lq_list, dim=0)
    hq = video_forward(model, lq, device, batch_size=10)

    for i in range(hq.shape[0]):
        # save image
        output_tensor = hq[i, :, :, :]
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()
    return True


def get_deblur_model():
    """Create model."""

    device = todos.model.get_device()

    model_path = "models/video_deblur.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = former.deblur_model()
    todos.model.load(model, checkpoint, key="params")
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # model = torch.jit.script(model)

    # todos.data.mkdir("output")
    # if not os.path.exists("output/video_deblur.torch"):
    #     model.save("output/video_deblur.torch")

    return model, device


def video_deblur_predict(input_file, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_deblur_model()

    print(f"  deblur {input_file}, save to {output_file} ...")
    lq_list = []

    def deblur_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        lq_list.append(input_tensor)

    video.forward(callback=deblur_video_frame)

    lq = torch.cat(lq_list, dim=0)
    hq = video_forward(model, lq, device, batch_size=16)

    for i in range(hq.shape[0]):
        # save image
        output_tensor = hq[i, :, :, :]
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True


def get_denoise_model():
    """Create model."""

    device = todos.model.get_device()

    model_path = "models/video_denoise.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = former.denoise_model()
    todos.model.load(model, checkpoint, key="params")
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # model = torch.jit.script(model)

    # todos.data.mkdir("output")
    # if not os.path.exists("output/video_denoise.torch"):
    #     model.save("output/video_denoise.torch")

    return model, device


def video_denoise_predict(input_file, sigma, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_denoise_model()

    print(f"  denoise {input_file}, save to {output_file} ...")
    lq_list = []

    def denoise_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        input_tensor = todos.data.frame_totensor(data)
        input_tensor[:, 3:4, :, :] = sigma / 255.0  # Add noise strength
        lq_list.append(input_tensor)

    video.forward(callback=denoise_video_frame)

    lq = torch.cat(lq_list, dim=0)
    hq = video_forward(model, lq, device, batch_size=10)

    for i in range(hq.shape[0]):
        # save image
        output_tensor = hq[i, :, :, :]
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True

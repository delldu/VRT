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


def model_load(model, path):
    """Load model."""

    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    if "params" in state_dict.keys():
        state_dict = state_dict["params"]

    target_state_dict = model.state_dict()

    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            # print(n)
            raise KeyError(n)

def get_zoom_model():
    """Create model."""

    device = todos.model.get_device()

    model_path = "models/video_zoom.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = former.zoom_model()
    model_load(model, checkpoint)
    model = model.to(device)
    model.eval()

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/video_zoom.torch"):
        model.save("output/video_zoom.torch")

    return model, device


def video_zoom_service(input_file, output_file, targ):
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
    progress_bar = tqdm(total=video.n_frames)

    def zoom_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = todos.model.forward(model, device, input_tensor)
        temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=zoom_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    return True


def video_zoom_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.zoom(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_zoom_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_zoom", video_zoom_service, host, port)


def get_deblur_model():
    """Create model."""

    device = todos.model.get_device()

    model_path = "models/video_deblur.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = former.deblur_model()
    model_load(model, checkpoint)
    model = model.to(device)
    model.eval()

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/video_deblur.torch"):
        model.save("output/video_deblur.torch")

    return model, device


def video_deblur_service(input_file, output_file, targ):
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

    print(f"  deblur {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def deblur_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = todos.model.forward(model, device, input_tensor)
        temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=deblur_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    return True


def video_deblur_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.deblur(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_deblur_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_deblur", video_deblur_service, host, port)


def get_denoise_model():
    """Create model."""

    device = todos.model.get_device()

    model_path = "models/video_denoise.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = former.denoise_model()
    model_load(model, checkpoint)
    model = model.to(device)
    model.eval()

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/video_denoise.torch"):
        model.save("output/video_denoise.torch")

    return model, device


def video_denoise_service(input_file, output_file, targ):
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

    print(f"  denoise {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def denoise_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = todos.model.forward(model, device, input_tensor)
        temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=denoise_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    return True


def video_denoise_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.denoise(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_denoise_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_denoise", video_denoise_service, host, port)

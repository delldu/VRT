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


def clip_forward(model, lq, device):
    block_size = 128
    overlap_size = 20
    scale_factor = model.upscale

    # test patch by patch
    b, d, c, h, w = lq.size()
    # pp b, d, c, h, w -- (1, 12, 3, 180, 320)
    c = c - 1 if model.nonblind_denoising else c

    stride = block_size - overlap_size # 128 - 20 -- 108
    h_list = list(range(0, h - block_size, stride)) + [max(0, h - block_size)]
    # h_list -- [0, 52]
    w_list = list(range(0, w - block_size, stride)) + [max(0, w - block_size)]
    # w_list -- [0, 108, 192]

    E = torch.zeros(b, d, c, h * scale_factor, w * scale_factor)
    W = torch.zeros_like(E)

    for i in h_list:
        for j in w_list:
            # block_size -- 128
            in_patch = lq[..., i : i + block_size, j : j + block_size].to(device)
            # in_patch.size() -- [1, 6, 3, 128, 128], in_patch.mean() -- 0.4992

            with torch.no_grad():
                out_patch = model(in_patch).detach().cpu()
            # (Pdb) in_patch.size() -- [1, 12, 3, 128, 128]
            # out_patch.size() -- [1, 12, 3, 512, 512]

            out_patch_mask = torch.ones_like(out_patch)

            # overlap_size -- 20
            if i < h_list[-1]:
                out_patch[..., -overlap_size // 2 :, :] *= 0
                out_patch_mask[..., -overlap_size // 2 :, :] *= 0
                # out_patch[..., -overlap_size // 2 :, :].size() -- [1, 12, 3, 10, 512]
                # out_patch_mask[..., -overlap_size // 2 :, :].size() -- [1, 12, 3, 10, 512]
            if j < w_list[-1]:
                out_patch[..., :, -overlap_size // 2 :] *= 0
                out_patch_mask[..., :, -overlap_size // 2 :] *= 0
            if i > h_list[0]:
                out_patch[..., : overlap_size // 2, :] *= 0
                out_patch_mask[..., : overlap_size // 2, :] *= 0
            if j > w_list[0]:
                out_patch[..., :, : overlap_size // 2] *= 0
                out_patch_mask[..., :, : overlap_size // 2] *= 0

            E[..., i * scale_factor : (i + block_size) * scale_factor, j * scale_factor : (j + block_size) * scale_factor].add_(out_patch)
            W[..., i * scale_factor : (i + block_size) * scale_factor, j * scale_factor : (j + block_size) * scale_factor].add_(out_patch_mask)
    output = E.div_(W)
    return output

def video_forward(model, lq, device, block_size = 12):
    overlap_size = 2
    scale_factor = model.upscale

    b, d, c, h, w = lq.size() # (1, 100, 3, 180, 320)
    c = c - 1 if model.nonblind_denoising else c

    stride = block_size - overlap_size # 12 - 2 == 10
    d_list = list(range(0, d - block_size, stride)) + [max(0, d - block_size)]
    # d_list -- [0, 10, 20, 30, 40, 50, 60, 70, 80, 88]

    E = torch.zeros(b, d, c, h * scale_factor, w * scale_factor)
    W = torch.zeros(b, d, 1, 1, 1)

    progress_bar = tqdm(total=len(d_list))
    for i in d_list:
        progress_bar.update(1)
        # block_size -- 12
        lq_clip = lq[:, i : i + block_size, ...]
        out_clip = clip_forward(model, lq_clip, device)
        out_clip_mask = torch.ones((b, min(block_size, d), 1, 1, 1))

        E[:, i : i + block_size, ...].add_(out_clip)
        W[:, i : i + block_size, ...].add_(out_clip_mask)
    output = E.div_(W)

    return output


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
    lq_list = []
    def zoom_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        lq_list.append(input_tensor)
    video.forward(callback=zoom_video_frame)

    lq = torch.cat(lq_list, dim=0).unsqueeze(0)
    hq = video_forward(model, lq, device, block_size=12)

    for i in range(hq.shape[1]):
        # save image
        output_tensor = hq[:, i, ...].squeeze().clamp_(0, 1)
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

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

# def deblur_clip_forward(model, lq, device):
#     block_size = 128
#     overlap_size = 20

#     # test patch by patch
#     b, d, c, h, w = lq.size()
#     pdb.set_trace()
#     # pp b, d, c, h, w -- (1, 10, 3, 729, 1280)

#     stride = block_size - overlap_size # 128 - 20 -- 108
#     h_list = list(range(0, h - block_size, stride)) + [max(0, h - block_size)]
#     # h_list -- [0, 52]
#     w_list = list(range(0, w - block_size, stride)) + [max(0, w - block_size)]
#     # w_list -- [0, 108, 192]

#     E = torch.zeros(b, d, c, h, w)
#     W = torch.zeros_like(E)

#     for i in h_list:
#         for j in w_list:
#             # block_size -- 128
#             in_patch = lq[..., i : i + block_size, j : j + block_size].to(device)
#             # in_patch.size() -- [1, 6, 3, 128, 128], in_patch.mean() -- 0.4992
#             pdb.set_trace()

#             with torch.no_grad():
#                 out_patch = model(in_patch).detach().cpu()
#             # (Pdb) in_patch.size() -- [1, 12, 3, 128, 128]
#             # out_patch.size() -- [1, 12, 3, 512, 512]

#             out_patch_mask = torch.ones_like(out_patch)
#             pdb.set_trace()

#             # overlap_size -- 20
#             if i < h_list[-1]:
#                 out_patch[..., -overlap_size // 2 :, :] *= 0
#                 out_patch_mask[..., -overlap_size // 2 :, :] *= 0
#                 # out_patch[..., -overlap_size // 2 :, :].size() -- [1, 12, 3, 10, 512]
#                 # out_patch_mask[..., -overlap_size // 2 :, :].size() -- [1, 12, 3, 10, 512]
#             if j < w_list[-1]:
#                 out_patch[..., :, -overlap_size // 2 :] *= 0
#                 out_patch_mask[..., :, -overlap_size // 2 :] *= 0
#             if i > h_list[0]:
#                 out_patch[..., : overlap_size // 2, :] *= 0
#                 out_patch_mask[..., : overlap_size // 2, :] *= 0
#             if j > w_list[0]:
#                 out_patch[..., :, : overlap_size // 2] *= 0
#                 out_patch_mask[..., :, : overlap_size // 2] *= 0

#             pdb.set_trace()

#             E[..., i : i + block_size, j : j + block_size].add_(out_patch)
#             W[..., i : i + block_size, j : j + block_size].add_(out_patch_mask)
#     output = E.div_(W)
#     return output

# def deblur_video_forward(model, lq, device):
#     block_size = 10
#     overlap_size = 2

#     b, d, c, h, w = lq.size() # (1, 100, 3, 720, 1280)

#     stride = block_size - overlap_size # 12 - 2 == 10
#     d_list = list(range(0, d - block_size, stride)) + [max(0, d - block_size)]

#     E = torch.zeros(b, d, c, h, w)
#     W = torch.zeros(b, d, 1, 1, 1)

#     progress_bar = tqdm(total=len(d_list))
#     for i in d_list:
#         progress_bar.update(1)
#         lq_clip = lq[:, i : i + block_size, ...]
#         out_clip = deblur_clip_forward(model, lq_clip, device)
#         out_clip_mask = torch.ones((b, min(block_size, d), 1, 1, 1))

#         E[:, i : i + block_size, ...].add_(out_clip)
#         W[:, i : i + block_size, ...].add_(out_clip_mask)
#     output = E.div_(W)

#     return output


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
    lq_list = []
    def deblur_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        lq_list.append(input_tensor)
    video.forward(callback=deblur_video_frame)

    lq = torch.cat(lq_list, dim=0).unsqueeze(0)
    hq = video_forward(model, lq, device, block_size=10)

    for i in range(hq.shape[1]):
        # save image
        output_tensor = hq[:, i, ...].squeeze().clamp_(0, 1)
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)

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

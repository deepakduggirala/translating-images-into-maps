import os
from pathlib import Path
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib as mpl
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

import src
import src.data.collate_funcs
import src.model.network as networks
from src.data.dataloader import nuScenesMaps
from src.utils import MetricDict

from src.data.dataloader import nuScenesMaps, read_split
from argslib import parse_args, init


def load_checkpoint(args, model, ckpt_name=None, ckpt_epoch=None):
    model_dir = Path(args.savedir) / args.name
    if not ckpt_name:
        if ckpt_epoch:
            ckpt_name = f'checkpoint-{str(ckpt_epoch).zfill(4)}.pth.gz'
        else:
            checkpt_fn = sorted(
                [
                    f
                    for f in os.listdir(str(model_dir))
                    if os.path.isfile(os.path.join(model_dir, f)) and ".pth.gz" in f
                ]
            )
            ckpt_name = checkpt_fn[-1]
    ckpt_fname = model_dir / ckpt_name
    print(f'Loading checkpoint {ckpt_name}')
    assert (ckpt_fname.exists())

    model_pth = os.path.join(model_dir, str(ckpt_fname))
    ckpt = torch.load(model_pth)
    model.load_state_dict(ckpt["model"])


def get_model(args):
    # Build model
    model = networks.__dict__[args.model_name](
        num_classes=len(args.pred_classes_nusc),
        frontend=args.frontend,
        grid_res=args.grid_res,
        pretrained=args.pretrained,
        img_dims=args.desired_image_size,
        z_range=args.z_intervals,
        h_cropped=args.cropped_height,
        dla_norm=args.dla_norm,
        additions_BEVT_linear=args.bevt_linear_additions,
        additions_BEVT_conv=args.bevt_conv_additions,
        dla_l1_n_channels=args.dla_l1_nchannels,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
    )

    device = torch.device("cuda")
    model = nn.DataParallel(model)
    _ = model.to(device)

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return model


def time_val_epoch(N, fname=None):
    i = 0

    # Print summary
    while True:
        start_time = time.perf_counter()
        yield
        i = i + 1
        batch_time = (time.perf_counter() - start_time)
        eta = (N - i) * batch_time

        s = "[Val: {:4d}/{:4d}] batch_time: {:.2f}s eta: {:s}".format(
            i, N, batch_time, str(timedelta(seconds=int(eta)))
        )

        if fname:
            with open(fname, "a") as fp:
                fp.write(s + '\n')
        print(s)


def evaluate(model, dataloader):
    model.eval()
    res_100 = None
    time_epoch_gen = time_val_epoch(N=len(dataloader))
    next(time_epoch_gen)
    for i, ((image, calib, grid2d), (cls_map, vis_mask)) in enumerate(dataloader):
        with torch.no_grad():
            pred_ms = model(image, calib, grid2d)

            # Upsample the largest prediction to 200x200
            pred_200x200 = F.interpolate(
                pred_ms[0], size=(200, 200), mode="bilinear"
            )
            pred_ms = [pred_200x200, *pred_ms]
            pred_ms_cpu = [pred.detach().cpu() for pred in pred_ms]

            if res_100 is not None:
                res_100 = torch.cat((res_100, pred_ms_cpu[1]))
            else:
                res_100 = torch.cat((pred_ms_cpu[1],))

        next(time_epoch_gen)

    return res_100


def main():
    args = parse_args(notebook=False)
    init(args)

    val_data = nuScenesMaps(
        root=args.root,
        split=args.val_split,
        grid_size=args.grid_size,
        grid_res=args.grid_res,
        classes=args.load_classes_nusc,
        dataset_size=args.data_size,
        desired_image_size=args.desired_image_size,
        mini=False,
        gt_out_size=(100, 100),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=src.data.collate_funcs.collate_nusc_s,
        drop_last=True,
        pin_memory=True
    )


    for ckpt_epoch in list(range(31, 41)):
        model = get_model(args)
        load_checkpoint(args, model, ckpt_epoch=ckpt_epoch)
        res_100 = evaluate(model, val_loader)

        experiment_dir = Path(args.savedir) / args.name
        results_dir = experiment_dir / 'inference_results'
        results_dir.mkdir(exist_ok=True)
        torch.save(res_100, results_dir / f'ckpt-{ckpt_epoch}-val-pred-100x100.pt')

if __name__=='__main__':
    main()
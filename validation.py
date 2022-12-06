import argparse
import json
import os
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src
import src.data.collate_funcs
import src.model.network as networks
from src.data.dataloader import nuScenesMaps
from src.utils import MetricDict


def validate(args, dataloader, model, epoch=0):
    print("\n==> Validating on {} minibatches\n".format(len(dataloader)))
    model.eval()
    epoch_loss = MetricDict()
    epoch_iou = MetricDict()
    epoch_loss_per_class = MetricDict()
    num_classes = len(args.pred_classes_nusc)
    times = []

    for i, ((image, calib, grid2d), (cls_map, vis_mask)) in enumerate(dataloader):
        # Move tensors to GPU
        image, calib, cls_map, vis_mask, grid2d = (
            image.cuda(),
            calib.cuda(),
            cls_map.cuda(),
            vis_mask.cuda(),
            grid2d.cuda(),
        )

        with torch.no_grad():
            # Run network forwards
            pred_ms = model(image, calib, grid2d)

            # Upsample largest prediction to 200x200
            pred_200x200 = F.interpolate(
                pred_ms[0], size=(200, 200), mode="bilinear"
            )
            # pred_200x200 = (pred_200x200 > 0).float()
            pred_ms = [pred_200x200, *pred_ms]

            # Get required gt output sizes
            map_sizes = [pred.shape[-2:] for pred in pred_ms]

            # Convert ground truth to binary mask
            gt_s1 = (cls_map > 0).float()
            vis_mask_s1 = (vis_mask > 0.5).float()

            # Downsample to match model outputs
            gt_ms = src.utils.downsample_gt(gt_s1, map_sizes)
            vis_ms = src.utils.downsample_gt(vis_mask_s1, map_sizes)

            # Compute IoU
            iou_per_sample, iou_dict = src.utils.compute_multiscale_iou(
                pred_ms, gt_ms, vis_ms, num_classes
            )
            # Compute per class loss for eval
            per_class_loss_dict = src.utils.compute_multiscale_loss_per_class(
                pred_ms, gt_ms,
            )

            epoch_iou += iou_dict
            epoch_loss_per_class += per_class_loss_dict

            # Visualize predictions
            # if epoch % args.val_interval * 4 == 0 and i % 50 == 0:
            #     vis_img = ToPILImage()(image[0].detach().cpu())
            #     pred_vis = pred_ms[1].detach().cpu()
            #     label_vis = gt_ms[1]
            #
            #     # Visualize scores
            #     vis_fig = visualize_score(
            #         pred_vis[0],
            #         label_vis[0],
            #         grid2d[0],
            #         vis_img,
            #         iou_per_sample[0],
            #         num_classes,
            #     )
            #     plt.savefig(
            #         os.path.join(
            #             args.savedir,
            #             args.name,
            #             "val_output_epoch{}_iter{}.png".format(epoch, i),
            #         )
            #     )

    print("\n==> Validation epoch complete")

    # Calculate per class IoUs over set
    scales = [pred.shape[-1] for pred in pred_ms]

    ms_cumsum_iou_per_class = torch.stack(
        [epoch_iou["s{}_iou_per_class".format(scale)] for scale in scales]
    )
    ms_count_per_class = torch.stack(
        [epoch_iou["s{}_class_count".format(scale)] for scale in scales]
    )

    ms_ious_per_class = (
        (ms_cumsum_iou_per_class / (ms_count_per_class + 1e-6)).cpu().numpy()
    )
    ms_mean_iou = ms_ious_per_class.mean(axis=1)

    # Calculate per class loss over set
    ms_cumsum_loss_per_class = torch.stack(
        [epoch_loss_per_class["s{}_loss_per_class".format(scale)] for scale in scales]
    )
    ms_loss_per_class = (
        (ms_cumsum_loss_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    total_loss = ms_loss_per_class.mean(axis=1).sum()

    with open(os.path.join(args.savedir, args.name, "val_loss.txt"), "a") as f:
        f.write("\n")
        f.write(
            "{},".format(epoch)
            + "{},".format(float(total_loss))
            + "".join("{},".format(v) for v in ms_mean_iou)
        )

    with open(os.path.join(args.savedir, args.name, "val_ious.txt"), "a") as f:
        f.write("\n")
        f.write(
            "Epoch: {},\n".format(epoch)
            + "Total Loss: {},\n".format(float(total_loss))
            + "".join(
                "s{}_ious_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_ious_per_class)
            )
            + "".join(
                "s{}_loss_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_loss_per_class)
            )
        )


def parse_args():
    parser = ArgumentParser()

    # ----------------------------- Data options ---------------------------- #
    parser.add_argument(
        "--root",
        type=str,
        default="nuscenes_data",
        help="root directory of the dataset",
    )
    parser.add_argument(
        "--nusc-version", type=str, default="v1.0-trainval", help="nuscenes version",
    )
    parser.add_argument(
        "--occ-gt",
        type=str,
        default="200down100up",
        help="occluded (occ) or unoccluded(unocc) ground truth maps",
    )
    parser.add_argument(
        "--gt-version",
        type=str,
        default="semantic_maps_new_200x200",
        help="ground truth name",
    )
    parser.add_argument(
        "--train-split", type=str, default="train_mini", help="ground truth name",
    )
    parser.add_argument(
        "--val-split", type=str, default="val_mini", help="ground truth name",
    )
    parser.add_argument(
        "--data-size",
        type=float,
        default=0.2,
        help="percentage of dataset to train on",
    )
    parser.add_argument(
        "--load-classes-nusc",
        type=str,
        nargs=14,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "road_segment",
            "lane",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to load for NuScenes",
    )
    parser.add_argument(
        "--pred-classes-nusc",
        type=str,
        nargs=12,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to predict for NuScenes",
    )
    parser.add_argument(
        "--lidar-ray-mask",
        type=str,
        default="dense",
        help="sparse or dense lidar ray visibility mask",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        nargs=2,
        default=(50.0, 50.0),
        help="width and depth of validation grid, in meters",
    )
    parser.add_argument(
        "--z-intervals",
        type=float,
        nargs="+",
        default=[1.0, 9.0, 21.0, 39.0, 51.0],
        help="depths at which to predict BEV maps",
    )
    parser.add_argument(
        "--grid-jitter",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="magn. of random noise applied to grid coords",
    )
    parser.add_argument(
        "--aug-image-size",
        type=int,
        nargs="+",
        default=[1280, 720],
        help="size of random image crops during training",
    )
    parser.add_argument(
        "--desired-image-size",
        type=int,
        nargs="+",
        default=[1600, 900],
        help="size images are padded to before passing to network",
    )
    parser.add_argument(
        "--yoffset",
        type=float,
        default=1.74,
        help="vertical offset of the grid from the camera axis",
    )

    # -------------------------- Model options -------------------------- #
    parser.add_argument(
        "--model-name",
        type=str,
        default="PyrOccTranDetr_S_0904_old_rep100x100_out100x100",
        help="Model to train",
    )
    parser.add_argument(
        "-r",
        "--grid-res",
        type=float,
        default=0.5,
        help="size of grid cells, in meters",
    )
    parser.add_argument(
        "--frontend",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="name of frontend ResNet architecture",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="choose pretrained frontend ResNet",
    )
    parser.add_argument(
        "--pretrained-bem",
        type=bool,
        default=False,
        help="choose pretrained BEV estimation model",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="iccv_segdet_pyrocctrandetr_s_0904_100x100_200down100up_dice_adam_lr5e5_di3_1600x900",
        help="name of pretrained model to load",
    )
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default="checkpoint-0020.pth.gz",
        help="name of checkpoint to load",
    )
    parser.add_argument(
        "--ignore", type=str, default=["nothing"], help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--ignore-reload",
        type=str,
        default=["nothing"],
        help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--focal-length", type=float, default=1266.417, help="focal length",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs=4,
        default=[8.0, 16.0, 32.0, 64.0],
        help="resnet frontend scale factor",
    )
    parser.add_argument(
        "--cropped-height",
        type=float,
        nargs=4,
        default=[20.0, 20.0, 20.0, 20.0],
        help="resnet feature maps cropped height",
    )
    parser.add_argument(
        "--y-crop",
        type=float,
        nargs=4,
        default=[15, 15.0, 15.0, 15.0],
        help="Max y-dimension in world space for all depth intervals",
    )
    parser.add_argument(
        "--dla-norm",
        type=str,
        default="GroupNorm",
        help="Normalisation for inputs to topdown network",
    )
    parser.add_argument(
        "--bevt-linear-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to linear layer in BEVT",
    )
    parser.add_argument(
        "--bevt-conv-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to conv layer in BEVT",
    )
    parser.add_argument(
        "--dla-l1-nchannels",
        type=int,
        default=64,
        help="vertical offset of the grid from the camera axis",
    )
    parser.add_argument(
        "--n-enc-layers",
        type=int,
        default=2,
        help="number of transfomer encoder layers",
    )
    parser.add_argument(
        "--n-dec-layers",
        type=int,
        default=2,
        help="number of transformer decoder layers",
    )

    # ---------------------------- Loss options ---------------------------- #
    parser.add_argument(
        "--loss", type=str, default="dice_loss_mean", help="Loss function",
    )
    parser.add_argument(
        "--exp-cf",
        type=float,
        default=0.0,
        help="Exponential for class frequency in weighted dice loss",
    )
    parser.add_argument(
        "--exp-os",
        type=float,
        default=0.2,
        help="Exponential for object size in weighted dice loss",
    )

    # ------------------------ Optimization options ----------------------- #
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("-l", "--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.99,
        help="factor to decay learning rate by every epoch",
    )

    # ------------------------- Training options ------------------------- #
    parser.add_argument(
        "-e", "--epochs", type=int, default=600, help="number of epochs to train for"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, help="mini-batch size for training"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=5,
        help="Gradient accumulation over number of batches",
    )

    # ------------------------ Experiment options ----------------------- #
    parser.add_argument(
        "--name", type=str,
        default="tiim_220613",
        help="name of experiment",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="experiments",
        help="directory to save experiments to",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        nargs="*",
        default=[0],
        help="ids of gpus to train on. Leave empty to use cpu",
    )
    parser.add_argument(
        "--num-gpu", type=int, default=1, help="number of gpus",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="number of worker threads to use for data loading",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="number of epochs between validation runs",
    )
    parser.add_argument(
        "--print-iter",
        type=int,
        default=5,
        help="print loss summary every N iterations",
    )
    parser.add_argument(
        "--vis-iter",
        type=int,
        default=20,
        help="display visualizations every N iterations",
    )
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _make_experiment(args):
    print("\n" + "#" * 80)
    print(datetime.now().strftime("%A %-d %B %Y %H:%M"))
    print(
        "Creating experiment '{}' in directory:\n  {}".format(args.name, args.savedir)
    )
    print("#" * 80)
    print("\nConfig:")
    for key in sorted(args.__dict__):
        print("  {:12s} {}".format(key + ":", args.__dict__[key]))
    print("#" * 80)

    # Create a new directory for the experiment
    savedir = os.path.join(args.savedir, args.name)
    os.makedirs(savedir, exist_ok=True)

    # # Create tensorboard summary writer
    summary = SummaryWriter(savedir)

    # # Save configuration to file
    with open(os.path.join(savedir, "config.json"), "w") as fp:
        json.dump(args.__dict__, fp)

    # # Write config as a text summary
    # summary.add_text(
    #     "config",
    #     "\n".join("{:12s} {}".format(k, v) for k, v in sorted(args.__dict__.items())),
    # )
    # summary.file_writer.flush()

    return None


def main():
    # Parse command line arguments
    args = parse_args()
    # args.root = os.path.join(os.getcwd(), args.root)
    print(args.root)
    args.savedir = os.path.join(os.getcwd(), args.savedir)
    print(args.savedir)

    # Build depth intervals along Z axis and reverse
    z_range = args.z_intervals
    args.grid_size = (z_range[-1] - z_range[0], z_range[-1] - z_range[0])

    # Calculate cropped heights of feature maps
    h_cropped = src.utils.calc_cropped_heights(
        args.focal_length, np.array(args.y_crop), z_range, args.scales
    )
    args.cropped_height = [h for h in h_cropped]

    num_gpus = torch.cuda.device_count()
    args.num_gpu = num_gpus

    ### Create experiment ###
    summary = _make_experiment(args)

    print("loading val data")
    val_data = nuScenesMaps(
        root=args.root,
        split=args.val_split,
        grid_size=args.grid_size,
        grid_res=args.grid_res,
        classes=args.load_classes_nusc,
        dataset_size=args.data_size,
        desired_image_size=args.desired_image_size,
        mini=False,
        gt_out_size=(200, 200),
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

    # Build model
    model = networks.__dict__[args.model_name](
        num_classes=len(args.pred_classes_nusc),
        frontend=args.frontend,
        grid_res=args.grid_res,
        pretrained=args.pretrained,
        img_dims=args.desired_image_size,
        z_range=z_range,
        h_cropped=args.cropped_height,
        dla_norm=args.dla_norm,
        additions_BEVT_linear=args.bevt_linear_additions,
        additions_BEVT_conv=args.bevt_conv_additions,
        dla_l1_n_channels=args.dla_l1_nchannels,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
    )

    if args.pretrained_bem:
        pretrained_model_dir = os.path.join(args.savedir, args.pretrained_model)
        # pretrained_ckpt_fn = sorted(
        #     [
        #         f
        #         for f in os.listdir(pretrained_model_dir)
        #         if os.path.isfile(os.path.join(pretrained_model_dir, f))
        #         and ".pth.gz" in f
        #     ]
        # )
        pretrained_pth = os.path.join(pretrained_model_dir, args.load_ckpt)
        pretrained_dict = torch.load(pretrained_pth)["model"]
        mod_dict = OrderedDict()

        # # Remove "module" from name
        for k, v in pretrained_dict.items():
            if any(module in k for module in args.ignore):
                continue
            else:
                name = k[7:]
                mod_dict[name] = v

        model.load_state_dict(mod_dict, strict=False)
        print("loaded pretrained model")

    device = torch.device("cuda")
    model = nn.DataParallel(model)
    model.to(device)

    # Setup optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), args.lr, )
    else:
        optimizer = optim.__dict__[args.optimizer](
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    # Check if saved model checkpoint exists
    model_dir = os.path.join(args.savedir, args.name)
    checkpt_fn = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f)) and ".pth.gz" in f
        ]
    )
    if len(checkpt_fn) != 0:
        model_pth = os.path.join(model_dir, checkpt_fn[-1])
        ckpt = torch.load(model_pth)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoch_ckpt = ckpt["epoch"] + 1
        print("starting training from {}".format(checkpt_fn[-1]))
    else:
        epoch_ckpt = 1
        pass

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    validate(args, val_loader, model)


if __name__ == "__main__":
    main()

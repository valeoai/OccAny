# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
import os
os.environ['OMP_NUM_THREADS'] = '3' # will affect the performance of pairwise prediction
os.environ['TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS'] = '1' # fix tensor.item() graph breaks

import argparse
import datetime
import json
import numpy as np
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
import logging
logger = logging.getLogger(__name__)

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# noqa: F401, needed when loading the model
from occany.datasets import get_data_loader  # noqa
from dust3r.inference import loss_of_one_batch, visualize_results, visualize_semantic  # noqa
from dust3r.inference_multiview import loss_of_one_batch as loss_of_one_batch_multiview_fast3r # noqa

from dust3r.losses import *  # noqa: F401, needed when loading the model
from occany.loss.losses import *  # noqa: F401, needed when loading the model
from occany.loss.losses_multiview import *  # noqa: F401, needed when loading the model

from occany.model.model_sam2 import SAM2
from occany.model.model_must3r import (
    CausalMust3rDecoder,
    Dust3rEncoder,
    Must3rDecoder,
    RaymapEncoderDiT,
)
from occany.must3r_inference import loss_of_one_batch_occany, loss_of_one_batch_occany_gen
from occany.utils.helpers import depth2rgb
from dust3r.utils.geometry import geotrf
from occany.model.must3r_blocks.head import ActivationType
 

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention

import occany.utils.io as checkpoints
import occany.model.must3r_blocks.optimizer as optim
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules
from itertools import chain
from PIL import Image
def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion

    parser.add_argument('--distill_model', default=None, \
        choices=['SAM2_large'],
        type=str, help="distillation model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--pretrained_occany', default=None, help='path of a starting checkpoint')
    parser.add_argument('--pretrained_occany_gen', default=None, help='path of a starting checkpoint')

    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--train_criterion_gen', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion gen")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")
    parser.add_argument('--test_criterion_gen', default=None, type=str, help="test criterion gen")
    parser.add_argument('--distill_criterion', default=None, type=str, help="distill criterion")
    parser.add_argument('--finetune_encoder', default=False, action='store_true', help="Also finetune dust3r's encoder")
    parser.add_argument('--decoder', default="CausalMust3rDecoder(img_size=(512, 512), \
        enc_embed_dim=1024, \
        embed_dim=768, \
        pointmaps_activation=ActivationType.LINEAR, \
        feedback_type='single_mlp', memory_mode='kv', mem_dropout=0.1, \
        dropout_mode='temporary', use_xformers_mask=True, use_mem_mask=True)", type=str, help="decoder (supports img_size=(512,512) or (768,768))")

    # dataset
    parser.add_argument('--train_dataset', default='[None]', type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--amp', choices=[False, "bf16", "fp16"], default=False,
                        help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--fixed_eval_set", action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
    parser.add_argument('--num_save_visual', default=1, type=int, help='number of visualizations to save')
    parser.add_argument('--img_size', default=512, type=int, help='image size')
    parser.add_argument('--sam_proj_lr_mult', type=float, default=1.0,
                        help='LR multiplier for SAM head projection parameters')
    # switch mode for train / eval pose / eval depth
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')

    # for pose eval
    parser.add_argument('--pose_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--pose_eval_stride', default=1, type=int, help='stride for pose evaluation')
    parser.add_argument('--scene_graph_type', default='swinstride-5-noncyclic', type=str, help='scene graph window size')
    parser.add_argument('--save_best_pose', action='store_true', default=False, help='save best pose')
    parser.add_argument('--n_iter', default=300, type=int, help='number of iterations for pose optimization')
    parser.add_argument('--save_pose_qualitative', action='store_true', default=False, help='save qualitative pose results')
    parser.add_argument('--temporal_smoothing_weight', default=0.01, type=float, help='temporal smoothing weight for pose optimization')
    parser.add_argument('--not_shared_focal', action='store_true', default=False, help='use shared focal length for pose optimization')
    parser.add_argument('--use_gt_focal', action='store_true', default=False, help='use ground truth focal length for pose optimization')
    parser.add_argument('--pose_schedule', default='linear', type=str, help='pose optimization schedule')

    parser.add_argument('--flow_loss_weight', default=0.01, type=float, help='flow loss weight for pose optimization')
    parser.add_argument('--flow_loss_fn', default='smooth_l1', type=str, help='flow loss type for pose optimization')
    parser.add_argument('--use_gt_mask', action='store_true', default=False, help='use gt mask for pose optimization, for sintel/davis')
    parser.add_argument('--motion_mask_thre', default=0.35, type=float, help='motion mask threshold for pose optimization')
    parser.add_argument('--sam2_mask_refine', action='store_true', default=False, help='use sam2 mask refine for the motion for pose optimization')
    parser.add_argument('--flow_loss_start_epoch', default=0.1, type=float, help='start epoch for flow loss')
    parser.add_argument('--flow_loss_thre', default=20, type=float, help='threshold for flow loss')
    parser.add_argument('--pxl_thresh', default=50.0, type=float, help='threshold for flow loss')
    parser.add_argument('--depth_regularize_weight', default=0.0, type=float, help='depth regularization weight for pose optimization')
    parser.add_argument('--translation_weight', default=1, type=float, help='translation weight for pose optimization')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode for pose evaluation')
    parser.add_argument('--full_seq', action='store_true', default=False, help='use full sequence for pose evaluation')
    parser.add_argument('--seq_list', nargs='+', default=None, help='list of sequences for pose evaluation')
    parser.add_argument('--time_cond', action='store_true', default=False, help='use time condition for pose optimization')
    parser.add_argument('--gen', action='store_true', default=False,
                        help='activate generation training with raymap prediction')
    parser.add_argument('--use_raymap_only_conditioning', action='store_true', default=False, help='use only raymap as conditioning instead of projected point features')
    parser.add_argument('--projection_features', type=str, default='pts3d_local,pts3d,rgb,conf,sam', 
                        help='comma-separated list of projection features to use: pts3d_local,pts3d,rgb,conf,sam,raymap (sam includes sam_256,sam_64,sam_32)')

    parser.add_argument('--loss_enc_feat', action='store_true', default=False, help='use encoder feature loss')
    parser.add_argument('--multiview', action='store_true', default=False, help='use multiview loss')
    parser.add_argument('--eval_dataset', type=str, default='sintel',
                    choices=['davis', 'kitti', 'bonn', 'scannet', 'tum', 'nyu', 'sintel'],
                    help='choose dataset for pose evaluation')

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False, help='do not crop the image for monocular depth evaluation')

    # output dir
    parser.add_argument('--sam_model', default='SAM2', choices=['SAM2'], type=str, help='SAM model version')
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path where to save the output")
    return parser


def load_distill_model(args, device):
    if args.distill_model is None:
        return None
    print('Loading distillation model: {:s}'.format(args.distill_model))
    if args.distill_model == 'SAM2_large':
        checkpoint_path = "./checkpoints/sam2.1_hiera_large.pt"
        config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    else:
        raise ValueError("Unknown distillation model: {:s}".format(args.distill_model))

    print('SAM2 model supports multiple resolutions including 512x512 and 768x768')
    distill_model = SAM2(
        model_cfg=str(config_path),
        sam2_checkpoint=str(checkpoint_path),
        device=device,
        image_size=args.img_size
    )
    distill_model.to(device)
    for param in distill_model.parameters():
        param.requires_grad = False
    distill_model.eval()
    return distill_model

def get_dtype(args):
    if args.amp:
        dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    else:
        dtype = torch.float32
    return dtype


def resolve_resume_checkpoint(args):
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f'Using explicit resume checkpoint: {args.resume}')
        return args.resume

    last_ckpt_fname = os.path.join(args.output_dir, 'checkpoint-last.pth')
    if os.path.isfile(last_ckpt_fname):
        print(f'Auto-resuming from: {last_ckpt_fname}')
        return last_ckpt_fname
    return None


def train(args):
    misc.init_distributed_mode_jz(args)

    
    toggle_memory_efficient_attention(enabled=True)
    register_legacy_checkpoint_modules()

    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    logger.info(f"global_rank: {global_rank}, world_size: {world_size}")
    # if main process, init wandb
    #     wandb.init(name=args.output_dir.split('/')[-1],

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    last_ckpt_fname = resolve_resume_checkpoint(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print(f"training_mode: {'gen' if args.gen else 'recon'}")

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed mode requires CUDA but no GPU was detected.")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = args.cudnn_benchmark

    
    distill_model = None
    distill_criterion = None
    if args.distill_model is not None:
        distill_model = load_distill_model(args, device)
        distill_criterion = eval(args.distill_criterion)
  
    img_encoder = Dust3rEncoder()
    print("use_time_cond: {}".format(args.time_cond))

    
    # Note: The decoder supports flexible image sizes (512x512, 768x768, etc.)
    # The img_size is specified via command-line --decoder argument
    # Example for 512: decoder = Must3rDecoder(img_size=(512, 512), ...)
    # Example for 768: decoder = Must3rDecoder(img_size=(768, 768), ...)
    if args.gen:
        raymap_encoder = RaymapEncoderDiT(use_time_cond=args.time_cond,
                                          use_raymap_only_conditioning=args.use_raymap_only_conditioning,
                                          projection_features=args.projection_features)
        raymap_encoder.to(device)
    else:
        raymap_encoder = None
    decoder = eval(args.decoder)
    decoder.to(device)
    img_encoder.to(device)

    img_encoder_without_ddp = img_encoder
    raymap_encoder_without_ddp = raymap_encoder
    decoder_without_ddp = decoder
    if getattr(decoder_without_ddp, 'sam_model', 'SAM2') != 'SAM2':
        raise ValueError(f"training_multiview only supports SAM2 distillation; got decoder sam_model={decoder_without_ddp.sam_model}")
    args.sam_model = getattr(decoder_without_ddp, 'sam_model', args.sam_model)
    args.pointmaps_activation = decoder_without_ddp.pointmaps_activation

   
   
    if args.pretrained and last_ckpt_fname is None:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        img_encoder_status = img_encoder.load_state_dict(ckpt['encoder'], strict=False)
        print('img_encoder load_state_dict status:', img_encoder_status)
        decoder_status = decoder.load_state_dict(ckpt['decoder'], strict=False)
        print('decoder load_state_dict status (from pretrained):', decoder_status)
        
        if args.pretrained_occany:
            print('Loading pretrained_occany: ', args.pretrained_occany)
            occany_ckpt = torch.load(args.pretrained_occany, map_location=device, weights_only=False)
            # img_encoder.load_state_dict(occany_ckpt['encoder'], strict=False)
            if raymap_encoder is not None and 'raymap_encoder' in occany_ckpt:
                raymap_encoder_status = raymap_encoder.load_state_dict(occany_ckpt['raymap_encoder'], strict=False)
                print('raymap_encoder load_state_dict status:', raymap_encoder_status)
            decoder_occany_status = decoder.load_state_dict(occany_ckpt['decoder'], strict=False)
            print('decoder load_state_dict status (from pretrained_occany):', decoder_occany_status)
            del occany_ckpt
        del ckpt

    # In gen mode, create gen_decoder from decoder and freeze the original decoder.
    if args.gen:
        import copy
        gen_decoder = copy.deepcopy(decoder)
        gen_decoder.to(device)
        if args.pretrained_occany_gen:
            print('Loading pretrained_occany_gen: ', args.pretrained_occany_gen)
            occany_ckpt = torch.load(args.pretrained_occany_gen, map_location=device, weights_only=False)
            gen_decoder_occany_status = gen_decoder.load_state_dict(occany_ckpt['gen_decoder'], strict=False)
            print('gen_decoder load_state_dict status (from pretrained_occany_gen):', gen_decoder_occany_status)
            raymap_encoder_occany_status = raymap_encoder.load_state_dict(occany_ckpt['raymap_encoder'], strict=False)
            print('raymap_encoder load_state_dict status (from pretrained_occany_gen):', raymap_encoder_occany_status)
            del occany_ckpt
        # Freeze the original decoder
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        gen_decoder_without_ddp = gen_decoder
        decoder_without_ddp = decoder
    else:
        gen_decoder = None
        gen_decoder_without_ddp = None
        decoder_without_ddp = decoder  # Set for non-distributed case
        
    if args.distributed:
        img_encoder = torch.nn.parallel.DistributedDataParallel(
            img_encoder, device_ids=[args.gpu], find_unused_parameters=True, static_graph=False)
        img_encoder_without_ddp = img_encoder.module
        
        # Only wrap decoder with DDP if it remains trainable in recon mode.
        if not args.gen:
            decoder = torch.nn.parallel.DistributedDataParallel(
                decoder, device_ids=[args.gpu], find_unused_parameters=True, static_graph=False)
            decoder_without_ddp = decoder.module
        # else: decoder is frozen, keep it unwrapped (decoder_without_ddp already set at line 326)
        
        if gen_decoder is not None:
            gen_decoder = torch.nn.parallel.DistributedDataParallel(
                gen_decoder, device_ids=[args.gpu], find_unused_parameters=True, static_graph=False)
            gen_decoder_without_ddp = gen_decoder.module
        if raymap_encoder is not None:
            raymap_encoder = torch.nn.parallel.DistributedDataParallel(
                raymap_encoder, device_ids=[args.gpu], find_unused_parameters=True, static_graph=False)
            raymap_encoder_without_ddp = raymap_encoder.module
    # else: Non-distributed case uses _without_ddp variables set at lines 292-294, 326, 330

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
   
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {}
    for dataset in args.test_dataset.split('+'):
        testset = build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        name_testset = dataset.split('(')[0]

        data_loader_test[name_testset] = testset

    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    train_criterion_gen = eval(args.train_criterion_gen).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.train_criterion).to(device)
    test_criterion_gen = eval(args.test_criterion_gen or args.train_criterion_gen).to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    
    # following timm: set wd as 0 for bias and norm layers
    if hasattr(decoder_without_ddp, 'head_sam') and hasattr(decoder_without_ddp.head_sam, 'neck'):
        try:
            print("[DEBUG2] neck conv3x3 bias requires_grad:",
                  decoder_without_ddp.head_sam.neck.convs[0].conv_3x3.bias.requires_grad)
        except Exception as e:
            print("[DEBUG2] could not read neck bias requires_grad:", e)
    if not args.finetune_encoder:
        for p in img_encoder_without_ddp.parameters():
            p.requires_grad = False
    param_groups = []
    if args.finetune_encoder:
        param_groups += optim.get_parameter_groups(img_encoder_without_ddp, 0, args.weight_decay)

    trainable_decoder = gen_decoder_without_ddp if gen_decoder_without_ddp is not None else decoder_without_ddp
    param_groups += optim.get_parameter_groups(trainable_decoder, img_encoder_without_ddp.depth, args.weight_decay)
    if raymap_encoder is not None:
        param_groups += optim.get_parameter_groups(raymap_encoder_without_ddp, 0, args.weight_decay)

    sam_proj_params = {
        param for name, param in trainable_decoder.named_parameters()
        if name.startswith("head_sam.proj.")
    }
    if sam_proj_params:
        print(f"[INFO] Applying SAM proj LR multiplier: {args.sam_proj_lr_mult}")
        updated_param_groups = []
        for group in param_groups:
            group_params = group["params"]
            sam_group_params = [param for param in group_params if param in sam_proj_params]
            other_group_params = [param for param in group_params if param not in sam_proj_params]
            if other_group_params:
                updated_param_groups.append({**group, "params": other_group_params})
            if sam_group_params:
                updated_param_groups.append({
                    **group,
                    "params": sam_group_params,
                    "lr_scale": group.get("lr_scale", 1.0) * args.sam_proj_lr_mult,
                })
        param_groups = updated_param_groups

    if misc.is_main_process():
        total_enc_params = sum(p.numel() for p in img_encoder_without_ddp.parameters())
        trainable_enc_params = sum(p.numel() for p in img_encoder_without_ddp.parameters() if p.requires_grad)
        print("[DEBUG-OPT] finetune_encoder=", args.finetune_encoder,
                "img_encoder trainable/total params=", trainable_enc_params, "/", total_enc_params,
                "num_param_groups=", len(param_groups))
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            gathered_test_stats = {}
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            for test_name, testset in data_loader_test.items():

                if test_name not in test_stats:
                    continue

                if getattr(testset.dataset.dataset, 'strides', None) is not None:
                    original_test_name = test_name.split('_stride')[0]
                    if original_test_name not in gathered_test_stats.keys():
                        gathered_test_stats[original_test_name] = []
                    gathered_test_stats[original_test_name].append(test_stats[test_name])

                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            if len(gathered_test_stats) > 0:
                for original_test_name, stride_stats in gathered_test_stats.items():
                    if len(stride_stats) > 1:
                        stride_stats = {k: np.mean([x[k] for x in stride_stats]) for k in stride_stats[0]}
                        log_stats.update({original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()})
                        if args.wandb:
                            log_dict = {original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()}
                            log_dict.update({'epoch': epoch})
                            wandb.log(log_dict)

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname):
        checkpoints.save_model(args=args, img_encoder=img_encoder_without_ddp, raymap_encoder=raymap_encoder_without_ddp,
                               decoder=decoder_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
                               epoch=epoch, fname=fname, gen_decoder=gen_decoder_without_ddp)

    checkpoints.load_model(args=args, chkpt_path=last_ckpt_fname, img_encoder=img_encoder_without_ddp,
                           raymap_encoder=raymap_encoder_without_ddp, 
                           decoder=decoder_without_ddp, optimizer=optimizer,
                           loss_scaler=loss_scaler, gen_decoder=gen_decoder_without_ddp)

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None



    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Test on multiple datasets
        new_best = False
        new_pose_best = False
        already_saved = False
        if (epoch >= args.start_epoch and args.eval_freq > 0 and epoch % args.eval_freq == 0) or args.eval_only:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                if args.eval_only:
                    log_writer = None
                
                stats = test_one_epoch(img_encoder=img_encoder, raymap_encoder=raymap_encoder, 
                                       decoder=decoder, 
                                       gen_decoder=gen_decoder,
                                       test_criterion=test_criterion,
                                       test_criterion_gen=test_criterion_gen,
                                       data_loader=testset,
                                       device=device, epoch=epoch,
                                       distill_model=distill_model,
                                       distill_criterion=distill_criterion,
                                       log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats
                

            # Ensure that eval_pose_estimation is only run on the main process
            if args.pose_eval_freq>0 and (epoch % args.pose_eval_freq==0 or args.eval_only):
                ate_mean, rpe_trans_mean, rpe_rot_mean, outfile_list, bug = eval_pose_estimation(args, model, device, save_dir=f'{args.output_dir}/{epoch}')
                print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')


                if ate_mean < best_pose_ate_sofar and not bug: # if the pose estimation is better, and w/o any error
                    best_pose_ate_sofar = ate_mean
                    new_pose_best = True

            # Synchronize all processes to ensure eval_pose_estimation is completed
            try:
                torch.distributed.barrier()
            except:
                pass

        

        # Train
        train_stats = train_one_epoch(
            img_encoder=img_encoder,
            raymap_encoder=raymap_encoder,
            decoder=decoder,
            gen_decoder=gen_decoder,
            train_criterion=train_criterion,
            train_criterion_gen=train_criterion_gen,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            distill_model=distill_model,
            distill_criterion=distill_criterion,
            log_writer=log_writer,
            args=args)

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)
        if args.eval_only and args.epochs <= 1:
            exit(0)

        # Save the 'last' checkpoint
        if global_rank == 0 and epoch >= args.start_epoch:
            save_model(epoch, 'last')
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch, str(epoch))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)

    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not test)

    sampler = getattr(loader, 'sampler', None)
    sampler_name = type(sampler).__name__ if sampler is not None else 'None'
    print(f'[INFO] {split} sampler: {sampler_name}')

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(img_encoder: torch.nn.Module, 
                    raymap_encoder: torch.nn.Module, 
                    decoder: torch.nn.Module, 
                    gen_decoder: torch.nn.Module,
                    train_criterion: torch.nn.Module,
                    train_criterion_gen: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args, distill_model, distill_criterion, log_writer=None):

    assert torch.backends.cuda.matmul.allow_tf32 == True
    if raymap_encoder is not None:
        raymap_encoder.train(True)
    img_encoder.train(True)
    # If gen_decoder exists, keep decoder frozen in eval mode and train gen_decoder instead
    if gen_decoder is not None:
        gen_decoder.train(True)
        # decoder stays in eval mode (frozen)
    else:
        decoder.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)
        dtype = get_dtype(args)
        # Use loss_of_one_batch_occany_gen in generation mode.
        if args.gen:
            batch_result = loss_of_one_batch_occany_gen(views=batch, 
                                                    loss_enc_feat=args.loss_enc_feat,
                                                    raymap_encoder=raymap_encoder, 
                                                    img_encoder=img_encoder, 
                                                    decoder=decoder,
                                                    decoder_gen=gen_decoder,
                                                    criterion=train_criterion, 
                                                    criterion_gen=train_criterion_gen,
                                                    device=device,
                                                    symmetrize_batch=True,
                                                    dtype=dtype,
                                                    not_pred_raymap=not args.gen,
                                                    pointmaps_activation=args.pointmaps_activation,
                                                    distill_criterion=distill_criterion,
                                                    distill_model=distill_model,
                                                    use_raymap_only_conditioning=args.use_raymap_only_conditioning,
                                                    sam_model=args.sam_model)
        else:
            batch_result = loss_of_one_batch_occany(views=batch, 
                                                    loss_enc_feat=args.loss_enc_feat,
                                                    raymap_encoder=raymap_encoder, 
                                                    img_encoder=img_encoder, 
                                                    decoder=decoder, 
                                                    criterion=train_criterion, 
                                                    criterion_gen=train_criterion_gen,
                                                    device=device,
                                                    symmetrize_batch=True,
                                                    dtype=dtype,
                                                    not_pred_raymap=not args.gen,
                                                    pointmaps_activation=args.pointmaps_activation,
                                                    distill_criterion=distill_criterion,
                                                    distill_model=distill_model,
                                                    sam_model=args.sam_model,
                                                    finetune_encoder=args.finetune_encoder)
        

        loss, loss_details = batch_result['loss']  # criterion returns two values
        
        loss_value = float(loss)
    
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            for k, v in loss_details.items():
                print("{}: {}".format(k, v))
            sys.exit(1)

        loss /= accum_iter


        if raymap_encoder is not None:
            raymap_encoder_parameters = raymap_encoder.parameters()
        else:
            raymap_encoder_parameters = []
        if args.finetune_encoder:
            parameters_chain = chain(img_encoder.parameters(), raymap_encoder_parameters, decoder.parameters())
        else:
            parameters_chain = chain(raymap_encoder_parameters, decoder.parameters())
            
        loss_scaler(loss, optimizer, parameters=parameters_chain,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            # If loss is zero and it's time to update, just zero the gradients
            optimizer.zero_grad()
            # torch.cuda.empty_cache()

        del loss
        del batch
        del batch_result

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        # Only perform all_reduce for logging if distributed training is active
        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            if args.distributed:
                loss_value_reduce = misc.all_reduce_mean(loss_value)  # All ranks must execute this
            else:
                loss_value_reduce = loss_value
            if log_writer is not None:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_lr', lr, epoch_1000x)
                log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
                for name, val in loss_details.items():
                    log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(img_encoder: torch.nn.Module, raymap_encoder: torch.nn.Module, 
                   decoder: torch.nn.Module, 
                   gen_decoder: torch.nn.Module,
                   test_criterion: torch.nn.Module,
                   test_criterion_gen: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, distill_model, distill_criterion, log_writer=None, prefix='test'):
    img_encoder.eval()
    if raymap_encoder is not None:
        raymap_encoder.eval()
    decoder.eval()
    if gen_decoder is not None:
        gen_decoder.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch) if not args.fixed_eval_set else data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch) if not args.fixed_eval_set else data_loader.sampler.set_epoch(0)

    n_draw = 0
    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):


    
        dtype = get_dtype(args)
        # Use loss_of_one_batch_occany_gen in generation mode.
        if args.gen:
            batch_result = loss_of_one_batch_occany_gen(views=batch,
                                                    loss_enc_feat=args.loss_enc_feat,
                                                    img_encoder=img_encoder,
                                                    raymap_encoder=raymap_encoder,
                                                    decoder=decoder,
                                                    decoder_gen=gen_decoder,
                                                    criterion=test_criterion,
                                                    criterion_gen=test_criterion_gen,
                                                    device=device,
                                                    symmetrize_batch=True,
                                                    dtype=dtype,
                                                    not_pred_raymap=not args.gen,
                                                    pointmaps_activation=args.pointmaps_activation,
                                                    distill_criterion=distill_criterion,
                                                    distill_model=distill_model,
                                                    use_raymap_only_conditioning=args.use_raymap_only_conditioning,
                                                    sam_model=args.sam_model)
        else:
            batch_result = loss_of_one_batch_occany(views=batch,
                                                    loss_enc_feat=args.loss_enc_feat,
                                                    img_encoder=img_encoder,
                                                    raymap_encoder=raymap_encoder,
                                                    decoder=decoder,
                                                    criterion=test_criterion,
                                                    criterion_gen=test_criterion_gen,
                                                    device=device,
                                                    symmetrize_batch=True,
                                                    dtype=dtype,
                                                    not_pred_raymap=not args.gen,
                                                    pointmaps_activation=args.pointmaps_activation,
                                                    distill_criterion=distill_criterion,
                                                    distill_model=distill_model,
                                                    sam_model=args.sam_model,
                                                    finetune_encoder=args.finetune_encoder)

        loss_tuple = batch_result['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)
       
        
        if misc.is_main_process() and idx %  1 == 0 and n_draw < 40: # 20 recon, 20 gen
            # Save/log only one 'gen' and one 'recon'
            pred_key, gt_key = "combined_preds", "combined_gt"
            bs = batch_result[pred_key]['rgb'].shape[0]
            for batch_idx in range(bs):
                n_draw += 1
                local_view_idx = 0
                pred_img = batch_result[pred_key]['rgb'][batch_idx] # (N, H, W, 3)
                gt_img = torch.stack([batch_result[gt_key][l_idx]['img'][batch_idx] for l_idx in range(len(batch_result[gt_key]))]) # (N, 3, H, W)
                gt_pts3d = torch.stack([batch_result[gt_key][l_idx]['pts3d'][batch_idx] for l_idx in range(len(batch_result[gt_key]))]) # (N, H, W, 3)
                is_raymap = [batch_result[gt_key][l_idx]['is_raymap'][batch_idx] for l_idx in range(len(batch_result[gt_key]))]
                timestep = [batch_result[gt_key][l_idx]['timestep'][batch_idx] for l_idx in range(len(batch_result[gt_key]))]
                pred_pts3d_local = batch_result[pred_key]['pts3d_local'][batch_idx] # (N, 3)
                gt_c2w = torch.stack([batch_result[gt_key][l_idx]['camera_pose'][batch_idx] for l_idx in range(len(batch_result[gt_key]))])
                gt_w2c = torch.linalg.inv(gt_c2w)
                gt_pts3d_local = geotrf(gt_w2c, gt_pts3d)
                valid_mask = [batch_result[gt_key][l_idx]['valid_mask'][batch_idx].detach().cpu().numpy() for l_idx in range(len(batch_result[gt_key]))]

                # Sort by timestep and reorder all arrays accordingly
                sorted_indices = sorted(range(len(timestep)), key=lambda i: timestep[i])
                timestep = [timestep[i] for i in sorted_indices]
                pred_img = pred_img[sorted_indices]  # Reorder pred_img
                gt_img = gt_img[sorted_indices]      # Reorder gt_img
                is_raymap = [is_raymap[i] for i in sorted_indices]
                pred_pts3d_local = pred_pts3d_local[sorted_indices]
                gt_pts3d_local = gt_pts3d_local[sorted_indices]
                valid_mask = [valid_mask[i] for i in sorted_indices]

                gt_img = gt_img.permute(0, 2, 3, 1)
                frame_id = batch_result[gt_key][0]['label'][batch_idx]


                # Scale to [0, 1] for visualization
                pred_img = (pred_img * 0.5 + 0.5).clamp(0, 1) * 255.0
                gt_img = (gt_img * 0.5 + 0.5).clamp(0, 1) * 255.0
                pred_depth_color = torch.stack([torch.from_numpy(
                    depth2rgb(pred_pts3d_local[j, :, :, 2].detach().cpu().numpy(), min_depth=0.1, max_depth=50)) for j in range(pred_pts3d_local.shape[0])])
                gt_depth_color = torch.stack([torch.from_numpy(
                    depth2rgb(gt_pts3d_local[j, :, :, 2].detach().cpu().numpy(),
                    valid_mask=valid_mask[j], min_depth=0.1, max_depth=50)) for j in range(gt_pts3d_local.shape[0])])
                
                assert pred_img.shape == gt_img.shape
                
                # Add red border where is_raymap is True
                border_width = 2
                side_width = 5
                for i_ray in range(len(is_raymap)):
                    if is_raymap[i_ray]:
                        # Add red border to pred_img[i]
                        pred_img[i_ray, :border_width, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Top border
                        pred_img[i_ray, -border_width:, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Bottom border
                        pred_img[i_ray, :, :side_width, :] = torch.tensor([255.0, 0.0, 0.0])  # Left border
                        
                        # Add red border to gt_img[i]
                        gt_img[i_ray, :border_width, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Top border
                        gt_img[i_ray, -border_width:, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Bottom border
                        
                        pred_depth_color[i_ray, :border_width, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Top border
                        pred_depth_color[i_ray, -border_width:, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Bottom border
                        
                        gt_depth_color[i_ray, :border_width, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Top border
                        gt_depth_color[i_ray, -border_width:, :, :] = torch.tensor([255.0, 0.0, 0.0])  # Bottom border
                        gt_depth_color[i_ray, :, -side_width:, :] = torch.tensor([255.0, 0.0, 0.0])  # Right border
            
                # Concatenate all N views vertically for pred and gt, then stack pred and gt horizontally
                N = pred_img.shape[0]
                pred_col = torch.cat([pred_img[i] for i in range(N)], dim=0)  # (N*H, W, 3)
                gt_col = torch.cat([gt_img[i] for i in range(N)], dim=0)      # (N*H, W, 3)
                pred_depth_col = torch.cat([pred_depth_color[i] for i in range(N)], dim=0)  # (N*H, W, 3)
                gt_depth_col = torch.cat([gt_depth_color[i] for i in range(N)], dim=0)      # (N*H, W, 3)
                combined = torch.cat([pred_col.detach().cpu(), gt_col.detach().cpu(), 
                                      pred_depth_col.detach().cpu(), gt_depth_col.detach().cpu()], dim=1)  # (N*H, 2W, 3)

                if log_writer is not None:
                    # Log to TensorBoard
                    combined_np = combined.detach().cpu().numpy()  # HWC
                    step = 1000 * epoch  # keep consistency with scalar convention
                    log_writer.add_image(f'{prefix}_{pred_key}/{frame_id}', combined_np / 255.0, step, dataformats='HWC')

                # Fallback: save to disk as before
                save_dir = f'{args.output_dir}/{prefix}_{pred_key}'
                os.makedirs(save_dir, exist_ok=True)
                image_path = f'{save_dir}/{frame_id}_epoch{epoch}'
                combined_np = combined.detach().cpu().numpy()
                combined_pil = Image.fromarray((combined_np).astype(np.uint8))
                combined_pil.save(f'{image_path}_concat.jpg')
         
                
             


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}


    for name, val in results.items():
        if log_writer is not None:
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)
        else:
            print(f"{prefix}_{name}: {val}")
    return results

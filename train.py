# import needed library
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import net_builder, get_logger, count_parameters
from train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup
from models.hsja.HSJA import HSJA
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:  # 只是一个提醒重复的作用
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:  # resume=False
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed  # =False
    ngpus_per_node = torch.cuda.device_count()

    args.batch_size = int(args.batch_size / args.world_size)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size  # = 1*0 =0

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:  # get
        tb_log = TBLog(save_path, 'tensorboard')  # （tb_dir, file_name）tensorboard保存的路径和名字
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)  # 日志记录状态
    logger.warning(f"USE GPU: {args.gpu} for training")

    args.bn_momentum = 1.0 - args.ema_m
    _net_builder = net_builder(args.net,
                               args.net_from_name,
                               {'depth': args.depth,
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'bn_momentum': args.bn_momentum,
                                'dropRate': args.dropout})

    model = HSJA(_net_builder,
                 args.num_classes,
                 args.ema_m,
                 args.T,
                 args.p_cutoff,
                 args.ulb_loss_ratio,
                 args.hard_label,
                 num_eval_iter=args.num_eval_iter,
                 tb_log=tb_log,  # log路径
                 logger=logger)  # log记录

    logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter * 0)
    ## set SGD and cosine lr
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.train_model.cuda(args.gpu)
            model.train_model = torch.nn.parallel.DistributedDataParallel(model.train_model,
                                                                          device_ids=[args.gpu])
            model.eval_model.cuda(args.gpu)

        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.train_model = model.train_model.cuda(args.gpu)
        model.eval_model = model.eval_model.cuda(args.gpu)

    else:  # no
        model.train_model = torch.nn.DataParallel(model.train_model).cuda()
        model.eval_model = torch.nn.DataParallel(model.eval_model).cuda()

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True

    # Construct Dataset & DataLoader
    # Load labeled and unlabeled data and use ten data enhancements
    train_dset = SSL_Dataset(name=args.dataset, train=True,
                             num_classes=args.num_classes, data_dir=args.data_dir)
    lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels, num=0, user_10_argument=True)

    # Load the evaluation dataset (but now the evaluation dataset is cifar10)
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, num_classes=args.num_classes, data_dir=args.data_dir)
    eval_dset = _eval_dset.get_dset(eval_cifar=False)  # 得到张的验证集

    # Load the test set, only after each iteration of the loop training, to test the consistency, not for training
    _eval_dset_cifar10 = SSL_Dataset(name=args.dataset, train=False, num_classes=args.num_classes,
                                     data_dir=args.data_dir)
    eval_dset_cifar10 = _eval_dset_cifar10.get_dset(eval_cifar=True)  # 得到cifar10的1w张测试集

    # Pack the dataset
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'eval_cifar10': eval_dset_cifar10}

    # Load the dataloader from the dataset
    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler=args.train_sampler,
                                              num_iters=args.every_num_train_iter,
                                              num_workers=args.num_workers,
                                              distributed=args.distributed)

    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size * args.uratio,
                                               data_sampler=args.train_sampler,
                                               num_iters=args.every_num_train_iter,
                                               num_workers=4 * args.num_workers,
                                               distributed=args.distributed)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size,
                                          num_workers=args.num_workers)

    loader_dict['eval_cifar10'] = get_data_loader(dset_dict['eval_cifar10'],
                                                  args.eval_batch_size,
                                                  num_workers=args.num_workers)
    best_uniform_acc = 0.00

    # START TRAINING
    for epoch in range(args.epoch):
        print(".............", "Iteration", epoch, "...............")

        trainer = model.Aug_cutmix_train

        model.set_data_loader(loader_dict)

        temp_best_uniform_acc, temp_best_model = trainer(args, logger=logger)  # 半监督用这句话
        temp_best_uniform_acc = temp_best_uniform_acc['eval/top-1-acc'].item()
        print("Iteration", epoch, ", Current CIFAR-10 validation accuracy:", temp_best_uniform_acc * 100)
        if temp_best_uniform_acc > best_uniform_acc:
            best_uniform_acc = temp_best_uniform_acc
        if epoch < 8:
            temp_lb_dset, temp_ulb_dset = train_dset.get_ssl_dset(ask_blackbox_num=args.num_labels,
                                                                  uncertainty_choise=False,
                                                                  user_10_argument=True,
                                                                  CoRe=True, random_ask=False,)

            # Update labels and unlabels
            dset_dict.update({"'train_lb'": temp_lb_dset})
            dset_dict.update({"'ulb_dset'": temp_ulb_dset})

            temp_train_lb = get_data_loader(temp_lb_dset,
                                            args.batch_size,
                                            data_sampler=args.train_sampler,
                                            num_iters=args.every_num_train_iter,
                                            num_workers=args.num_workers,
                                            distributed=args.distributed)

            temp_train_ulb = get_data_loader(temp_ulb_dset,
                                             args.batch_size * args.uratio,
                                             data_sampler=args.train_sampler,
                                             num_iters=args.every_num_train_iter,
                                             num_workers=4 * args.num_workers,
                                             distributed=args.distributed)

            loader_dict.update({"train_lb": temp_train_lb})
            loader_dict.update({"train_ulb": temp_train_ulb})

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)
    print("Final training result: Highest model consistency accuracy:", best_uniform_acc * 100)
    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='cifar10')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', default=True, action='store_true')

    '''
    Training Configuration of HSJA
    '''

    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--num_train_iter', type=int, default=110000,
                        help='total number of training iterations')
    parser.add_argument('--every_num_train_iter', type=int, default=10,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=20,help='Query the number of black boxes per iteration')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', action='store_true', help='use mixed precision training or not')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=34)
    parser.add_argument('--widen_factor', type=int, default=1)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)  # 原=1

    '''
    multi-GPUs & Distrbitued Training
    '''
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    main(args)

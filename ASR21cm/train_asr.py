import datetime
import logging
import math
import time
import torch
from os import path as osp

from ASR21cm.data import build_dataloader
from ASR21cm.data.ASR21cm_dataset import create_collate_fn
from basicsr import init_tb_loggers, load_resume_state
from basicsr.data import build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import AvgTimer, MessageLogger, get_env_info, get_root_logger, get_time_str, make_exp_dirs, mkdir_and_rename
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def create_train_val_dataloader(opt, logger, **kwargs):
    simple_sampler = kwargs.get('simple_sampler', False)
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_set.phase = 'train'
            if not simple_sampler:
                train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            else:
                if opt['dist']:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=opt['world_size'], rank=opt['rank'])
                else:
                    train_sampler = None
            collate_fn = create_collate_fn(dataset_opt, phase=phase)
            train_loader = build_dataloader(train_set, dataset_opt, collate_fn, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            if logger is not None:
                logger.info('Training statistics:'
                            f'\n\tNumber of train images: {len(train_set)}'
                            f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                            f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                            f'\n\tWorld size (gpu number): {opt["world_size"]}'
                            f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                            f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_set.phase = phase
            collate_fn = create_collate_fn(dataset_opt, phase=phase)
            val_loader = build_dataloader(val_set, dataset_opt, collate_fn, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            if logger is not None:
                logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    simple_sampler = False
    result = create_train_val_dataloader(opt, logger, simple_sampler=simple_sampler)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
    print(f'type(train_loader) = {type(train_loader)}, len(train_loader) = {len(train_loader)}, type(train_sampler) = {type(train_sampler)}, len(val_loaders) = {len(val_loaders)}')

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    if not simple_sampler:
        # dataloader prefetcher
        prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
        if prefetch_mode is None or prefetch_mode == 'cpu':
            prefetcher = CPUPrefetcher(train_loader)
        elif prefetch_mode == 'cuda':
            prefetcher = CUDAPrefetcher(train_loader, opt)
            logger.info(f'Use {prefetch_mode} prefetch dataloader')
            if opt['datasets']['train'].get('pin_memory') is not True:
                raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        else:
            raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")
    else:
        prefetcher = None
    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    # with torch.autograd.profiler.emit_nvtx():
    for epoch in range(start_epoch, total_epochs + 1):
        # train_sampler.set_epoch(epoch)
        if isinstance(train_sampler, EnlargedSampler) or isinstance(train_sampler, torch.utils.data.distributed.DistributedSampler):
            train_sampler.set_epoch(epoch)
        if prefetcher is not None:
            prefetcher.reset()
            train_data = prefetcher.next()
        else:
            train_data_iter = iter(train_loader)
            train_data = next(train_data_iter)

        torch.cuda.memory._record_memory_history() if torch.cuda.is_available() and opt.get('memory_profiling', False) else None
        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()

            if isinstance(train_sampler, EnlargedSampler):
                train_data = prefetcher.next()
            else:
                try:
                    train_data = next(train_data_iter)
                except StopIteration:
                    train_data = None
        # end of iter
    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()
    torch.cuda.memory._dump_snapshot(filename=osp.join(opt['path']['experiments_root'], f'{current_iter}_memory_snapshot.pickle')) if torch.cuda.is_available() and opt.get('memory_profiling', False) else None


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
import numpy as np

from .DistributedProxySampler import DistributedProxySampler

    
def split_ssl_data(data, target, num_labels,include_lb_to_ulb=False):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data  如果为True，标记数据也包含在未标记数据中
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx = sample_labeled_data(data, target, num_labels)       #随机选择有标签数据的index，生成有标签数据
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx)))) #unlabeled_data index of data  除去有标签的index就是unlabelled
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:                               #label不包含在unlabel里
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]
    
    
def sample_labeled_data(data, target,  num_labels,):      # 第一次的时候随机选择数据问黑盒
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''

    lb_data = []
    lbs = []
    lb_idx = []
    idx = np.random.choice(range(0, len(data)),num_labels, False)
    lb_idx = idx
    lb_data = data[idx]
    lbs = target[idx]

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__ 
                      if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)

        
def get_data_loader(dset,
                    batch_size = None,
                    shuffle = False,
                    num_workers = 0,
                    pin_memory = True,
                    data_sampler = None,
                    replacement = True,
                    num_epochs = None,
                    num_iters = None,
                    generator = None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """
    
    assert batch_size is not None
        
    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=pin_memory)
    
    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)
        
        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1


        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset)*num_epochs

        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas

        else:
            num_samples = len(dset)

        
        if data_sampler.__name__ == 'RandomSampler':    
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")
        
        if distributed:
            '''
            Different with DistributedSampler, 
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            '''
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler, 
                          num_workers=num_workers, pin_memory=pin_memory)

    
def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
import random

import numpy as np
from scipy.stats import entropy
import torch
from sklearn.metrics import mutual_info_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import itertools
from .augmentation.randaugment import RandAugment
from .data_utils import get_sampler_by_name, get_data_loader, get_onehot, split_ssl_data
from .dataset import BasicDataset
from tqdm import tqdm
from torch.utils.data.dataset import T_co
import cv2 as cv
import torchvision
from torchvision import datasets, transforms
import json
from numpy import os
from skimage import io
from PIL import Image


def get_transform(train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor()])
    else:
        return transforms.Compose([transforms.ToTensor()])


class MyDataset(Dataset):  # 自己定义的Dataset，需要完成三个方法的实现，
    def __init__(self, data_1) -> None:  # 用来告诉我数据在哪
        self.data = data_1
        # print(self.data[0])
        # print("self.data", self.data.shape,
        #       type(self.data))  # self.data (10000, 3, 32, 32) <class 'numpy.ndarray'>
        # print("self.labels", len(self.labels), type(self.labels))  # self.labels 10000 <class 'numpy.ndarray'>
        # self.transform = transform
        self.len = len(self.data)

    def __getitem__(self, index) -> T_co:  # 创建索引，通过索引能找到数据
        data = self.data[index]
        # labels = self.labels[index]
        return data

    def __len__(self):  # 返回data长度
        return self.len


def ask_black_model(data):  # 问黑盒得到target
    device = 'cuda'
    blackbox = torch.load('E:\\22.10.2_code\\5.19Fix\\blackbox.pth')
    my_lb_data_dataset = MyDataset(data)  # 将有标签数据lb_data，放入MyDataset
    my_lb_data_loader = DataLoader(dataset=my_lb_data_dataset, batch_size=64, shuffle=False)

    my_lb_targets = []

    with torch.no_grad():
        for inputs in my_lb_data_loader:

            inputs = inputs.to(device)

            inputs = inputs / 255

            input = inputs.permute(0, 3, 1, 2)

            outputs = blackbox(
                input)  # input:1024,3,32,32          output:1024,10      都是tensor    原黑盒outputs = [64,10,1,1]  现[64,10]

            pse = torch.softmax(outputs, dim=1)

            _, max_idx = torch.max(pse, dim=1)
            # print("问过黑盒的标签：",max_idx)
            t = max_idx.cpu()

            temp_targets = t.detach().numpy().astype(np.uint8)

            for i in temp_targets:
                # my_lb_targets.append(i)  # 同模型结构
                my_lb_targets.append(int(i[0][0]))  # 不同

            # print(my_lb_targets)
    return my_lb_targets


def load_data(path, mode):
    xs = []
    ys = []

    if mode == 'train':
        print("开始加载imagenet训练数据集.....")
        data_files = [os.path.join(path, 'train_data_batch_%d.json' % idx) for idx in range(1, 2)]
    else:
        # assert mode == 'val', 'Mode not supported.'
        print("开始加载imagenet测试数据集.....")
        data_files = [os.path.join(path, 'val_data.json')]

    for data_file in data_files:
        print('Loading', data_file)

        with open(data_file, 'rb') as data_file_handle:
            d = json.load(data_file_handle)

        x = np.array(d['data'], dtype=np.float32)
        y = np.array(d['labels'])

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i - 1 for i in y]
        img_size = 32
        img_size2 = img_size * img_size
        x = np.dstack((x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:], x[:, :img_size2]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))
        xs.append(x)
        ys.append(np.array(y))

    if len(xs) == 1:
        data = xs[0]
        labels = ys[0]
    else:
        data = np.concatenate(xs, axis=0)
        labels = np.concatenate(ys, axis=0)

    return data, labels


def load_label(path, mode):  # 加载imagenet数据集
    xs = []
    ys = []

    if mode == 'train':
        print("开始加载imagenet训练数据集.....")
        data_files = [os.path.join(path, 'train_data_batch_%d.json' % idx) for idx in range(1, 2)]
    else:
        # assert mode == 'val', 'Mode not supported.'
        print("开始加载imagenet测试数据集.....")
        data_files = [os.path.join(path, 'val_data.json')]

    for data_file in data_files:
        print('Loading', data_file)

        with open(data_file, 'rb') as data_file_handle:
            d = json.load(data_file_handle)

        # x = np.array(d['data'], dtype=np.float32)
        y = np.array(d['labels'])

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i - 1 for i in y]
        img_size = 32
        img_size2 = img_size * img_size
        # x = np.dstack((x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:], x[:, :img_size2]))
        # x = x.reshape((x.shape[0], img_size, img_size, 3))
        # xs.append(x)
        ys.append(np.array(y))

    if len(xs) == 1:
        data = xs[0]
        labels = ys[0]
    else:
        # data = np.concatenate(xs, axis=0)
        labels = np.concatenate(ys, axis=0)
    # print(type(data), data.shape)
    print(type(labels), labels.shape)
    return labels


class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir='./data'):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """

        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform(train)
        self.all_lb_data = None
        self.cut_ulb_data = None
        self.all_lb_targets = None
        self.ulb_targets = None

    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        dset = dset(self.data_dir, train=self.train, download=True)
        data, targets = dset.data, dset.targets
        return data, targets

    def get_dset(self, use_strong_transform=False,
                 strong_transform=None, onehot=False, eval_cifar=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """

        data, targets = self.get_data()

        if eval_cifar:  # 原cifar10的测试集  评估一致性
            targets = ask_black_model(data)

        else:
            data_path = 'E:\\22.10.2_code\\5.19Fix\\data\\imagenet'

            data, targets = load_data(data_path, mode='val')
            data = np.array(data, dtype=np.uint8)
            data = data[:6000]
            targets = ask_black_model(data)
            print("Validation set:", type(targets), data.shape, len(targets))
            with open('data.json', 'w') as f:
                json.dump({'data': data.tolist(), 'targets': targets}, f)

        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        return BasicDataset(data, targets, num_classes, transform,
                            use_strong_transform, strong_transform, onehot)

    def get_ssl_dset(self, ask_blackbox_num=100, use_strong_transform=True, strong_transform=None, onehot=False,
                     num=None, uncertainty_choise=False, user_10_argument=False, CoRe=False, random_ask=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        temp_all_lb_data = self.all_lb_data  # Used for globally storing labeled data, gradually increasing
        temp_cut_ulb_data = self.cut_ulb_data  # Used to store unlabeled data that has been queried in the black box
        temp_all_lb_targets = self.all_lb_targets  # Used to store the labels of all labeled data queried in the black box
        temp_all_ulb_targets = self.ulb_targets  # Used to store the labels of all labeled data queried in the black box

        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        device = 'cuda'
        if num == 0:  # The initial data is the selected top data
            # Read the selected dataset
            data_path = 'E:/22.10.2_code/5.19Fix/data/imagenet'

            labels = load_label(data_path, mode='train')  # Load original labels (not used)

            lb_idx = []  # Labels of labeled data

            # Load the filtered high-quality dataset using JSON format Here we are loading the dataset for the
            # initial substitute model, and we need to modify this part proportionally based on different query
            # budgets such as 10k, 15k, 20k, 25k, 30k, etc.
            origin_data_files = ["E:/22.10.2data/top_all_img_6_2-3266.json"]
            for data_file in origin_data_files:
                print('Loading', data_file)
                with open(data_file, 'r') as data_file_handle:
                    d = json.load(data_file_handle)
                    origin_data = np.array(d).astype(np.uint8)
            print("Initial dataset:", type(origin_data), origin_data.shape)

            # Load the unlabeled high-quality dataset, which is the dataset we obtained after filtering with high confidence
            all_data_files = ["E:/22.10.2data/top_all_img_6_2-3266.json"]
            for data_file in all_data_files:
                print('Loading', data_file)
                with open(data_file, 'r') as data_file_handle:
                    d = json.load(data_file_handle)
                    data = np.array(d).astype(np.uint8)
            print("Dataset used for training:", type(data), data.shape)
            lb_targets = ask_black_model(origin_data)
            lb_targets = np.array(lb_targets)
            lb_data = origin_data
            ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))
            ulb_targets = labels[ulb_idx]
            ulb_data = data[ulb_idx]

            print("lb_data:", len(lb_data), type(lb_data), lb_data.shape)
            print("lb_targets:", len(lb_targets), type(lb_targets), lb_targets.shape)
            print("ulb_data:", len(ulb_data), type(ulb_data), ulb_data.shape)
            print("ulb_targets:", len(ulb_targets), type(ulb_targets), ulb_targets.shape)

            temp_all_ulb_targets = ulb_targets
            temp_cut_ulb_data = ulb_data
            temp_all_lb_data = lb_data
            temp_all_lb_targets = lb_targets
        # Active learning selection strategy - entropy based uncertainty strategy
        if uncertainty_choise:
            # In the active learning section, ask for a well-trained alternative model for each iteration
            thiefmodel = torch.load('./saved_models/cifar10-40/model_last_iter_only_model.pth')
            # Query the alternative model with all unlabeled data, and judge the data at the decision boundary
            # according to the output of the alternative model
            my_ulb_data_dataset = MyDataset(temp_cut_ulb_data)
            my_ulb_data_loader = DataLoader(dataset=my_ulb_data_dataset, batch_size=64, shuffle=False)

            all_entropies = []

            with torch.no_grad():
                for inputs in tqdm(my_ulb_data_loader):
                    inputs = inputs.to(device)

                    inputs = inputs / 255

                    input = inputs.permute(0, 3, 1, 2)
                    # input:1024,3,32,32      output:1024,10     tensor
                    outputs = thiefmodel(input)

                    pse = torch.softmax(outputs, dim=1)

                    t = pse.cpu()

                    pse = t.detach().numpy()

                    entropies = np.array([entropy(yv) for yv in pse])

                    all_entropies.extend(entropies)

                all_entropies = np.array(all_entropies)
                # After sorting, I only need the first ask_blackbox_num (default is 100)
                sort_fist_second = np.argsort(all_entropies * -1)[:ask_blackbox_num]

            need_ask_blackbox_ulb_data = temp_cut_ulb_data[sort_fist_second]
            # Combine all labeled data that has been asked about the black box with labeled data that has just been
            # asked about the black box this time
            temp_all_lb_data = np.append(temp_all_lb_data, need_ask_blackbox_ulb_data,
                                         axis=0)

            a = ask_black_model(need_ask_blackbox_ulb_data)  # 将筛选出来的数据问黑盒，得到标签

            a = np.array(a)

            b = np.array(temp_all_ulb_targets[sort_fist_second])
            print(" After asking the black box, the same number as the original real label: ", np.sum(a == b))

            # Combine labels of all labeled data
            temp_all_lb_targets = np.append(temp_all_lb_targets, a, axis=0)
            # Delete the queried data based on the indices
            temp_cut_ulb_data = np.delete(temp_cut_ulb_data, sort_fist_second, axis=0)
            temp_all_ulb_targets = np.delete(temp_all_ulb_targets, sort_fist_second, axis=0)

        # Random selection strategy
        if random_ask:
            random_idx = random.sample(range(1, len(temp_cut_ulb_data)), ask_blackbox_num)
            need_ask_random_data = temp_cut_ulb_data[random_idx]
            temp_all_lb_data = np.append(temp_all_lb_data, need_ask_random_data, axis=0)
            a = ask_black_model(need_ask_random_data)
            a = np.array(a)
            temp_all_lb_targets = np.append(temp_all_lb_targets, a, axis=0)  # 将所有label数据的标签结合
            temp_cut_ulb_data = np.delete(temp_cut_ulb_data, random_idx, axis=0)  # 从unlabel里删除刚选出来的label索引
            temp_all_ulb_targets = np.delete(temp_all_ulb_targets, random_idx, axis=0)  # 删除无标签数据对应的索引的标签
        # CoRe selection strategy
        if CoRe is True:
            thiefmodel = torch.load('./saved_models/cifar10-40/model_last_iter_only_model.pth')
            my_ulb_data_dataset = MyDataset(temp_cut_ulb_data)
            my_ulb_data_loader = DataLoader(dataset=my_ulb_data_dataset, batch_size=64, shuffle=False)
            icc_results = []
            all_result_abs = []
            all_max_difference = []
            al_max_then_min = []
            with torch.no_grad():
                for inputs in tqdm(my_ulb_data_loader):
                    # print("inputs====",type(inputs),inputs.shape)
                    argument_inputs = inputs
                    argument_inputs = argument_inputs.cpu()
                    base_inputs = argument_inputs.detach().numpy()
                    argument_inputs = base_inputs[:, :, ::-1]  # 数据增强的一种方式，使用的是镜像翻转
                    temp_argument_inputs = []
                    for image in argument_inputs:
                        M = np.float32([[1, 0, 0], [0, 1, 15]])
                        temp_image = cv.warpAffine(image, M, (32, 32))
                        temp_argument_inputs.append(temp_image)
                    # temp_image
                    temp_argument_inputs = np.array(temp_argument_inputs)
                    argument_inputs = torch.from_numpy(np.ascontiguousarray(temp_argument_inputs))

                    # The following code is an alternative model for the original image
                    inputs = inputs.to(device)
                    inputs = inputs / 255
                    input = inputs.permute(0, 3, 1, 2)
                    outputs = thiefmodel(input)
                    pse = torch.softmax(outputs, dim=1)
                    t = pse.cpu()
                    pse = t.detach().numpy()
                    # The following code asks for an alternative model for the enhanced image
                    argument_inputs = argument_inputs.to(device)
                    argument_inputs = argument_inputs / 255
                    argument_input = argument_inputs.permute(0, 3, 1, 2)
                    argument_outputs = thiefmodel(argument_input)
                    argument_pse = torch.softmax(argument_outputs, dim=1)
                    argument_t = argument_pse.cpu()
                    argument_pse = argument_t.detach().numpy()

                    for i in range(len(argument_pse)):
                        max_temp_pse = np.sort(pse[i])[-1]
                        max_temp_argument_pse = np.sort(argument_pse[i])[-1]
                        if max_temp_pse >= max_temp_argument_pse:
                            max_then_min = max_temp_argument_pse / max_temp_pse
                        else:
                            max_then_min = max_temp_pse / max_temp_argument_pse
                        al_max_then_min.append(max_then_min)

            al_max_then_min = np.array(al_max_then_min)

            icc_results_idx = np.argsort(al_max_then_min)

            icc_results_idx = icc_results_idx[:ask_blackbox_num]

            mi_ask_black_box_data = temp_cut_ulb_data[icc_results_idx]
            temp_all_lb_data = np.append(temp_all_lb_data, mi_ask_black_box_data,axis=0)

            a = ask_black_model(mi_ask_black_box_data)

            a = np.array(a)

            temp_all_lb_targets = np.append(temp_all_lb_targets, a, axis=0)

            temp_cut_ulb_data = np.delete(temp_cut_ulb_data, icc_results_idx, axis=0)

            temp_all_ulb_targets = np.delete(temp_all_ulb_targets, icc_results_idx, axis=0)

        self.all_lb_data = temp_all_lb_data
        self.cut_ulb_data = temp_cut_ulb_data
        self.all_lb_targets = temp_all_lb_targets
        self.ulb_targets = temp_all_ulb_targets

        print(" Existing label data: ", type(self.all_lb_data), self.all_lb_data.shape)
        print(" No labeled data: ", type(self.cut_ulb_data), self.cut_ulb_data.shape)

        lb_dset = BasicDataset(self.all_lb_data, self.all_lb_targets, num_classes,
                               transform, False, None, onehot, user_10_argument)

        ulb_dset = BasicDataset(self.cut_ulb_data, self.ulb_targets, num_classes,
                                transform, use_strong_transform, strong_transform, onehot, user_10_argument=False)

        return lb_dset, ulb_dset

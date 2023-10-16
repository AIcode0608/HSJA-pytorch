from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

from PIL import Image
import numpy as np
import copy


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    """
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 user_10_argument=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        
        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot
        self.user_10_argument = user_10_argument
        self.transform = transform

        # if use_strong_transform:
        #     if strong_transform is None:             #强增强转换
        #         self.strong_transform = copy.deepcopy(transform)
        #         self.strong_transform.transforms.insert(0, RandAugment(3,5))
        # else:
        #     self.strong_transform = strong_transform

        if self.user_10_argument:
            print("The labeled data was enhanced 10 times....")
            self.strong_transform = copy.deepcopy(transform)
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform2 = copy.deepcopy(transform)
            self.strong_transform2.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform3 = copy.deepcopy(transform)
            self.strong_transform3.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform4 = copy.deepcopy(transform)
            self.strong_transform4.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform5 = copy.deepcopy(transform)
            self.strong_transform5.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform6 = copy.deepcopy(transform)
            self.strong_transform6.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform7 = copy.deepcopy(transform)
            self.strong_transform7.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform8 = copy.deepcopy(transform)
            self.strong_transform8.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform9 = copy.deepcopy(transform)
            self.strong_transform9.transforms.insert(0, RandAugment(3, 5))

            self.strong_transform10 = copy.deepcopy(transform)
            self.strong_transform10.transforms.insert(0, RandAugment(3, 5))
        else:
            if use_strong_transform:
                if strong_transform is None:
                    self.strong_transform = copy.deepcopy(transform)
                    self.strong_transform.transforms.insert(0, RandAugment(3,5))
            else:
                self.strong_transform = strong_transform

    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        
        #set idx-th target
        if self.targets is None:
            target = None
        else:   #数据生成onehot
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
            
        #set augmented images
            
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)

            if self.user_10_argument:
                return img_w, self.strong_transform(img), self.strong_transform2(img), self.strong_transform3(
                    img), self.strong_transform4(img), self.strong_transform5(img), self.strong_transform6(
                    img), self.strong_transform7(img), self.strong_transform8(img), self.strong_transform9(
                    img), self.strong_transform10(img), target  # 原图、增强、标签
            else:
                if self.use_strong_transform:
                    return img_w, self.strong_transform(img), target
                else:
                    return img_w, target

    
    def __len__(self):
        return len(self.data)
from jittor.dataset import CIFAR10, CIFAR100, MNIST
from .ImageNet import ImageNet
from jittor import transform
from .argument import three_augment, create_transform
from .constant import *
import numpy as np
from PIL import Image

def create_dataset(
        name,
        root,
        split='validation',
        transform=None,
        target_transform=None,
        download=False,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        **kwargs
):

    if split == 'validation':
        train = False
    else:
        train = True
    if name == 'CIFAR10':
        ds = CIFAR10(root=root, train=train, transform=transform, 
                     target_transform=target_transform,download=download, **kwargs).set_attrs(batch_size=batch_size, 
                                                                shuffle=shuffle, transform=transform, 
                                                                num_workers=num_workers)
        return ds
    elif name == 'CIFAR100':
        ds = CIFAR100(root=root, train=train, transform=transform, 
                     target_transform=target_transform,download=download, **kwargs).set_attrs(batch_size=batch_size, 
                                                                shuffle=shuffle, transform=transform, 
                                                                num_workers=num_workers)
        return ds
    elif name == 'MNIST':
        ds = MNIST(root=root, train=train, transform=transform, 
                     target_transform=target_transform,download=download, **kwargs).set_attrs(batch_size=batch_size, 
                                                                shuffle=shuffle, transform=transform, 
                                                                num_workers=num_workers)
        return ds
    elif name == 'ImageNet':
        ds = ImageNet(root=root, train=train, **kwargs).set_attrs(batch_size=batch_size, 
                                                                shuffle=shuffle, transform=transform, 
                                                                num_workers=num_workers)
        return ds
    else:
        print('Please implement the custom dataset')
        return None

def build_dataset(is_train, args, class_map=None):

    student_transform = build_transform_deit(is_train, args)
    teacher_transform = build_transform_deit(is_train, args)
    
    if args.data_set == "CIFAR100":
        dataset = CIFAR100(
            args.data_path, train=is_train, transform=student_transform, download=True
        )
        nb_classes = 100

    elif args.data_set == "CIFAR10":
        dataset = CIFAR10(
            args.data_path, train=is_train, transform=student_transform, download=True
        )
        nb_classes = 10

    elif args.data_set == "CIFAR10LT":
        if is_train:
            dataset = KD_IMBALANCECIFAR10(
                root=args.data_path,
                imb_type=args.imb_type,
                imb_factor=args.imb_factor,
                rand_number=0,
                train=True,
                download=True,
                student_transform=student_transform,
                teacher_transform=teacher_transform,
            )
        else:
            dataset = CIFAR10(
                args.data_path,
                train=is_train,
                transform=student_transform,
                download=True,
            )
        nb_classes = 10

    elif args.data_set == "CIFAR100LT":
        if is_train:
            dataset = KD_IMBALANCECIFAR100(
                root=args.data_path,
                imb_type=args.imb_type,
                imb_factor=args.imb_factor,
                rand_number=0,
                train=True,
                download=True,
                student_transform=student_transform,
                teacher_transform=teacher_transform,
            )
        else:
            dataset = CIFAR100(
                args.data_path,
                train=is_train,
                transform=student_transform,
                download=True,
            )
        nb_classes = 100

    return dataset, nb_classes


def build_transform_deit(is_train, args):
    size = args.input_size
    resize_im = size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        # Three-augment from DeiT-3
        if args.ThreeAugment:
            trans = three_augment(args)

        trans = create_transform(
            input_size=size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            trans.transforms[0] = transform.RandomCrop(size, padding=4)
        return trans

    print("Deit val")
    t = []
    if resize_im:
        new_size = int((256 / 224) * size)
        t.append(
            transform.Resize(
                new_size, mode=Image.BILINEAR
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transform.CenterCrop(size))

    t.append(transform.ToTensor())
    t.append(transform.ImageNormalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    # t.append(transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]))

    return transform.Compose(t)

# Code modified for DeiT-LT
class KD_IMBALANCECIFAR10(CIFAR10):
    cls_num = 10

    def __init__(
        self,
        root,
        imb_type="exp",
        imb_factor=0.01,
        rand_number=0,
        train=True,
        student_transform=None,
        teacher_transform=None,
        target_transform=None,
        download=False,
    ):
        super(KD_IMBALANCECIFAR10, self).__init__(
            root, train, target_transform=target_transform, download=download
        )
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * the_img_num
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.student_transform is not None:
            student_img = self.student_transform(img)
            # teacher_img = self.teacher_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return student_img, target

# Code modified for DeiT-LT
class KD_IMBALANCECIFAR100(KD_IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    cls_num = 100
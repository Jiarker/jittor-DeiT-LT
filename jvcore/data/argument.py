import jittor
from jittor import transform
import random
from PIL import ImageFilter, ImageOps, Image
from jvcore.data.constant import *
import math
from typing import Optional, Tuple, Union

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transform.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

def three_augment(args=None):
    """
    * 3-Augment from DeiT-III
    * (https://arxiv.org/pdf/2204.07118.pdf)
    """

    img_size = args.input_size
    remove_random_resized_crop = args.src
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    secondary_tfl = [
        transform.Resize(img_size, interpolation=Image.BILINEAR),
        transform.RandomCrop(img_size, padding=4, padding_mode="reflect"),
        transform.RandomHorizontalFlip(),
        transform.RandomChoice(
            [gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]
        ),
    ]

    if args.color_jitter is not None and not args.color_jitter == 0:
        secondary_tfl.append(
            transform.ColorJitter(
                args.color_jitter, args.color_jitter, args.color_jitter
            )
        )
    final_tfl = [
        transform.ToTensor(),
        transform.ImageNormalize(mean=jittor.array(mean), std=jittor.array(std)),
    ]
    return transform.Compose(secondary_tfl + final_tfl)

def transforms_imagenet_train(
    img_size: int = 224,
    hflip: float = 0.5,
    vflip: float = 0.,
    color_jitter: float = 0.4,
    grayscale_prob: float = 0.,
    gaussian_blur_prob: float = 0.,
    interpolation: int = Image.BILINEAR,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    re_prob: float = 0.,
    re_mode: str = 'const',
    re_count: int = 1,
    use_prefetcher: bool = False,
    normalize: bool = True,
):
    """
    ImageNet-oriented image transforms for training in Jittor.
    """
    transforms = []
    
    # Resize and CenterCrop
    if interpolation == 'bilinear':
        interpolation = Image.BILINEAR  # Jittor's interpolation mode
    transforms.append(transform.Resize(img_size, mode=interpolation))
    transforms.append(transform.CenterCrop(img_size))
    
    # RandomHorizontalFlip
    if hflip > 0:
        transforms.append(transform.RandomHorizontalFlip(p=hflip))
    
    # RandomVerticalFlip
    if vflip > 0:
        transforms.append(transform.RandomVerticalFlip(p=vflip))
    
    # ColorJitter is not directly available in Jittor, need to implement manually
    if color_jitter > 0:
        if color_jitter > random.random():
            transforms.append(transform.ColorJitter())
    
    # RandomGrayscale
    if grayscale_prob > 0:
        transforms.append(transform.RandomGray(p=grayscale_prob))
    
    # GaussianBlur is not directly available in Jittor, need to implement manually
    if gaussian_blur_prob > 0:
        transforms.append(GaussianBlur())
    
    # ToTensor and Normalize
    # if use_prefetcher:
    #     transforms.append(ToNumpy())  # Assuming ToNumpy is defined to convert to numpy array
    if not normalize:
        transforms.append(transform.pil_to_tensor())  # Assuming MaybePILToTensor is defined
    else:
        transforms.append(transform.ToTensor())
        transforms.append(transform.ImageNormalize(mean=mean, std=std))
    
    # if re_prob > 0:
    #     # Implement RandomErasing manually or find an alternative
    #     pass

    return transform.Compose(transforms)

def transforms_imagenet_eval(
        img_size: int = 224,
        crop_pct: float = 0.875,
        interpolation: int = Image.BILINEAR,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        use_prefetcher: bool = False,
        normalize: bool = True,
):
    """ ImageNet-oriented image transform for evaluation and inference in Jittor. """
    tfl = []

    if crop_pct < 1.0:
        # Resize to crop_pct of target size
        scale_size = int(img_size / crop_pct)
        if interpolation == 'bilinear':
            interpolation = Image.BILINEAR  # Jittor's interpolation mode
        tfl.append(transform.Resize(scale_size, mode=interpolation))
        tfl.append(transform.CenterCrop(img_size))

    if not normalize:
        # Convert to tensor without scaling, keeps original dtype
        tfl.append(transform.ToTensor())
    else:
        tfl.append(
            transform.ImageNormalize(
                mean=mean,
                std=std,
            ),
        )

    return transform.Compose(tfl)


def create_transform(
        input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224,
        is_training: bool = False,
        no_aug: bool = False,
        train_crop_mode: Optional[str] = None,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        color_jitter_prob: Optional[float] = None,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        interpolation: int = Image.BILINEAR,
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        crop_border_pixels: Optional[int] = None,
        tf_preprocessing: bool = False,
        use_prefetcher: bool = False,
        normalize: bool = True,
        separate: bool = False,
):
    """

    Args:
        input_size: Target input size (channels, height, width) tuple or size scalar.
        is_training: Return training (random) transforms.
        no_aug: Disable augmentation for training (useful for debug).
        train_crop_mode: Training random crop mode ('rrc', 'rkrc', 'rkrr').
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string (see auto_augment.py).
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        crop_pct: Inference crop percentage (output size / resize size).
        crop_mode: Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        crop_border_pixels: Inference crop border of specified # pixels around edge of original image.
        tf_preprocessing: Use TF 1.0 inference preprocessing for testing model ports
        use_prefetcher: Pre-fetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalization tensor output w/ provided mean/std (if prefetcher not used).
        separate: Output transforms in 3-stage tuple.

    Returns:
        Composed transforms or tuple thereof
    """
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_training:
        transform = transforms_imagenet_train(
            img_size,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            grayscale_prob=grayscale_prob,
            gaussian_blur_prob=gaussian_blur_prob,
            interpolation=interpolation,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            use_prefetcher=use_prefetcher,
            normalize=normalize,
            )
    else:
        transform = transforms_imagenet_eval(
            img_size,
            crop_pct=crop_pct,            
            interpolation=interpolation,
            mean=mean,
            std=std,
            use_prefetcher=use_prefetcher,
            normalize=normalize,
        )
    return transform
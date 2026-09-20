"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""
# 出处:landmark2face_wy/data/image_folder.pyc(py3.8,反汇编见
# evidence/modules/_heygem_orphans/disasm/modules_heygem_landmark2face_wy_data_image_folder.pyc.dis)。
# 恢复说明:模块 docstring 逐字取自模块 [Constants][0];import 形态按模块 [Disassembly]
# (import torch.utils.data as data 的 fromlist=None + ROT_TWO 形态、from PIL import Image、
# import os、import os.path);make_dataset 的 `images[:min(max_dataset_size, len(images))]` 与
# ImageFolder.__init__ 的 RuntimeError 文案按字节码还原;旧候选里多出的 self.paths 赋值不在字节码中,已删除。
# 口径说明:.dis 里 make_dataset/ImageFolder.__init__ 出现的全局名 'AssertionError' 是 py3.8 把
# `assert cond, msg` 编译成 LOAD_GLOBAL AssertionError + CALL_FUNCTION + RAISE_VARARGS 的产物,
# 源码按字节码保留 assert 语句,不写成 raise AssertionError。
# 未还原细节:无。
import torch.utils.data as data
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

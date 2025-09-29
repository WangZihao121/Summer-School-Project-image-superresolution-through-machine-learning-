import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import calc_patch_size, convert_rgb_to_y


# 使用装饰器 calc_patch_size 来自动计算 patch 大小（具体逻辑在 utils.py 中）
@calc_patch_size
def train(args):
    # 创建一个 HDF5 文件，用于存储训练数据（低分辨率和高分辨率 patch）
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []  # 存放低分辨率图像块
    hr_patches = []  # 存放高分辨率图像块

    # 遍历输入目录下的所有图片
    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')  # 打开并转为 RGB
        hr_images = []

        # 如果开启数据增强（with_aug），则对图像进行缩放和旋转增强
        if args.with_aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:  # 不同比例缩放
                for r in [0, 90, 180, 270]:      # 不同角度旋转
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)  # 不增强时只保留原图

        # 遍历增强后的所有 HR 图像
        for hr in hr_images:
            # 调整 HR 图像大小，使其能被 scale 整除
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)

            # 生成对应的 LR 图像（缩小 scale 倍）
            lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)

            # 转换为 numpy 数组并转为 float32
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)

            # 转换为 Y 通道（亮度），因为超分辨率通常只在 Y 通道上训练
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            # 从 LR 图像中裁剪 patch，同时在 HR 图像中裁剪对应的高分辨率 patch
            for i in range(0, lr.shape[0] - args.patch_size + 1, args.scale):   #for循环中的三个参数分别对应i的起始值，结束值和每次i变化的步长
                for j in range(0, lr.shape[1] - args.patch_size + 1, args.scale):
                    # LR patch
                    lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])  #lr_patches 最终是一个三维 numpy 数组，里面存放了所有从 LR 图像裁剪出来的 patch，每个 patch 的大小是 (patch_size, patch_size)，数据类型是 float32。
                    # 对应的 HR patch（注意要乘以 scale）
                    hr_patches.append(
                        hr[i*args.scale:i*args.scale+args.patch_size*args.scale,
                           j*args.scale:j*args.scale+args.patch_size*args.scale]
                    )

    # 转换为 numpy 数组
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    # 保存到 HDF5 文件
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    # 创建 HDF5 文件，用于存储验证/测试数据
    h5_file = h5py.File(args.output_path, 'w')

    # 在 HDF5 文件中创建两个 group，分别存放 LR 和 HR 图像
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    # 遍历输入目录下的所有图片
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')

        # 调整 HR 图像大小，使其能被 scale 整除
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)

        # 生成对应的 LR 图像
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)

        # 转换为 numpy 数组并转为 float32
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        # 转换为 Y 通道
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        # 将每张图像单独存储在 group 下
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)   # 输入图像目录
    parser.add_argument('--output-path', type=str, required=True)  # 输出 HDF5 文件路径
    parser.add_argument('--scale', type=int, default=2)            # 缩放因子
    parser.add_argument('--with-aug', action='store_true')         # 是否启用数据增强
    parser.add_argument('--eval', action='store_true')             # 是否生成验证集
    args = parser.parse_args()

    # 根据参数选择执行训练数据准备或验证数据准备
    if not args.eval:
        train(args)
    else:
        eval(args)

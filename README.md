## FSRCNN的使用背景

SRCNN计算成本过高，而移动端处理器（ARM架构）对复杂的卷积运算非常敏感
$\color{red}{\textbf{了解为什么arm对计算复杂度很敏感（问蒿老师这个需不需要做）}}$

## 参考文献
$\color{red}{\textbf{去读}}$
This repository is implementation of the ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367).

<center><img src="./thumbnails/fig1.png"></center>

其中提到了FSRCNN相对于SRCNN的改进之处：
$\color{red}{\textbf{问蒿老师我需不需要去识别这些东西在代码中的体现，去识别一下代码中的体现}}$
1.introduce a deconvolution layer at the end of the network
2.reformulate the mapping layer by shrinking the input feature dimension before mapping and expanding back afterwards
3.adopt smaller filter sizes but more mapping layers

取得了更好的效果：
1.speed up of more than 40 times with even superior restoration quality
$\color{red}{\textbf{需不需要去将SRCNNrun一下然后对比效率}}$
2.present the parameter settings that can achieve real-time performance on a generic CPU while still maintaining good performance
$\color{red}{\textbf{这话什么意思？}}$
3.transfer strategy is also proposed for fast training and testing across different upscaling factors
$\color{red}{\textbf{这话什么意思？}}$

## $\color{red}{\textbf{作者列出的两个关键点}}$

- Added the zero-padding这个在论文中提到了。在model.py文件中。零填充的目的是保持图像尺寸和保护边缘信息
- Used the Adam instead of the SGD这个在论文中没提到

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/01z95js39kgw1qv/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/qx4swlt2j7u4twr/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/vobvi2nlymtvezb/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/4kzqmtqzzo29l1x/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/kyhbhyc5a0qcgnp/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/ihtv1acd48cof14/Set5_x4.h5?dl=0) |

Otherwise, you can use `prepare.py` to create custom dataset.

```bash
python train.py --train-file "BLAH_BLAH/91-image_x3.h5" \
                --eval-file "BLAH_BLAH/Set5_x3.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 20 \
                --num-workers 8 \
                --seed 123                
```

## Test

Pre-trained weights can be downloaded from the links below.

| Model | Scale | Link |
|-------|-------|------|
| FSRCNN(56,12,4) | 2 | [Download](https://www.dropbox.com/s/1k3dker6g7hz76s/fsrcnn_x2.pth?dl=0) |
| FSRCNN(56,12,4) | 3 | [Download](https://www.dropbox.com/s/pm1ed2nyboulz5z/fsrcnn_x3.pth?dl=0) |
| FSRCNN(56,12,4) | 4 | [Download](https://www.dropbox.com/s/vsvumpopupdpmmu/fsrcnn_x4.pth?dl=0) |

The results are stored in the same path as the query image.

```bash
python test.py --weights-file "BLAH_BLAH/fsrcnn_x3.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
```

## Results

PSNR was calculated on the Y channel.

### Set5

| Eval. Mat | Scale | Paper | Ours (91-image) |
|-----------|-------|-------|-----------------|
| PSNR | 2 | 36.94 | 37.12 |
| PSNR | 3 | 33.06 | 33.22 |
| PSNR | 4 | 30.55 | 30.50 |

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>FSRCNN x3 (34.66 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/lenna.bmp""></center>
    	</td>
    	<td>
    		<center><img src="./data/lenna_bicubic_x3.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/lenna_fsrcnn_x3.bmp"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>FSRCNN x3 (28.55 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/butterfly_GT.bmp""></center>
    	</td>
    	<td>
    		<center><img src="./data/butterfly_GT_bicubic_x3.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/butterfly_GT_fsrcnn_x3.bmp"></center>
    	</td>
    </tr>
</table>

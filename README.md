## 为什么SRCNN不行

SRCNN计算成本过高，而移动端处理器（ARM架构）对复杂的卷积运算非常敏感。
为什么arm架构的芯片对卷积运算很敏感：
arm架构采用RISC协议，RISC取指压力大： 对于同样的卷积计算，输入给RISC芯片的二进制指令串往往比CISC长得多，芯片需要从内存里不停地“吃”指令。因为指令串很长，芯片的“嘴巴”（取指单元）就会很忙，为了完成卷积，RISC 要发出比 CISC 多得多的指令电信号。每发一次信号，芯片内部的电容就要充放电一次，这就是发热和耗电的物理根源。因为 RISC 每次只能算一小步，中间结果必须频繁地存入和取出寄存器。这在物理电路上意味着电子在导线里不停地跑来跑去，路径变长，这就是延迟的物理根源

## 解决方案：

- FSRCNN
- ESPCNN

## FSRCNN参考文献
$\color{red}{\textbf{去读}}$
This repository is implementation of the ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367).

<center><img src="./thumbnails/fig1.png"></center>

其中提到了FSRCNN相对于SRCNN的改进之处：

1.introduce a deconvolution layer at the end of the network  -->  model.py:self.last_part = nn.ConvTranspose2d

2.reformulate the mapping layer by shrinking the input feature dimension (model.py:nn.Conv2d(d, s, kernel_size=1)) before mapping and expanding back afterwards(model.py:nn.Conv2d(s, d, kernel_size=1))

3.adopt smaller filter sizes but more mapping layers

在model.py中
![](./screenshot/Screenshot1.png)

-
-
-
-
取得了更好的效果：

1.speed up of more than 40 times with even superior restoration quality
$\color{red}{\textbf{需不需要去将SRCNNrun一下然后对比效率}}$

2.present the parameter settings that can achieve real-time performance on a generic CPU while still maintaining good performance
$\color{red}{\textbf{这话什么意思？}}$

3.transfer strategy is also proposed for fast training and testing across different upscaling factors
$\color{red}{\textbf{这话什么意思？}}$

## $\color{red}{\textbf{作者列出的两个关键点}}$

- Added the zero-padding这个在论文中提到了。在model.py文件中。零填充的目的是保持图像尺寸和保护边缘信息

- Used the Adam instead of the SGD（train.py: optimizer = optim.Adam）


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

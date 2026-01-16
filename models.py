import math
from torch import nn        #nn是pytorch中的神经网络模块，用于构建和定义深度学习模型

# 定义 FSRCNN 模型类，继承自 nn.Module
class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):      #__init__()只在创建模型时执行一次
        super(FSRCNN, self).__init__()       # 初始化父类 nn.Module


        
        #第一层，特征提取层
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),               
            nn.PReLU(d)
        )        #这行代码创建了一个顺序容器 nn.Sequential，它是nn.Module的一个子类，你可以把它理解成一个“流水线”，如果给它一个数据，那数据会依次流过每一层。
                 #self.first_part是nn.Sequential这个类的实例化，成了一个有两个层的神经网络。
                 #此处不能理解为nn.Sequential这个类继承了nn.Conv2d和nn.PReLU这两个类
                 #nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2)   =   nn.Conv2d(num_channels, d, 5, 5//2) 这是个卷积层
                 #nn.Conv2d 是线性变换，输出 = 卷积(输入 × 权重) + 偏置
                 #nn.PReLU 是非线性变换，输出 = 输入的非线性映射

        
        #第二层
        # (1) 收缩层 (Shrinking)：降通道数
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]  # 放了两个类在列表中

        # (2) 非线性映射层 (Non-linear Mapping)：提取特征
        for _ in range(m):    #    _ 是一个常用的 Python 约定，表示不关心循环变量的值，只需要执行指定次数的循环
            self.mid_part.extend([    
                nn.Conv2d(s, s, kernel_size=3, padding=3//2), 
                nn.PReLU(s)
            ])
            # list1.extend(list2)  # 将list2中的所有元素逐个添加到list1末尾

        # (3) 扩张层 (Expanding)：恢复通道数，为重建做准备
        self.mid_part.extend([
            nn.Conv2d(s, d, kernel_size=1),  # 通道恢复
            nn.PReLU(d)
        ])
        self.mid_part = nn.Sequential(*self.mid_part)  # 将列表转换为顺序模块，*self.mid_part会将列表解包，【a，b】=>a，b


        #第三层
        #卷积是将数据变少也就提取了特征，而反卷积是将数据变多图像重建
        #反卷积上采样层（重建层）：将特征图放大为高分辨率图像
        self.last_part = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9,
            stride=scale_factor,
            padding=9//2,
            output_padding=scale_factor-1
        )

        self._initialize_weights()  # 初始化权重


    
    # 权重初始化函数，这个函数中初始化了self.first_part，self.mid_part和self.last_part中的卷积/反卷积层数据，而没有初始化它们包含的nn.PReLU非卷积层的数据，nn.PReLU非线性映射层的数据本身有默认值所以不需要初始化
    def _initialize_weights(self):

        #第一层初始化
        for m in self.first_part:
            #虽然self.first_part不是一个列表类型数据，但是nn.Sequential 实现了 __getitem__ 方法，使其支持索引。这个循环其实是在遍历self.first_part神经网络中的每个层
            if isinstance(m, nn.Conv2d):    #isinstance函数：判断m是不是nn.Conv2d类型或者其子类
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2/(m.out_channels * m.weight.data[0][0].numel())))    #通过正态分布（均值是mean，方差是std（使用了He初始化方法来确定最佳std大小））随机生成数据填入卷积核
                # 当m是卷积层的时候，m.weight是卷积核的可训练版本，m.weight.data是卷积核本身的数值
                nn.init.zeros_(m.bias.data) # m.bias.data是卷积层的偏置参数的数值部分，nn.init.zeros_将其设为0。在卷积运算中：输出 = 卷积(输入 × 权重) + 偏置

        #第二层初始化
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2/(m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        #第三层初始化
        #std小的原因：最后一层直接输出图像的像素值。使用极小的随机值初始化，可以让模型在训练初期输出接近于 0 或均值的图像，这有助于稳定训练的开始阶段，防止一开始输出乱七八糟的噪声导致损失函数（Loss）直接爆表。
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001) 
        nn.init.zeros_(self.last_part.bias.data)


    
    # 前向传播函数：定义数据流动路径
    def forward(self, x):
        x = self.first_part(x)  #特征提取层，等效于下面这段注释的代码
        #def __init__···
        #    ···
        #    self.part1 = nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2)
        #    self.part2 = nn.PReLU(d)
        #    ···
        #···
        #x = self.part2(self.part1(x))
        #self.part1(x)意味着x会作为输入输入nn.Conv2d这个类中定义的一个函数，输出就是self.part1(x)
        
        x = self.mid_part(x)    #特征收缩和非线性映射
        x = self.last_part(x)   #反卷积上采样
        return x  # 返回高分辨率图像

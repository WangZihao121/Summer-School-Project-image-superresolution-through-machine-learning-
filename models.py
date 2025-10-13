import math
from torch import nn        #nn是pytorch中的神经网络模块，用于构建和定义深度学习模型

# 定义 FSRCNN 模型类，继承自 nn.Module
class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):      #__init__()只在创建模型时执行一次
        super(FSRCNN, self).__init__()       # 初始化父类 nn.Module



        
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),               
            nn.PReLU(d)
        )        #这行代码创建了一个顺序容器 nn.Sequential，它是nn.Module的一个子类，你可以把它理解成一个“流水线”，如果给它一个数据，那数据会依次流过每一层。
                 #self.first_part是nn.Sequential这个类的实例化，成了一个有两个层的神经网络。
                 #此处不能理解为nn.Sequential这个类继承了nn.Conv2d和nn.PReLU这两个类
                 #nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2)   =   nn.Conv2d(num_channels, d, 5, 5//2) 这是个卷积层
        

        
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]  # 通道压缩。放了两个类在列表中
        for _ in range(m):
            self.mid_part.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=3//2),  # 非线性映射层
                nn.PReLU(s)
            ])
        self.mid_part.extend([
            nn.Conv2d(s, d, kernel_size=1),  # 通道恢复
            nn.PReLU(d)
        ])
        self.mid_part = nn.Sequential(*self.mid_part)  # 将列表转换为顺序模块



        
        # 反卷积上采样层：将特征图放大为高分辨率图像
        self.last_part = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9,
            stride=scale_factor,
            padding=9//2,
            output_padding=scale_factor-1
        )

        self._initialize_weights()  # 初始化权重

    # 权重初始化函数：使用 He 初始化方法对卷积层权重进行初始化
    def _initialize_weights(self):
        for m in self.first_part:
            #虽然self.first_part不是一个列表类型数据，但是nn.Sequential 实现了 __getitem__ 方法，使其支持索引。这个循环其实是在遍历self.first_part神经网络中的每个层
            if isinstance(m, nn.Conv2d):    #isinstance函数：判断m是不是nn.Conv2d类型或者其子类
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2/(m.out_channels * m.weight.data[0][0].numel())))    #通过正态分布随机抽样生成卷积核数据
                # 当m是卷积层的时候，m.weight是卷积核的可训练版本，m.weight.data是卷积核本身的数值
                nn.init.zeros_(m.bias.data) # m.bias.data是卷积层的偏置参数的数值部分，nn.init.zeros_将其设为0。在卷积运算中：输出 = 卷积(输入 × 权重) + 偏置
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2/(m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)  # 输出层初始化
        nn.init.zeros_(self.last_part.bias.data)

    # 前向传播函数：定义数据流动路径
    def forward(self, x):
        x = self.first_part(x)  #特征提取层，x会先进入nn.Conv2d（PyTorch 提供的一个类，可以理解为g（x）函数）然后g（x）进入nn.PReLU后输出h（x）并将这个值赋给x。等效于下面这段注释的代码
        #def __init__···
        #    ···
        #    self.part1 = nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2)
        #    self.part2 = nn.PReLU(d)
        #    ···
        #···
        #x = self.part2(self.part1(x))
        #self.part1是nn.Conv2d的实例化，self.part1(x)意味着x会作为输入输入nn.Conv2d这个类中定义的一个函数，得到的输出就是self.part1(x)
        
        x = self.mid_part(x)    #特征收缩和非线性映射
        x = self.last_part(x)   #反卷积上采样
        return x  # 返回高分辨率图像

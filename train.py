import argparse #用于从命令行接收参数
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm  #tqdm 是一个用于显示循环进度的 Python 库。
#此时就引入了models.py中的FSRCNN函数/FSRCNN类
from models import FSRCNN

from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr


if __name__ == '__main__':   #如果这个文件是直接被运行的，则执行下面的代码；如果这个文件是被作为模块导入（import）到其他文件中的，则不执行下面的代码

    
    #################命令行参数解析 (Argument Parsing)####################

    
    parser = argparse.ArgumentParser(description='FSRCNN超分辨率模型训练脚本')
    
    # 必需参数
    parser.add_argument('--train-file', type=str, required=True, 
                       help='训练数据文件路径（HDF5格式）')
    parser.add_argument('--eval-file', type=str, required=True,
                       help='验证数据文件路径（HDF5格式）')
    parser.add_argument('--outputs-dir', type=str, required=True,
                       help='模型输出目录')
    
    # 可选参数
    parser.add_argument('--weights-file', type=str,
                       help='预训练权重文件路径（可选）')
    parser.add_argument('--scale', type=int, default=2,
                       help='超分辨率缩放因子（默认：2）')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率（默认：0.001）')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='训练批次大小（默认：16）')
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='训练总轮数（默认：20）')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='数据加载工作进程数（默认：8）')
    parser.add_argument('--seed', type=int, default=123,
                       help='随机种子（默认：123）')
    
    args = parser.parse_args()
    #此后在代码里，你可以通过 args.lr 获取学习率，通过 args.scale 获取放大倍数。


    
    #################### 设备与输出目录设置 (Environment Setup)##############################

    
    
    # 根据缩放因子创建子目录，例如: outputs-dir/x2/
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))    #os.path.join（字符串1，字符串2，字符串3···）作用是将字符串们用‘/’直接拼接在一起

    #如果args.outputs_dir这个路径不存在，那么就创建这个路径文件夹
    if not os.path.exists(args.outputs_dir):   
        os.makedirs(args.outputs_dir)

    # ==================== 设备配置 ====================
    # 启用CUDA基准优化（提升GPU性能）
    cudnn.benchmark = True
    
    # 自动检测并使用GPU（如果可用），否则使用CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置随机种子以保证结果可重现
    torch.manual_seed(args.seed)


    
    ######################模型、损失函数与优化器 (Model, Loss & Optimizer)设置###################################


    
    # 创建FSRCNN模型并部署到相应设备（GPU/CPU）（将FSRCNN模型的参数传输到GPU上，计算转移到GPU上进行）
    model = FSRCNN(scale_factor=args.scale).to(device)
    
    # 定义损失函数（均方误差损失，适用于图像重建任务）
    criterion = nn.MSELoss()
    
    # 定义优化器（Adam优化器）（目的：在之后训练模型期间对不同卷积层智能地动态调整学习率）
    #optimizer是optim.Adam的实例化
    #传入optim.Adam一个字典列表和lr=args.lr，optim.Adam会依次检查各个字典中是否定义了lr这个键，没有就添加lr键，它的值就用args.lr。于是生成的新的字典列表可以通过optimizer.param_groups来调用
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},  
        {'params': model.mid_part.parameters()},      
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1} 
    ], lr=args.lr)


    
    #######################数据加载器 (Data Loaders)设置#################################


    
    train_dataset = TrainDataset(args.train_file)  # 加载训练数据

    #DataLoader是pytorch中定义的类
    #train_dataloader包含了一个打乱了顺序的数字列表。比如 [50, 2, 999, 14...]。它不存图，只存“货号”。
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,           # 训练时打乱数据顺序
        num_workers=args.num_workers,  # 多进程数据加载
        pin_memory=True         # 加速GPU数据传输
    )
    
    # 创建验证数据集和数据加载器（批次大小为1，逐张图像验证）
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # ==================== 训练状态初始化 ====================
    best_weights = copy.deepcopy(model.state_dict())  # 保存最佳模型权重
    best_epoch = 0     # 最佳epoch编号
    best_psnr = 0.0    # 最佳PSNR值


    
    ##################################开始训练循环##################################


    
    print("开始训练...")
    for epoch in range(args.num_epochs):

        
        # ---------- 训练阶段 ----------

        
        model.train()  # 设置模型为训练模式（启用dropout等）
        epoch_losses = AverageMeter()  # 初始化损失记录器

        # 使用进度条显示训练进度
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), 
                 ncols=80) as t:    #tqdm（total总迭代次数，ncols进度条宽度）用于为循环添加进度条
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))    #t.set_description（）在进度条左侧显示信息

            # 遍历训练数据集的每个批次
            for data in train_dataloader: #这行代码的详细解释在《学习如何写代码》文档中，data此时是一个包含了两个张量的列表，两个张量分别储存了args.batch_size个LR和HR的图像数据
                inputs, labels = data       #inputs: 低分辨率图像, labels: 高分辨率真实图像

                # 将数据移动到相应设备
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播：模型预测
                preds = model(inputs)  #具体逻辑见《学习如何写代码》

                # 计算损失：预测值与真实值的差异
                loss = criterion(preds, labels) #loss是一个torch类型的张量

                # 更新平均损失
                epoch_losses.update(loss.item(), len(inputs))    #loss.item() 的作用是从包含单个元素的 PyTorch Tensor 中提取出 Python 数值。

                # 反向传播和优化
                optimizer.zero_grad()  # 清空之前的梯度
                loss.backward()        # 反向传播计算梯度
                optimizer.step()       # 更新模型参数

                # 更新进度条显示
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))    #在进度条的右侧显示额外的信息
                t.update(len(inputs))  # 更新进度

        # 保存当前epoch的模型权重
        model_path = os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_path)
        print(f'已保存 epoch {epoch} 的模型权重: {model_path}')


        
        # ---------- 验证阶段 ----------

        
        model.eval()  # 设置模型为评估模式（禁用dropout等）
        epoch_psnr = AverageMeter()  # 初始化PSNR记录器

        # 遍历验证数据集
        for data in eval_dataloader:    #data在循环中是一个包含两个元素的元组，经过一系列操作后是EvalDataset这个类中getitem()函数的返回值
            inputs, labels = data       #data=（验证集中的第一张numpy数组lr图片，第一张numpy数组hr图片），分别赋给了inputs, labels

            # 将数据移动到相应设备
            inputs = inputs.to(device)    #inputs 的数据类型保持不变，只是处理数据的设备变了
            labels = labels.to(device)

            # 验证阶段不计算梯度（节省内存和计算资源）
            with torch.no_grad():
                # torch.no_grad使得在这个代码块中，所有操作都不会计算梯度，也不会在计算图中记录操作
                
                # 模型预测并限制像素值在[0, 1]范围内,clamp 将小于 0 的值统一变为 0，大于 1 的值统一变为 1
                preds = model(inputs).clamp(0.0, 1.0)

            # 计算PSNR（峰值信噪比，图像质量评估指标）并更新
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        # 输出当前epoch的验证结果
        print('验证 PSNR: {:.2f} dB'.format(epoch_psnr.avg))    #“:.2f”是让epoch_psnr.avg取两位小数



        
        # ---------- 保存最佳模型 ----------
        # 如果当前epoch的PSNR优于历史最佳，则更新最佳模型
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())  # 深拷贝保存权重，修改深拷贝中的任何内容都不会影响原对象。
            print(f'新的最佳模型！epoch: {epoch}, PSNR: {best_psnr:.2f} dB')


    
    #############################训练完成#################################


    
    # 输出最终结果并保存最佳模型权重
    print('训练完成！')
    print('最佳 epoch: {}, 最佳 PSNR: {:.2f} dB'.format(best_epoch, best_psnr))
    
    # 保存最佳模型权重到文件
    best_model_path = os.path.join(args.outputs_dir, 'best.pth')
    torch.save(best_weights, best_model_path)
    print(f'最佳模型已保存至: {best_model_path}')

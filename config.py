
import torch

# 超参数设置
num_epochs = 40
num_classes = 10
batch_size = 20 #32
learning_rate = 0.001

# 设备设置
# torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行，pytorch-cpu版本不需要运行这句
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
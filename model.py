
import torch.nn as nn

class ConvColorNet(nn.Module):
    def __init__(self, nc):
        super(ConvColorNet, self).__init__()
        self.nef= 64 #  encoder filters in first conv layer
        self.ngf=64
        self.nc= nc # chanel
        self.nBottleneck =4000 # dim for bottleneck of encoder
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(self.nc, self.nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 16 x 16
            nn.Conv2d(self.nef, self.nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 8 x 8
            nn.Conv2d(self.nef, self.nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 4 x 4
            nn.Conv2d(self.nef * 2, self.nBottleneck, 4, bias=False),
            # tate size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(self.nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(self.nBottleneck, self.ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 2, self.ngf,  4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf ),
            nn.ReLU(True),
            # state size. (ngf) x 8 x 8
            nn.ConvTranspose2d(self.ngf, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf ),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output

if __name__ == '__main__':
    model=ConvColorNet()
    print(model)
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import NetD, NetG

# 生成器和判别器网络权重初始化 custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否使用GPU
print( torch.cuda.is_available())

dataroot = "data2/"  # 数据集目录  Data set directory
batch_size = 5  # 批样本数 Batch size
image_size = 96  # 训练图像的大小 size of training images
nc = 3  # 训练图像通道数 Number of channels
nz = 100  # 生成器的输入向量大小 Size of z vector (generator input)
ngf = 64  # 生成器生成图片的大小 Size of feature maps in generator
ndf = 64  # 判别器判别图片的大小 Size of feature maps in discriminator
num_epochs = 100000  # 训练的epochs数 Number of training epochs
lr = 0.0002  # 学习率 Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

# 图像读入与预处理 Image reading and preprocessing
transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = dset.ImageFolder(dataroot, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

netG = NetG(ngf, nz).to(device)  # 生成器网络 Generator
netG.apply(weights_init)  # 生成器网络w初始化 Generator weights initialization
netD = NetD(ndf).to(device)  # 判别器网络 Discriminator
netD.apply(weights_init)  # 生成器网络w初始化 Discriminator weights initialization
# 打印网络模型 Print the model
print(netG)
print(netD)

criterion = nn.BCELoss()  # 初始化损失函数 Initialize BCELoss function

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(100, nz, 1, 1, device=device)

# 评估真假的标签 真为1 假为0 Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Adam优化 Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 记录训练过程 Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

print("开始训练...")
# 对每一个epoch For each epoch
for epoch in range(num_epochs):
    # 对于每一个batch For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # 使用真的图片训练鉴别器D 尽可能判别为1 Train with real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)  # imgs
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # 判别器输出标签 Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # 计算损失函数 Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # 反向传播 Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # 使用假的图片训练鉴别器D 尽可能判别为0 Train with all-fake batch
        # 产生batch虚假噪声 Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 用生成器生成假图 Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # 使用判别器判别真假 detach固定生成器G，Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # 计算损失函数 Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # 反向传播 Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 总误差 Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # 更新判别器 Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        label.fill_(real_label)  # 让G生成的假图尽可能被判别器判别为1 fake labels are real for generator cost
        # 输出假图判别的标签 Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # 计算损失函数 Calculate G's loss based on this output
        errG = criterion(output, label)
        # 反向传播 Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # 更新生成器 Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 记录损失画图 Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 观察生成器 Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            # 保存图片
            vutils.save_image(fake, '%s/fake_samples_epoch_%03d.png' % ('inm/', epoch), normalize=True)
        iters += 1

    # 保存模型 do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('models/', epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('models/', epoch))

# 画损失图 Plot G_losses、D_losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
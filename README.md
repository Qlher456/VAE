# 自动编码器(Auto Encoder ,AE)
 
自动编码器主要由两部分组成：编码器（Encoder）和解码器（Decoder）。编码器和解码器可以看作是两个函数，一个用于将高维输入（如图片）映射为低维编码（code），

另一个用于将低维编码（code）映射为高维输出（如生成的图片）。这两个函数可以是任意形式，但在深度学习中，我们用神经网络去学习这两个函数。

再使用Decoder部分，随机生成一个code然后输入，就可以得到一张生成的图像。但实际上这样的生成效果并不好，因此AE多用于数据压缩，而数据生成则使用VAE更好。

# AE的缺陷

AE的Encoder是将图片映射成“数值编码”，Decoder是将“数值编码”映射成图片。这样存在的问题是，在训练过程中，随着不断降低输入图片与输出图片之间的误差，模型会过拟合，泛化性能不好。

也就是说对于一个训练好的AE，输入某个图片，就只会将其编码为某个确定的code，输入某个确定的code就只会输出某个确定的图片，如果这个latent code来自于没见过的图片，那么生成的图片也不会好

eg. 假设我们训练好的AE将“新月”图片encode成code=1（这里假设code只有1维），将其decode能得到“新月”的图片；将“满月”encode成code=10，同样将其decode能得到“满月”图片。

这时候如果我们给AE一个code=5，我们希望是能得到“半月”的图片，但由于之前训练时并没有将“半月”的图片编码，或者将一张非月亮的图片编码为5，那么我们就不太可能得到“半月”的图片。

因此AE多用于数据的压缩和恢复，用于数据生成时效果并不理想。
 
# AE总结

不将图片映射成“数值编码”，而将其映射成“分布”。还是刚刚的例子，我们将“新月”图片映射成μ=1的正态分布，那么就相当于在1附近加了噪声，

此时不仅1表示“新月”，1附近的数值也表示“新月”，只是1的时候最像“新月”。将"满月"映射成μ=10的正态分布，10的附近也都表示“满月”。

那么code=5时，就同时拥有了“新月”和“满月”的特点，那么这时候decode出来的大概率就是“半月”了。这就是VAE的思想。

# 变分自动编码器(Variational Auto-Encoder ,VAE)

VAE是一个深度生成模型，其最终目的是生成出概率分布P(x)，x即输入数据。
 
在VAE中，我们通过高斯混合模型(Gaussian Mixture Model)来生成P(x)，也就是说P(x)是由一系列高斯分布叠加而成的，每一个高斯分布都有它自己的参数μ和σ。
 
我们借助一个变量z∼N(0,I)(注意z是一个向量，生成自一个高斯分布)，找一个映射关系，将向量z映射成这一系列高斯分布的参数向量μ(z)和σ(z)。

有了这一系列高斯分布的参数我们就可以得到叠加后的P(x)的形式，即x∣z∼N(μ(z),σ(z))。(这里的“形式”仅是对某一个向量z所得到的)，

映射关系P ( x ∣ z ) P(x|z)P(x∣z)如图所示。

 ![image](https://github.com/user-attachments/assets/a7f1529e-51d9-4a4e-bf80-bf70b2fc5db9)

输入向量z，得到参数向量μ(z)和σ(z)。这个映射关系是要在训练过程中更新N权重得到的。这部分作用相当于最终的解码器(decoder)。

对于某一个向量z我们知道了如何找到P(x)。那么对连续变量z依据全概率公式有：

P(X)=∫_z^ ▒〖P(z)P(x│z)dz〗

但是很难直接计算积分部分，因为我们很难穷举出所有的向量z用于计算积分。又因为P(x)难以计算，那么真实的后验概率

P ( z ∣ x ) = P ( z ) P ( x ∣ z ) / P ( x )同样是不容易计算的，

这也就是为什么下文要引入q(z∣x)来近似真实后验概率P(z∣x)。

因此我们用极大似然估计来估计P(x)，有似然函数L：

L=∑_X▒〖logP(x)〗

这里我们额外引入一个分布q(z∣x)， z∣x ∼N(μ′(x),σ′(x))。这个分布表示形式如下：
 
这个分布同样是用一个神经网络来完成，向量z zz根据NN输出的参数向量μ′(x)和σ′(x)运算得到，注意这三个向量具有相同的维度，这部分作用相当于最终的编码器(encoder)。

█(log⁡P(x)&=∫_z▒  q(z│x)  log⁡P(x) dz       ∵∫_z▒  q(z|x)dz=1@&=∫_z▒  q(z|x)log⁡P(z,x)/P(z|x)  dz@&=∫_z▒  q(z|x)log⁡(P(z,x)/(q(z|x))⋅(q(z|x))/P(z|x) )dz@&=∫_z▒  q(z|x)log⁡(q(z|x))/P(z|x)  dz+∫_z▒  q(z|x)log⁡P(z,x)/(q(z|x)) dz@&=D_KL (q(z|x)||P(z|x))+∫_z▒  q(z|x)log⁡(P(z,x))/(q(z|x)) dz@&≥∫_z▒  q(z│x)  log⁡〖P(z,x)/(q(z│x) ) dz〗       ∵D_KL (q||P)>0)

我们将∫_z▒  q(z│x)  log⁡〖P(z,x)/(q(z│x) ) dz〗称为log⁡P(x)的(variational) lower bound (变分下界)，简称为Lb，最大化就等价于Lb最大化似然函数L，接下来：

█(L_b&=∫_z▒  q(z|x)log⁡P(z,x)/q(z|x)  dz@&=∫_z▒  q(z|x)log⁡((P(z))/(q(z|x))⋅P(x|z))dz@&=∫_z▒  q(z|x)log⁡(P(z))/(q(z|x)) dz+∫_z▒  q(z|x)log⁡P(x|z)dz@&=-D_KL (q(z|x)||P(z))+∫_z▒  q(z|x)log⁡P(x|z)dz@&=-D_KL (q(z|x)||P(z))+E_(q(z|x)) [log⁡〖P(x|z)〗 ] )

最大化Lb包括以下部分：

1) minimizingDKL(q(z∣x)∣∣P(z))，使得后验分布近似值q(z∣x)接近先验分布P(z)。也就是说通过q(z∣x)生成的编码z不能太离谱，要与某个分布相当才行，这里是对中间编码生成起了限制作用。

当q(z∣x)和P(z)都是高斯分布时，推导式中有Appendix B：

D_KL (q(z│x)‖P(z))=-1/2 ∑_j^J▒〖(1+log⁡〖(σ_j )^2 〗-(μ_j )^2-(σ_j )^2)〗

其中J表示向量z的总维度数，σj和μj表示q(z∣x)输出的参数向量σ和μ的第j个元素。(这里的σ和μ等于前文中μ′( x ) 和σ′( x )。

2) maximizingEq(z∣x)[logP(x∣z)]，即在给定编码器输出q(z∣x)下解码器输出P(x∣z)越大越好，这部分也就相当于最小化Reconstruction Error(重建损失)。
   
由此我们可以得出VAE的原理图：

![image](https://github.com/user-attachments/assets/37fb1ac5-0ef5-41ef-bd7a-f2f6db908b45)

5.3 总结

VAE在产生新数据的时候是基于已有数据来做的，或者说是对已有数据进行某种组合而得到新数据的，它并不能生成或创造出新数据。另一方面是VAE产生的图像比较模糊。

而大名鼎鼎的GAN利用对抗学习的方式，既能生成新数据，也能产生较清晰的图像。后续的更是出现了很多种变形。

# AE.py

使用AE模型生成人脸面部图像

在1/4块A100上，使用python 3.8环境pytorch框架

使用早停法在迭代到442次时停止

Epoch [1/1000] Loss: 0.0018

![Epoch-1_Step-0](https://github.com/user-attachments/assets/66764adc-3b38-457b-badc-3839d1173f6a)


Epoch [50/1000] Loss: 0.0003

![Epoch-50_Step-0](https://github.com/user-attachments/assets/ea3a10cb-69ee-4497-82e5-048cdee25cab)

Epoch [100/1000] Loss: 0.0003

![Epoch-100_Step-0](https://github.com/user-attachments/assets/e4d2b030-a1d8-48b2-9d93-71d2620a1682)

Epoch [150/1000] Loss: 0.0003

![Epoch-150_Step-0](https://github.com/user-attachments/assets/c5e9ad88-c8ee-479b-960e-5c7595aab9a3)

Epoch [200/1000] Loss: 0.0002

![Epoch-200_Step-0](https://github.com/user-attachments/assets/eb08ac28-bbf5-458f-ab80-af99ee62d011)

Epoch [250/1000] Loss: 0.0002

![Epoch-250_Step-0](https://github.com/user-attachments/assets/0c4c84ac-74d6-417a-980a-0bbb74f9fa54)

Epoch [300/1000] Loss: 0.0002

![Epoch-300_Step-0](https://github.com/user-attachments/assets/596a4877-d21e-4ef1-bf37-7655f9887152)

Epoch [350/1000] Loss: 0.0002

![Epoch-400_Step-0](https://github.com/user-attachments/assets/da5fdaa6-97b0-4e3a-b715-f41aa76f7039)

Epoch [400/1000] Loss: 0.0002

![Epoch-400_Step-0](https://github.com/user-attachments/assets/d50dae27-173e-4bad-9904-b43d6bd5619f)

Epoch [442/1000] Loss: 0.0002

![Epoch-442_Step-0](https://github.com/user-attachments/assets/9972d666-f180-4869-b773-641bd41f7274)

Early stopping


# VAE.py

使用VAE模型生成人脸面部图像

在算力云上使用4090D使用早停法迭代1000次训练  

训练环境：pytorch 1.11.0 python 3.8 cuda 11.3 ubantu

![image](https://github.com/user-attachments/assets/2538ed2d-f72f-49a7-9c11-6fb29adc822d)

在迭代到520次时早停法停止。

![image](https://github.com/user-attachments/assets/b3d28f6e-f4e9-47b9-b0c1-ee1f09bfcf5c)

训练过程损失函数如下

![image](https://github.com/user-attachments/assets/5b78dd0e-199d-4c50-8113-cd2ac0e9b96b)

利用训练得到的模型，生成人脸面部图像

![image](https://github.com/user-attachments/assets/cacd47a0-d6df-4c5e-a77d-ae6ab8ab534e)



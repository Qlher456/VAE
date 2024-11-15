# VAE

这是使用VAE模型生成人脸面部图像的项目

在算力云上使用4090D使用早停法迭代1000次训练  

训练环境：pytorch 1.11.0 python 3.8 cuda 11.3 ubantu

![image](https://github.com/user-attachments/assets/2538ed2d-f72f-49a7-9c11-6fb29adc822d)

在迭代到520次时早停法停止。

![image](https://github.com/user-attachments/assets/b3d28f6e-f4e9-47b9-b0c1-ee1f09bfcf5c)

训练过程损失函数如下

![image](https://github.com/user-attachments/assets/5b78dd0e-199d-4c50-8113-cd2ac0e9b96b)

利用训练得到的模型，生成人脸面部图像

![image](https://github.com/user-attachments/assets/cacd47a0-d6df-4c5e-a77d-ae6ab8ab534e)



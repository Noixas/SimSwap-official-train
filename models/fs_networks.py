# """
# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# """

# from email.mime import image
# import torch
# import torch.nn as nn

# import wandb
# from transformers import AutoFeatureExtractor, SwinModel, SwinConfig

# class InstanceNorm(nn.Module):
#     def __init__(self, epsilon=1e-8):
#         """
#             @notice: avoid in-place ops.
#             https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
#         """
#         super(InstanceNorm, self).__init__()
#         self.epsilon = epsilon

#     def forward(self, x):
#         x   = x - torch.mean(x, (2, 3), True)
#         tmp = torch.mul(x, x) # or x ** 2
#         tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
#         return x * tmp

# class ApplyStyle(nn.Module):
#     """
#         @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
#     """
#     def __init__(self, latent_size, channels):
#         super(ApplyStyle, self).__init__()
#         self.linear = nn.Linear(latent_size, channels * 2)

#     def forward(self, x, latent):
#         style = self.linear(latent)  # style => [batch_size, n_channels*2]
#         shape = [-1, 2, x.size(1), 1, 1]
#         style = style.view(shape)    # [batch_size, 2, n_channels, ...]
#         #x = x * (style[:, 0] + 1.) + style[:, 1]
#         x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
#         return x

# class ResnetBlock_Adain(nn.Module):
#     def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
#         super(ResnetBlock_Adain, self).__init__()

#         p = 0
#         conv1 = []
#         if padding_type == 'reflect':
#             conv1 += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv1 += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
#         self.conv1 = nn.Sequential(*conv1)
#         self.style1 = ApplyStyle(latent_size, dim)
#         self.act1 = activation

#         p = 0
#         conv2 = []
#         if padding_type == 'reflect':
#             conv2 += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv2 += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
#         self.conv2 = nn.Sequential(*conv2)
#         self.style2 = ApplyStyle(latent_size, dim)


#     def forward(self, x, dlatents_in_slice):
#         y = self.conv1(x)
#         y = self.style1(y, dlatents_in_slice)
#         y = self.act1(y)
#         y = self.conv2(y)
#         y = self.style2(y, dlatents_in_slice)
#         out = x + y
#         return out



# class Generator_Adain_Upsample(nn.Module):
#     def __init__(self, input_nc, output_nc, latent_size, n_blocks=6, deep=False,
#                  norm_layer=nn.BatchNorm2d,
#                  padding_type='reflect',
#                  transf=False):
#         assert (n_blocks >= 0)
#         super(Generator_Adain_Upsample, self).__init__()
#         activation = nn.ReLU(True)
#         self.deep = deep
#         self.transf = transf
#         if self.transf==True:
#             self.feature_extractor_swin = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
#             configuration = SwinConfig()
#             # self.swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
#             self.swin_model = SwinModel(configuration) # Similar to: "microsoft/swin-tiny-patch4-window7-224") #output after forward[B,49,768]
#             # We process the output and get [B,768,7,7]
#             # wandb.watch(self.swin_model)
#             # for param in self.swin_model.parameters():
#             #     param.requires_grad = False
#             # # self.replication_l = nn.ReplicationPad2d((0, 0, 367, 368))
#             # self.swin_model.train(False) 
#             # self.conv_gen = nn.Conv2d(1, 512, kernel_size=1, padding=0)
#             #in_channels=num_features, out_channels=config.encoder_stride**2 * 3,
#             # 32^2 * 3 = 1024 * 3 = 3072
#             # PixelShuffle  [https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html]
#             # useful for implementing efficient sub-pixel convolution with a stride of 1/r
#             # input (*, C_in, H_in, W_in)
#             # output (*, C_out, H_out, W_out)
#             # where:
#             # C_out = C_in / upscale_factor^2 => 3072/(32^2) = 3[Here 32^2 is 1024] 
#             # H_out = H_in * upscale_factor = 7*32 = 224
#             # W_out = W_in * upscale_factor = 7*32 = 224
#             # self.mask_based_conv = nn.Sequential(nn.Conv2d(768, 32**2*3, kernel_size=1), nn.PixelShuffle(32))#, #works
#             #Goal is to match the 512x28x28 of simswap
#             # 7*4 = 28 so upscale factor is 4.
#             # 4^2 * 3 = 48
#             # 512 = 768 / x^2 => x^2 = 768/512 => x^2 = 1.5 => x = 1.224
#             # C_out = C_in / upscale_factor^2 => 3072/(32^2) = 3[Here 32^2 is 1024] 
#             # H_out = H_in * upscale_factor = 7*32 = 224
#             # W_out = W_in * upscale_factor = 7*32 = 224
#             # self.mask_based_conv = nn.Sequential(nn.Conv2d(768, 4**2*3, kernel_size=1), nn.PixelShuffle(4))
#                                         # nn.Tanh())
#             # self.before_iid = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1), nn.ReLU(True))#, nn.PixelShuffle(32))#,
#                                         # nn.Tanh())
#             self.swin_upsample = nn.Sequential(
#                                     nn.Upsample(scale_factor=4, mode='bilinear'),
#                                     nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
#                                     nn.BatchNorm2d(512), activation
#                                     )
#             # self.mask_based_conv = nn.Sequential(nn.Conv2d(768, 32**2*3, kernel_size=1), nn.PixelShuffle(32),norm_layer(64), activation)#,
#         else:
#             self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
#                                             norm_layer(64), activation)
#             ### downsample
#             self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                                     norm_layer(128), activation)
#             self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#                                     norm_layer(256), activation)
#             self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#                                     norm_layer(512), activation)
#             if self.deep:
#                 self.down4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#                                         norm_layer(512), activation)



#         ### resnet blocks
#         BN = []
#         for i in range(n_blocks):
#             BN += [
#                 ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
#         self.BottleNeck = nn.Sequential(*BN)

#         if self.deep:
#             self.up4 = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm2d(512), activation
#             )
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256), activation
#         )
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128), activation
#         )
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64), activation
#         )
#         self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
#                                         nn.Tanh())

#     def forward(self, input, dlatents):
#         x = input  # 3*224*224
#         if self.transf==True:
#             pixel_values = self.feature_extractor_swin(images= x, return_tensors="pt")
#             last_hidden_state = self.swin_model(pixel_values) #[B, 49, 768] 49 since image was split in 7x7 regions and each region has an emb of 768 dim
#             last_hidden_states = last_hidden_states.transpose(1, 2) #[B, 768, 49])   
#             batch_size, num_channels, sequence_length = last_hidden_states.shape
#             height = width = int(sequence_length**0.5) #7
#             sequence_output = last_hidden_states.reshape(batch_size, num_channels, height, width) #[B, 768, 7, 7]
#             x = self.swin_upsample(sequence_output) #[B, 512, 28, 28]
#         else:
#             skip1 = self.first_layer(x)
#             skip2 = self.down1(skip1)
#             skip3 = self.down2(skip2)
#             if self.deep:
#                 skip4 = self.down3(skip3)
#                 x = self.down4(skip4)
#             else:
#                 x = self.down3(skip3)

#         for i in range(len(self.BottleNeck)):
#             x = self.BottleNeck[i](x, dlatents)

#         if self.deep:
#             x = self.up4(x)
#         x = self.up3(x) #[4, 256, 56, 56]
#         x = self.up2(x) #[4, 128, 112, 112])
#         x = self.up1(x) #[4, 64, 224, 224]
#         x = self.last_layer(x) #[4, 3, 224, 224]
#         x = (x + 1) / 2

#         return x

# class Discriminator(nn.Module):
#     def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
#         super(Discriminator, self).__init__()

#         kw = 4
#         padw = 1
#         self.down1 = nn.Sequential(
#             nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)
#         )
#         self.down2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
#             norm_layer(128), nn.LeakyReLU(0.2, True)
#         )
#         self.down3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
#             norm_layer(256), nn.LeakyReLU(0.2, True)
#         )
#         self.down4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw),
#             norm_layer(512), nn.LeakyReLU(0.2, True)
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=kw, stride=1, padding=padw),
#             norm_layer(512),
#             nn.LeakyReLU(0.2, True)
#         )

#         if use_sigmoid:
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()
#             )
#         else:
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
#             )

#     def forward(self, input):
#         out = []
#         x = self.down1(input)
#         out.append(x)
#         x = self.down2(x)
#         out.append(x)
#         x = self.down3(x)
#         out.append(x)
#         x = self.down4(x)
#         out.append(x)
#         x = self.conv1(x)
#         out.append(x)
#         x = self.conv2(x)
#         out.append(x)
        
#         return out
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from Swin_Models.swin_transformer import SwinTransformer
from Swin_Models.swin_transformer_v2 import SwinTransformerV2
from Swin_Models.swin_transformer_AdaIN import SwinTransformerAdaIN
import torch
import torch.nn as nn


# import wandb
# from transformers import AutoFeatureExtractor, SwinModel, SwinConfig, SwinForMaskedImageModeling
import wandb
from .swinIR import SwinIR 



######################################


class SwinConfig():
    r"""
    This is the configuration class to store the configuration of a [`SwinModel`]. It is used to instantiate a Swin
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Swin
    [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
    architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        depths (`list(int)`, *optional*, defaults to [2, 2, 6, 2]):
            Depth of each layer in the Transformer encoder.
        num_heads (`list(int)`, *optional*, defaults to [3, 6, 12, 24]):
            Number of attention heads in each layer of the Transformer encoder.
        window_size (`int`, *optional*, defaults to 7):
            Size of windows.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
        qkv_bias (`bool`, *optional*, defaults to True):
            Whether or not a learnable bias should be added to the queries, keys and values.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        use_absolute_embeddings (`bool`, *optional*, defaults to False):
            Whether or not to add absolute position embeddings to the patch embeddings.
        patch_norm (`bool`, *optional*, defaults to True):
            Whether or not to add layer normalization after patch embedding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        encoder_stride (`int`, `optional`, defaults to 32):
            Factor to increase the spatial resolution by in the decoder head for masked image modeling.
        Example:
    ```python
    >>> from transformers import SwinModel, SwinConfig
    >>> # Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> configuration = SwinConfig()
    >>> # Initializing a model from the microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> model = SwinModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        patch_norm=True,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        encoder_stride=32,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.path_norm = patch_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        # we set the hidden_size attribute in order to make Swin work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
#######################################




class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out



class Generator_Adain_Upsample(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=6, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect',
                 transf=False,
                 transf_window_size=4, 
                 transf_embed_dim=256, 
                 transf_mlp_ratio=2, 
                 transf_depths=[ 2,6],
                 transf_num_heads= [ 4, 4]):

        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample, self).__init__()
        wandb.run.log_code(".")
        # wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        activation = nn.ReLU(True)
        self.activation = activation
        self.deep = deep
        self.transf = transf
        self.new_decoder = False
        self.transf_AdaIN = True
        if self.transf_AdaIN ==True:
            # self.swin_model = SwinTransformerAdaIN(embed_dim=256, depths=[ 2,2,6],num_heads= [ 4, 8,8]) # Similar to: "microsoft/swin-tiny-patch4-window7-224") #output after forward[B,49,768]
            self.swin_model = SwinTransformerAdaIN(window_size=transf_window_size, embed_dim=transf_embed_dim, mlp_ratio=transf_mlp_ratio, depths=transf_depths,num_heads=transf_num_heads) # Similar to: "microsoft/swin-tiny-patch4-window7-224") #output after forward[B,49,768]
            # scale = 8 #16
            # self.last_layer = nn.Sequential( nn.BatchNorm2d(512), activation, nn.Conv2d(512, scale**2*3, kernel_size=1) ,nn.PixelShuffle(scale)) # When using hidden_state 2
            # self.last_layer = nn.Sequential( nn.Conv2d(1024, scale**2*3, kernel_size=1) ,nn.PixelShuffle(scale)) # When using hidden_state 2
            # self.up4 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(512), activation
            # )                        
          
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256), activation
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128), activation
            )
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), activation
            )
            self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))
        elif self.transf==True:
            # upscale = 4
            # window_size = 8
            # height = 224#(1024 // upscale // window_size + 1) * window_size
            # width = 224#(720 // upscale // window_size + 1) * window_size
            # self.SwinIR_model = SwinIR(upscale=2, img_size=(height, width),
            #        window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
            #        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
            # self.feature_extractor_swin = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            # # self.feature_extractor_swin.cuda()
            # self.conv_first = nn.Conv2d(3, 768,  kernel_size=3, stride=1, padding=1)
            # configuration = SwinConfig(embed_dim=128, depths=[ 2, 2, 18, 2 ],num_heads= [ 4, 8, 16, 32 ])#,window_size=14)
            # configuration = SwinConfig(embed_dim=128, depths=[ 2, 2, 6],num_heads= [ 4, 8, 16])#, 32 ])#,window_size=14)
            # configuration = SwinConfig(embed_dim=128, depths=[ 6, 6],num_heads= [ 4, 8])#, 32 ])#,window_size=14)

            # configuration = SwinConfig(embed_dim=256, depths=[ 4, 4],num_heads= [ 4, 8])#, 32 ])#,window_size=14)
            # configuration = SwinConfig(embed_dim=128, depths=[ 2, 2 ,18],num_heads= [ 4,8,8 ])#, 32 ])#,window_size=14)

                        #SwinConfig {
                        #   "attention_probs_dropout_prob": 0.0,
                        #   "depths": [
                        #     2,
                        #     2,
                        #     6,
                        #     2
                        #   ],
                        #   "drop_path_rate": 0.1,
                        #   "embed_dim": 96,
                        #   "encoder_stride": 32,
                        #   "hidden_act": "gelu",
                        #   "hidden_dropout_prob": 0.0,
                        #   "hidden_size": 768,
                        #   "image_size": 224,
                        #   "initializer_range": 0.02,
                        #   "layer_norm_eps": 1e-05,
                        #   "mlp_ratio": 4.0,
                        #   "model_type": "swin",
                        #   "num_channels": 3,
                        #   "num_heads": [
                        #     3,
                        #     6,
                        #     12,
                        #     24
                        #   ],
                        #   "num_layers": 4,
                        #   "patch_size": 4,
                        #   "path_norm": true,
                        #   "qkv_bias": true,
                        #   "transformers_version": "4.17.0",
                        #   "use_absolute_embeddings": false,
                        #   "window_size": 7
                        # }

            # print(configuration)
                            # {
                            # "do_normalize": true,
                            # "do_resize": true,
                            # "feature_extractor_type": "ViTFeatureExtractor",
                            # "image_mean": [
                            #     0.485,
                            #     0.456,
                            #     0.406
                            # ],
                            # "image_std": [
                            #     0.229,
                            #     0.224,
                            #     0.225
                            # ],
                            # "resample": 3,
                            # "size": 224
                            # }
            # self.swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224") 
            
            # self.swin_model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-patch4-window7-224")
            # self.post_swin = nn.Sequential(norm_layer(256), activation)
            
            # self.swin_model = SwinModel.from_pretrained("../simmim/simmim_finetune__swin_base__img224_window7__100ep")
            # self.swin_model = SwinModel(configuration) # Similar to: "microsoft/swin-tiny-patch4-window7-224") #output after forward[B,49,768]
            self.swin_model = SwinTransformer(embed_dim=128, depths=[ 2, 2,6],num_heads= [ 4, 8, 8]) # Similar to: "microsoft/swin-tiny-patch4-window7-224") #output after forward[B,49,768]
            # self.swin_model = SwinTransformerV2(embed_dim=128, depths=[ 2, 2],num_heads= [ 4, 8]) # Similar to: "microsoft/swin-tiny-patch4-window7-224") #output after forward[B,49,768]
            
            # self.post_swin = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1), norm_layer(512), activation)
            # self.post_swin = nn.Sequential(norm_layer(512), activation)
            print("======== SWIN MODEL CONFIGURATION ================")
            # print(self.swin_model.config)
            print("==================================================")
            # We process the output and get [B,768,7,7]
            # wandb.watch(self.swin_model)
            # for param in self.swin_model.parameters():
            #     param.requires_grad = False
            # # self.replication_l = nn.ReplicationPad2d((0, 0, 367, 368))
            # self.swin_model.train(False) 
            # self.conv_gen = nn.Conv2d(1, 512, kernel_size=1, padding=0)
            #in_channels=num_features, out_channels=config.encoder_stride**2 * 3,
            # 32^2 * 3 = 1024 * 3 = 3072
            # PixelShuffle  [https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html]
            # useful for implementing efficient sub-pixel convolution with a stride of 1/r
            # input (*, C_in, H_in, W_in)
            # output (*, C_out, H_out, W_out)
            # where:
            # C_out = C_in / upscale_factor^2 => 3072/(32^2) = 3[Here 32^2 is 1024] 
            # H_out = H_in * upscale_factor = 7*32 = 224
            # W_out = W_in * upscale_factor = 7*32 = 224
            # self.mask_based_conv = nn.Sequential(nn.Conv2d(768, 32**2*3, kernel_size=1), nn.PixelShuffle(32))#, #works
            #Goal is to match the 512x28x28 of simswap
            # 7*4 = 28 so upscale factor is 4.
            # 4^2 * 3 = 16 * 3 = 48
            # 512 = 768 / x^2 => x^2 = 768/512 => x^2 = 1.5 => x = 1.224
            # C_out = C_in / upscale_factor^2 => 3072/(4^2) = 192[Here 4^2 is 16] 
            # H_out = H_in * upscale_factor = 7*4 = 28
            # W_out = W_in * upscale_factor = 7*4 = 28
            # self.swin_upsample = nn.Sequential(nn.Conv2d(768, 4**2*512, kernel_size=1), nn.PixelShuffle(4))
            # self.swin_upsample = nn.Sequential(
            #                         nn.Upsample(scale_factor=4, mode='bilinear'),
            #                         nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
            #                         nn.BatchNorm2d(512), activation
            #                         )
            # self.mask_based_conv = nn.Sequential(nn.Conv2d(D, 32**2*3, kernel_size=1), nn.PixelShuffle(32),norm_layer(64), activation)#,
            
        else:
            self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                            norm_layer(64), activation)
            ### downsample
            self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                    norm_layer(128), activation)
            self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                    norm_layer(256), activation)
            self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                    norm_layer(512), activation)
                                    
            if self.deep:
                self.down4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                        norm_layer(512), activation)
        
        if self.new_decoder == False and self.transf_AdaIN ==False:
            ### resnet blocks
            BN = []
            for i in range(n_blocks):
                BN += [
                    ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
            self.BottleNeck = nn.Sequential(*BN)

            # if self.deep:
            #     self.up4 = nn.Sequential(
            #         nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            #         nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(512), activation
            #     )
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256), activation
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128), activation
            )
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), activation
            )
            self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))
        elif self.transf_AdaIN ==False:
            ### resnet blocks
            BN = []
            for i in range(n_blocks):
                BN += [
                    ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
            self.BottleNeck = nn.Sequential(*BN)
            # self.up2 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            #     nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(768), activation
            # )
            # self.up4 = nn.Sequential(
            #         nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False), #7=>14
            #         nn.Conv2d(768, 640, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(640), activation
            #     )
            # self.up3 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False), #14->28
            #     nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(512), activation
            # )
            # self.up2 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False), #28 -> 56
            #     nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(384), activation
            # )
            # self.up1 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False), #56 ->112
            #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(256), activation
            # )
            # self.up0 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False), #56 ->112
            #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(128), activation
            # )
            # self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(128, output_nc, kernel_size=7, padding=0))
            
            # self.up1 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False), #112 ->224
            #     nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(3), activation
            # )
            # self.last_layer = nn.Sequential(nn.Conv2d(768, 32**2*3, kernel_size=1), nn.Tanh(), nn.PixelShuffle(32)) # Run 9
            # self.conv_after_body = nn.Conv2d(768, 768, 3, 1, 1)
            # scale = 8#16
            # self.last_layer = nn.Sequential(nn.Conv2d(512, 16**2*3, kernel_size=1) ,nn.PixelShuffle(16)) # Run 8 #When using hidden_state 2
            # self.last_layer = nn.Sequential( nn.Conv2d(512, scale**2*3, kernel_size=1) ,nn.PixelShuffle(scale)) # When using hidden_state 2
            ###JUNE
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256), activation
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128), activation
            )
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), activation
            )
            self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))



            # scale = 8#16
            # self.semi_last = nn.Sequential( nn.BatchNorm2d(512), nn.Tanh())
            # self.last_layer = nn.Sequential(  nn.Conv2d(512, scale**2*3, kernel_size=1) ,nn.PixelShuffle(scale)) # When using hidden_state 2


    def forward(self, input, dlatents, return_stats=False):
        x = input  # 3*224*224
        ##################################################
        ########### ENCODER ##############################
        ##################################################

        if self.transf==True:
            # print(x.shape)
            # pixel_values = self.feature_extractor_swin(images= x, return_tensors="pt")
            # pixel_values = self.conv_first(x) 
            pixel_values = x
            # sequence_output = self.SwinIR_model(x)
            if self.transf_AdaIN == True:
                 swin_output = self.swin_model(pixel_values,dlatents,output_hidden_states=True)
                #  print('ABSHDJ')
                #  print(swin_output.shape)
            else:
                swin_output = self.swin_model(pixel_values,output_hidden_states=True)
            # torch.Size([16, 3136, 128]) [0]
            # torch.Size([16, 784, 256]) [1] -28
            # torch.Size([16, 196, 512]) #May 24   -14
            # torch.Size([16, 49, 1024]) [3] -
            # torch.Size([16, 49, 1024]) [4]
            
            # last_hidden_state = swin_output.hidden_states[1] #[B, 49, 768] 49 since image was split in 7x7 regions and each region has an emb of 768 dim
            last_hidden_state = swin_output #[B, 49, 768] 49 since image was split in 7x7 regions and each region has an emb of 768 dim
            # print(last_hidden_state.shape)
            # print(last_hidden_state.shape)
            last_hidden_states = last_hidden_state.transpose(1, 2) #[B, 768, 49])   
            batch_size, num_channels, sequence_length = last_hidden_states.shape
            height = width = int(sequence_length**0.5) #7
            sequence_output = last_hidden_states.reshape(batch_size, num_channels, height, width) #[B, 768, 7, 7]
            if self.transf_AdaIN == False:
                last_hidden_state = self.post_swin(sequence_output)
            else:
                last_hidden_state = sequence_output
                # print(last_hidden_state.shape)

            sequence_output = last_hidden_state # I know is redundant so should clean, its used to quickly iterate printing stats 

            x = sequence_output   
            # print(x.shape)
        else: #Standard simswap
            skip1 = self.first_layer(x)
            skip2 = self.down1(skip1)
            skip3 = self.down2(skip2)
            if self.deep:
                skip4 = self.down3(skip3)
                x = self.down4(skip4)
            else:
                x = self.down3(skip3)
            if return_stats:
                    last_hidden_state = x
        ################################################################
        ######### Bottleneck - IID ###############################
        ############################################
        if self.transf_AdaIN ==False:
            for i in range(len(self.BottleNeck)):
                x = self.BottleNeck[i](x, dlatents)
            #     bot.append(x)
            if return_stats:
                std_mean_bottleneck = torch.std_mean(x)
                max_bottleneck = torch.max(x)
                min_bottleneck = torch.min(x)
        else:
            if return_stats:
                std_mean_bottleneck = [-99999, -99999]
                max_bottleneck = -99999
                min_bottleneck = -99999


        #############################################################
        ######## Decoder #######################
        ###############################

        if self.new_decoder == False:   
            # x = self.up4(x)
       
            if self.deep:
                x = self.up4(x)
                # features.append(x)
            # print(x.shape)
            x = self.up3(x)#Output: [B, 256, 56, 56]
            # features.append(x)
            x = self.up2(x)#[B, 128, 112, 112])
            # features.append(x)
            x = self.up1(x)#[B, 64, 224, 224]
                # features.append(x)            
            # x = self.conv_after_body(x) + pixel_values
            # x = self.upsample(x)
            semi_last_out = x
            x = self.last_layer(x)
        else:                               
            semi_last_out = self.semi_last(x)
            x = semi_last_out
            x = self.last_layer(x)#[B, 3, 224, 224]
             
        # exit(0)                     
        ###################################3
        ##### STATS @@#################33
        ##############################

        
        if return_stats:
            std_mean_last_hidden_state = torch.std_mean(last_hidden_state)
            max_last_hidden_state = torch.max(last_hidden_state)
            min_last_hidden_state = torch.min(last_hidden_state)

            std_mean_semi_last = torch.std_mean(semi_last_out)
            max_semi_last = torch.max(semi_last_out)
            min_semi_last = torch.min(semi_last_out)


            std_mean_img_fake = torch.std_mean(x)
            max_img_fake = torch.max(x)
            min_img_fake = torch.min(x)
                
            stats_tensors = {
            "std_encoder_out":std_mean_last_hidden_state[0] ,
            "mean_encoder_out":std_mean_last_hidden_state[1] ,
            "max_encoder_out":max_last_hidden_state,
            "min_encoder_out":min_last_hidden_state,

            "std_bottleneck":std_mean_bottleneck[0] ,
            "mean_bottleneck":std_mean_bottleneck[1] ,
            "max_bottleneck":max_bottleneck,
            "min_bottleneck":min_bottleneck,

            "std_semi_last":std_mean_semi_last[0] ,
            "mean_semi_last":std_mean_semi_last[1] ,
            "max_semi_last":max_semi_last,
            "min_semi_last":min_semi_last ,
            
            "std_img_fake":std_mean_img_fake[0] ,
            "mean_img_fake":std_mean_img_fake[1] ,
            "max_img_fake":max_img_fake,
            "min_img_fake":min_img_fake            
        }
            return x, stats_tensors
        else:
            return x


#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: fs_model_fix_idnorm_donggp_saveoptim copy.py
# Created Date: Wednesday January 12th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 21st April 2022 8:13:37 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn
# import wandb
from .base_model import BaseModel
# from .fs_networks_fix import Generator_Adain_Upsample
from .fs_networks_transformer import Generator_Adain_Upsample

from pg_modules.projected_discriminator import ProjectedDiscriminator
  
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
        self.isTrain = opt.isTrain
        torch.cuda.set_device(int(opt.gpu_ids[0]))
        # Generator network
        print("creating generator...")
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep, transf=opt.transf,
                                            transf_window_size=opt.window_size, 
                                            transf_embed_dim=opt.transf_embed_dim, 
                                            transf_mlp_ratio=opt.mlp_ratio, 
                                            transf_depths=opt.depth,
                                            transf_num_heads=opt.heads)
        
        print("generator moving to cuda...")
        self.netG.cuda(int(opt.gpu_ids[0]))
        # wandb.watch(self.netG )
        print('================  Generator Architecture ================')
        print( self.netG)
        print('=========================================================')

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint['model'].module
        self.netArc = self.netArc.cuda(int(opt.gpu_ids[0]))
        self.netArc.eval()
        self.netArc.requires_grad_(False)

        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(int(opt.gpu_ids[0])))
        if not self.isTrain:
            pretrained_path =  opt.checkpoints_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            return
        
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        # self.netD.feature_network.requires_grad_(False)
        self.netD.cuda(int(opt.gpu_ids[0]))

    
        if self.isTrain:
            # define loss functions
            self.criterionFeat  = nn.L1Loss()
            self.criterionRec   = nn.L1Loss()

           # initialize optimizers

            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)
            # self.scheduler = torch.optim.MultiStepLR(self.optimizer_G, milestones=[1000,10000], gamma=0.5)
        # load networks
        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print (pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
        torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return self.cos(x1, x2)
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))
        # return torch.dot(x1 , x2) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))



    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)
        '''if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)'''

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr



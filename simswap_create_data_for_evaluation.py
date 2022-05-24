
"""
This script is used to generate the swapped faces of the 10k dataset of faceforensics++ to perform evaluation in the model
"""
# srun python dataset_creation_wholeimage_swapsingle.py --isTrain false --use_mask  --name people --Arc_path arcface_model/arcface_checkpoint.tar --dataset_both_folder "/home/racmulsa/10k_dataset/original/both/" --output_path "/home/racmulsa/10k_dataset/simswap/" --no_simswaplogo

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import wandb

from models.projected_model import fsModel
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os 
from util.norm import SpecificNorm 
import os
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")
def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def str2bool(v):
    return v.lower() in ('true')

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def generate_data_with_model(opt, model_name = 'simswap_retry'):
    opt.output_path =  os.path.join("../../deep_fake_master_thesis/10k_dataset/",model_name)
    opt.checkpoints_dir = os.path.join("./checkpoints/",model_name)
    opt.output_path =  os.path.join(opt.output_path,str(opt.which_epoch))
    # opt.which_epoch = 500000
    print(opt)
    opt.gpu_ids = [0]
    torch.cuda.set_device(opt.gpu_ids[0])
    # start_epoch, epoch_iter = 1, 0
    crop_size = 224

    # torch.nn.Module.dump_patches = True
    logoclass = None# watermark_image('./simswaplogo/simswaplogo.png')

    model = fsModel()
    model.initialize(opt)
    model.eval()
    model.netG.eval()

    spNorm = SpecificNorm()
    app = Face_detect_crop(name='antelope', root='./insightface_func/models',cuda_device = int(opt.gpu_ids[0]))
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    
    list_folders_dataset = os.listdir(opt.dataset_both_folder)
    errors_list = []

    mean = torch.tensor([0.485, 0.456, 0.406]).cuda(int(opt.gpu_ids[0])).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda(int(opt.gpu_ids[0])).view(1,3,1,1)
    print("Swapping faces for",str(len(list_folders_dataset)), 'sources images')


    os.makedirs(opt.output_path, exist_ok=True)
    with torch.no_grad():
        for folder in tqdm(list_folders_dataset):
            try:
                data_in_folder = os.listdir( os.path.join(opt.dataset_both_folder,folder))
                src_pic_path = [filename for filename in data_in_folder if filename.split('_')[0]=='src'][0] #want only one element
                trj_pics_list = [filename for filename in data_in_folder if filename.split('_')[0]=='trj'] #want the whole list of files
                pic_a = os.path.join(opt.dataset_both_folder,folder, src_pic_path) #opt.pic_a_path
                # print(pic_a)
                img_a_whole = cv2.imread(pic_a)
                img_a_align_crop, _ = app.get(img_a_whole,crop_size)
                img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
                img_a = transformer_Arcface(img_a_align_crop_pil)
                img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        
                # convert numpy to tensor
                img_id = img_id.cuda(int(opt.gpu_ids[0]))
                # print('Creating latent')
                #create latent id
                img_id_downsample = F.interpolate(img_id, size=(112,112), mode='bicubic')
                # print(img_id_downsample.shape)
                latend_id = model.netArc(img_id_downsample)
                latend_id = F.normalize(latend_id, p=2, dim=1)
        
                # print('forward pass')
                ############## Forward Pass ######################
                for pic_b_path in trj_pics_list:
                  try:
                      pic_b = os.path.join(opt.dataset_both_folder,folder, pic_b_path) #opt.pic_b_path
                      output_file = os.path.join(opt.output_path, pic_b_path.split('.')[0]+'_whole_swapsingle.jpg')
                      if os.path.isfile(output_file):
                        continue
                      img_b_whole = cv2.imread(pic_b)
                    #   print('Swaping faces from source',src_pic_path,'  to  tarjet' ,pic_b_path)
                      img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
                      # detect_results = None
                      swap_result_list = []
              
                      b_align_crop_tenor_list = []
              
                      for b_align_crop in img_b_align_crop_list:
              
                          b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                          batch, c, w, h = b_align_crop_tenor.shape
                          b_align_crop_tenor = b_align_crop_tenor.view(-1, c, w,h)   #HARDCODED
                        #   print(b_align_crop_tenor.shape)

                          b_align_crop_tenor =  b_align_crop_tenor.sub_(mean).div_(std)
                          swap_result = model.netG(b_align_crop_tenor, latend_id)#(None, b_align_crop_tenor, latend_id, None, True)[0]
                          swap_result = swap_result.mul_(std).add_(mean)#
                          swap_result_list.append(swap_result)
                          b_align_crop_tenor_list.append(b_align_crop_tenor)             
                      
                          net =None
              
                      reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, img_b_whole, logoclass, \
                          output_file, no_simswaplogo=True,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)
              
                    #   print(' ')
                  except:
                      errors_list.append('Error with pic b'+ pic_b_path)
                      with open('error_list.txt', 'w') as f:
                          for item in errors_list:
                              f.write("%s\n" % item)
            except Exception as e:
                errors_list.append('Error with folder '+ folder)
                with open('error_list.txt', 'w') as f:
                    for item in errors_list:
                        f.write("%s\n" % item)

        print('************ Done ! ************')

if __name__ == '__main__':
    opt = TestOptions()
    opt.parser.add_argument("--dataset_both_folder", type=str, default="../../deep_fake_master_thesis/10k_dataset/original/both/", help="Dataset folder whith the 10k data frames")
    # opt.parser.add_argument('--gpu_ids', default='1')
    # opt.parser.add_argument('--name', type=str, default='simswap', help='name of the experiment. It decides where to store samples and models')
    opt.parser.add_argument('--Gdeep', type=str2bool, default='False')
    opt.parser.add_argument('--transf', type=str2bool, default='False')

    opt= opt.parse()
    opt.output_path = "../../deep_fake_master_thesis/10k_dataset/"
    opt.checkpoints_dir = "./checkpoints/"
    for checkpoint_num in range(100000,400000,50000):
        # if checkpoint_num == 200000 or checkpoint_num == 300000:

        opt.which_epoch = checkpoint_num
        print('Generating data for epoch ',checkpoint_num)
        generate_data_with_model(opt,opt.name)
# python simswap_create_data_for_evaluation.py --isTrain false --name simswap224_retry --Arc_path arcface_model/arcface_checkpoint.tar --dataset_both_folder "../../deep_fake_master_thesis/10k_dataset/original/both/" --output_path  "../../deep_fake_master_thesis/10k_dataset/" --no_simswaplogo
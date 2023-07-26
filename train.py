import os
import random
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDataset
from utils.funcs import EarlyStop, denorm
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.visualize import visualize_tensor
from utils.downsample import bicubic_sampling
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.swinir_arch import SwinIR
from models.swin2sr import Swin2SR
#from models.SwinIR import SwinIR
from datetime import datetime
from losses.gms_loss import MSGMS_Loss
from losses.ssim_loss import SSIM_Loss
import warnings
from torchmetrics.functional import peak_signal_noise_ratio as psnr 

warnings.filterwarnings('ignore')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def main(): 
    parser = argparse.ArgumentParser(description='SRAD anomaly detection')
    parser.add_argument('--obj', type=str, default='pcb1')
    parser.add_argument('--data_type', type=str, default='visa')
    parser.add_argument('--scale_factor', type=int, default=8)
    parser.add_argument('--aug_mode', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--sr_model', type=str, default='swinir')
    parser.add_argument('--loss', type=str, default='L2SG')
    
    #parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--epochs', type=int, default=400, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--grayscale', action='store_true', help='color or grayscale input image')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--belta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    #parser.add_argument('--k_value', type=int, nargs='+', default=[2, 4, 8, 16])
    args = parser.parse_args()

    args.input_channel = 1 if args.grayscale else 3

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    if args.wandb:
        import wandb
        now = datetime.now()
        formattedDate = now.strftime("%Y%m%d_%H%M%S")
        wandb_title = "SRAD_" +args.data_type +args.obj+ formattedDate 
        wandb.init(project=wandb_title)
    print("using device: "+str(device))


    args.prefix = time_file_str()
    args.save_dir = './results/' + args.sr_model+ '/' + args.data_type + '/' + args.obj + '/seed_{}/'.format(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if args.data_type == 'mvtec':
        train_dataset = MVTecDataset(class_name=args.obj,aug_mode=args.aug_mode, resize=args.img_size, patch_size=args.patch_size, scale_factor=args.scale_factor)
        test_dataset = MVTecDataset(class_name=args.obj, is_train=False, resize=args.img_size,patch_size=args.patch_size, scale_factor=args.scale_factor)
    elif args.data_type == 'visa':
        train_dataset = VisaDataset(class_name=args.obj,aug_mode=args.aug_mode, resize=args.img_size,patch_size=args.patch_size, scale_factor=args.scale_factor)
        test_dataset = VisaDataset(class_name=args.obj, is_train=False, resize=args.img_size,patch_size=args.patch_size, scale_factor=args.scale_factor)
        #import pdb; pdb.set_trace()
    else:
        assert "Data type not exist."

    img_nums = len(train_dataset)
    valid_num = int(img_nums * args.validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    
    

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    
    ##>>>>> model selection <<<<<
    if args.sr_model == 'swin2sr':
        model = Swin2SR(
            upscale=args.scale_factor,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6], 
            embed_dim=180, 
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, 
            upsampler='pixelshuffle', 
            resi_connection='1conv'
            )

    elif args.sr_model == 'swinir_classic':
        model = SwinIR(
            upscale=args.scale_factor,
            in_chans=3,
            img_size=int(args.patch_size/args.scale_factor),
            window_size=8,
            img_range=1.,
            depths=[6,6,6,6,6,6],
            embed_dim=180,
            num_heads=[6,6,6,6,6,6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
            )
    elif args.sr_model == 'swinir_light':
        model = SwinIR(
            upscale=args.scale_factor,
            in_chans=3,
            img_size=int(args.patch_size/args.scale_factor),
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='1conv'
            )
        
        if args.pretrained:
            param_key_g = "params"
            if args.scale_factor == 2:
                pretrained_dict = torch.load("weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth")
            elif args.scale_factor == 4:
                pretrained_dict = torch.load("weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
            else:
                assert("not available weights for this scale: "+str(args.scale_factor))
            model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)
    ############################

    elif args.sr_model == 'edsr':
        model= EDSR(num_in_ch=3,num_out_ch=3,num_block=16,upscale=args.scale_factor)
        
        if args.pretrained:
            state_dict = torch.load("weights/EDSR_Mx2_f64b16_DIV2K_official-3ba7b086.pth")
            model.load_state_dict(state_dict['params'])

    elif args.sr_model == 'edsr_b16f128':
        model= EDSR(num_in_ch=3,num_out_ch=3,num_block=16,num_feat=128,upscale=args.scale_factor)

    elif args.sr_model == 'edsr_b32f64':
        model= EDSR(num_in_ch=3,num_out_ch=3,num_block=32,num_feat=64,upscale=args.scale_factor)

    elif args.sr_model == 'edsr_b32f128':
        model= EDSR(num_in_ch=3,num_out_ch=3,num_block=32,num_feat=128,upscale=args.scale_factor)

    else: 
        assert("not implemented model")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print_log("Total parameters: {:,}".format(total_params),log)

    #asd = torch.load("weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
    #print(asd.keys())
    # import pdb;pdb.set_trace()
    

    ## >>>>>pretrained<<<<
    
    ####################################
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.99))

    ##  fetch fixed data
    _,_,_,x_normal_fixed, _ = next(iter(val_loader))
    x_normal_fixed = x_normal_fixed.to(device)

    _,_,_,x_test_fixed, _ = next(iter(test_loader))
    x_test_fixed = x_test_fixed.to(device)

    # start training
    save_name = os.path.join(args.save_dir, '{}_model.pt'.format(args.obj))
    early_stop = EarlyStop(patience=20, save_name=save_name)
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        train_loss = train(args, model, epoch, train_loader, optimizer, log)
        val_loss = val(args, model, epoch, val_loader, log)

        if epoch % 20 == 0:
            save_sample = os.path.join(args.save_dir, '{}-images.jpg'.format(epoch))
            save_sample2 = os.path.join(args.save_dir, '{}test-images.jpg'.format(epoch))
            save_snapshot(args, x_normal_fixed, x_test_fixed, model, save_sample, save_sample2, log)

        if (early_stop(val_loss, model, optimizer, log)):
            break
        
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        if args.wandb:
            wandb.log({"epoch": epoch, "loss": train_loss})
    log.close()


    #train(args, )

def train(args, model, epoch, train_loader, optimizer, log):
    model.train()

    ### Loss selection 
    if args.loss == "L1":
        l1_losses = AverageMeter()
        mae = nn.L1Loss(reduction='mean')

    elif args.loss == "L2SG":
        l2_losses = AverageMeter()
        gms_losses = AverageMeter()
        ssim_losses = AverageMeter()
        ssim = SSIM_Loss()
        mse = nn.MSELoss(reduction='mean')
        msgms = MSGMS_Loss()

    ### Training part 
    for (_, _, _, hr_patch,lr_patch) in tqdm(train_loader):
        optimizer.zero_grad()
        ## feed to gpu
        hr_patch = hr_patch.to(device)
        lr_patch = lr_patch.to(device)
        
        output = model(lr_patch)
        # import pdb; pdb.set_trace()
        #save_image(denorm(x_concat.data.cpu()), save_dir, nrow=1, padding=0)
        
        # >>>>>> with L2 ssim gms Loss <<<<<<<<
        if args.loss == "L2SG":
            l2_loss = mse(hr_patch, output)
            gms_loss = msgms(hr_patch, output)
            ssim_loss = ssim(hr_patch, output)
            loss = args.gamma * l2_loss + args.alpha * gms_loss + args.belta * ssim_loss

            l2_losses.update(l2_loss.item(), hr_patch.size(0))
            gms_losses.update(gms_loss.item(), hr_patch.size(0))
            ssim_losses.update(ssim_loss.item(), hr_patch.size(0))
        #########################################
        # >>>>>> with L1 Loss <<<<<<<<
        if args.loss == "L1":
            
            l1_loss = mae(hr_patch, output)
            loss = args.gamma * l1_loss
            l1_losses.update(l1_loss.item(), hr_patch.size(0))
        #######################################

        loss.backward()
        optimizer.step()

    if args.loss == "L1":
        print_log(('Train Epoch: {} L1_Loss: {:.6f}'.format(
        epoch, l1_losses.avg)), log)

    if args. loss == "L2SG":
        print_log(('Train Epoch: {} L2_Loss: {:.6f} GMS_Loss: {:.6f} SSIM_Loss: {:.6f}'.format(
            epoch, l2_losses.avg, gms_losses.avg, ssim_losses.avg)), log)
        
    return loss

def val(args, model, epoch, val_loader, log):
    model.eval()
    losses = AverageMeter()
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()
    mae = nn.L1Loss(reduction='mean')
    psnr_scores = []
    ## val part 
    for (_, _, _, hr_patch, lr_patch) in tqdm(val_loader):
        hr_patch = hr_patch.to(device)
        lr_patch = lr_patch.to(device)
        

        with torch.no_grad():
            output = model(lr_patch)
            if args.loss == "L1":
                l1_loss = mae(hr_patch, output)
                loss = args.gamma * l1_loss
            elif args.loss == "L2SG":
                l2_loss = mse(hr_patch, output)
                gms_loss = msgms(hr_patch, output)
                ssim_loss = ssim(hr_patch, output)
                loss = args.gamma * l2_loss + args.alpha * gms_loss + args.alpha * ssim_loss
            psnr_scores.append(float(psnr(hr_patch, output)))
            losses.update(loss.item(), hr_patch.size(0))
    
    

    psnr_score = np.mean(psnr_scores)

    print_log(('Valid Epoch: {} loss: {:.6f} psnr: {:.6f}'.format(epoch, losses.avg, psnr_score)), log)

    return losses.avg

def save_snapshot(args, x, x2, model, save_dir, save_dir2, log):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        ### NOTE: MUST MODIFY scale factor general way and float for basicsr
        
        downsampled = bicubic_sampling(x,scale_factor=1/args.scale_factor)
        recon = model(downsampled.float())
        # import pdb;pdb.set_trace()
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        #save_image(denorm(x_concat.data.cpu()), save_dir, nrow=1, padding=0)
        save_image(denorm(x_concat.data.cpu()), save_dir, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir)), log)
        
        x_fake_list = x2
        downsampled = bicubic_sampling(x2,scale_factor=1/args.scale_factor)
        recon = model(downsampled.float())
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        #save_image(denorm(x_concat.data.cpu()), save_dir2, nrow=1, padding=0)
        save_image(denorm(x_concat.data.cpu()), save_dir2, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir2)), log)

def adjust_learning_rate(args, optimizer, epoch):
    if epoch == 250:
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



if __name__=="__main__":
    
    main()


    # model = EDSR(num_in_ch=num_in_ch, num_out_ch=num_out_ch)
    # model.to(device)

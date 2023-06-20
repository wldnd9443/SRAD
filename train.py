import os 
import random
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDataset
from utils.funcs import EarlyStop, denorm
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.visualize import visualize_tensor
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.swinir_arch import SwinIR
#from models.SwinIR import SwinIR
from datetime import datetime
from losses.gms_loss import MSGMS_Loss
from losses.ssim_loss import SSIM_Loss
from utils.downsample import bicubic_sampling, bilinear_sampling
import warnings
warnings.filterwarnings('ignore')
# ## train parameter
# device = 'cuda'
# use_cuda = True
# val_ratio = 0.1
# batch_size = 32
# epoch = 5
# lr = 0.01
# weight_decay = 0.00001

## model parameter
num_in_ch = 3
num_out_ch = 3
#scale_factor = 4 

## ETC
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def main(): 
    parser = argparse.ArgumentParser(description='SRAD anomaly detection')
    parser.add_argument('--obj', type=str, default='toothbrush')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--aug_mode', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--sr_model', type=str, default='swinir')
    
    #parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--epochs', type=int, default=300, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--grayscale', action='store_true', help='color or grayscale input image')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--belta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of Adam')
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
    args.save_dir = './results' + args.data_type + '/' + args.obj + '/seed_{}/'.format(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if args.data_type == 'mvtec':
        train_dataset = MVTecDataset(class_name=args.obj,aug_mode=args.aug_mode)
        test_dataset = MVTecDataset(class_name=args.obj, is_train=False)
    elif args.data_type == 'visa':
        train_dataset = VisaDataset(class_name=args.obj,aug_mode=args.aug_mode)
        test_dataset = VisaDataset(class_name=args.obj, is_train=False)
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

    
    ##>>>> model selection <<<<<
    
    if args.sr_model == 'swinir':
        model = SwinIR(
                upscale=args.scale_factor,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6,6,6,6,6,6],
                embed_dim=180,
                num_heads=[6,6,6,6,6,6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv'
                )
        if args.pre_trained:
            param_key_g = "params"
            pretrained_dict = torch.load("weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
            model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)
    ############################

    if args.sr_model == 'edsr':
        model= EDSR(num_in_ch=3,num_out_ch=3,upscale=args.scale_factor)
        
        if args.pre_trained:
            state_dict = torch.load("weights/EDSR_Mx2_f64b16_DIV2K_official-3ba7b086.pth")
            model.load_state_dict(state_dict['params'])
    model.to(device)  


    #asd = torch.load("weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
    #print(asd.keys())
    # import pdb;pdb.set_trace()
    

    ## >>>>>pretrained<<<<
    
    ####################################
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ## debug fetch fixed data
    # x_normal_fixed, _, _ = iter(val_loader).next()
    x_normal_fixed, _, _ = next(iter(val_loader))
    x_normal_fixed = x_normal_fixed.to(device)

    # x_test_fixed, _, _ = iter(test_loader).next()
    x_test_fixed, _, _ = next(iter(test_loader))
    x_test_fixed = x_test_fixed.to(device)

    # start training
    save_name = os.path.join(args.save_dir, '{}_{}_model.pt'.format(args.obj, args.prefix))
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

        if epoch % 5 == 0:
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
    
    # l2_losses = AverageMeter()
    gms_losses = AverageMeter()
    ssim_losses = AverageMeter()
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()

    l1_losses = AverageMeter()
    mae = nn.L1Loss(reduction='mean')

    for (data, _, _) in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        downsampled = bilinear_sampling(data, scale_factor=1/args.scale_factor)
        ### >>>>> SwinIR needs this line <<<<
        downsampled = downsampled.float()
        output = model(downsampled)
        #import pdb; pdb.set_trace()

        # >>>>>> with L2 Loss <<<<<<<<
        # l2_loss = mse(data, output)
        # gms_loss = msgms(data, output)
        # ssim_loss = ssim(data, output)

        #loss = args.gamma * l2_loss + args.alpha * gms_loss + args.belta * ssim_loss

        # l2_losses.update(l2_loss.item(), data.size(0))
        # gms_losses.update(gms_loss.item(), data.size(0))
        # ssim_losses.update(ssim_loss.item(), data.size(0))
        #########################################

        # >>>>>> with L1 Loss <<<<<<<<
        l1_loss = mae(data, output)
        gms_loss = msgms(data, output)
        ssim_loss = ssim(data, output)

        loss = args.gamma * l1_loss + args.alpha * gms_loss + args.belta * ssim_loss
        # loss = args.gamma * l1_loss

        l1_losses.update(l1_loss.item(), data.size(0))
        gms_losses.update(gms_loss.item(), data.size(0))
        ssim_losses.update(ssim_loss.item(), data.size(0))
        #######################################

        loss.backward()
        optimizer.step()

    # print_log(('Train Epoch: {} L2_Loss: {:.6f} GMS_Loss: {:.6f} SSIM_Loss: {:.6f}'.format(
    #     epoch, l2_losses.avg, gms_losses.avg, ssim_losses.avg)), log)
    print_log(('Train Epoch: {} L1_Loss: {:.6f} GMS_Loss: {:.6f} SSIM_Loss: {:.6f}'.format(
        epoch, l1_losses.avg, gms_losses.avg, ssim_losses.avg)), log)
    # print_log(('Train Epoch: {} L1_Loss: {:.6f}'.format(
    #     epoch, l1_losses.avg)), log)
    return loss

def val(args, model, epoch, val_loader, log):
    model.eval()
    losses = AverageMeter()
    ssim = SSIM_Loss()
    # mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()
    mae = nn.L1Loss(reduction='mean')
    for (data, _, _) in tqdm(val_loader):
        data = data.to(device)
        
        downsampled =bilinear_sampling(data,scale_factor=1/args.scale_factor)
        downsampled = downsampled.float()
        with torch.no_grad():
            output = model(downsampled)
            l1_loss = mae(data, output)
            # l2_loss = mse(data, output)
            gms_loss = msgms(data, output)
            ssim_loss = ssim(data, output)

            # loss = args.gamma * l2_loss + args.alpha * gms_loss + args.alpha * ssim_loss
            loss = args.gamma * l1_loss + args.alpha * gms_loss + args.belta * ssim_loss
            # loss = args.gamma * l1_loss
            losses.update(loss.item(), data.size(0))
    print_log(('Valid Epoch: {} loss: {:.6f}'.format(epoch, losses.avg)), log)

    return losses.avg

def save_snapshot(args, x, x2, model, save_dir, save_dir2, log):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        ### NOTE: MUST MODIFY scale factor generalized way and float for basicsr
        #downsampled = bilinear_sampling(x,scale_factor=1/args.scale_factor)
        recon = model(bilinear_sampling(x,scale_factor=1/args.scale_factor).float())
        
        
        #import pdb; pdb.set_trace()
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(denorm(x_concat.data.cpu()), save_dir, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir)), log)

        x_fake_list = x2
        recon = model(bilinear_sampling(x2,scale_factor=1/args.scale_factor).float())
        x_concat = torch.cat((x_fake_list, recon), dim=3)
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

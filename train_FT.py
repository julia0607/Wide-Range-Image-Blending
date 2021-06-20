import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor
import os
from os.path import join
from models.Generator import Generator
from utils.loss import IDMRFLoss
from models.Discriminator import Discriminator
from utils.utils import gaussian_weight
from tensorboardX import SummaryWriter
from datasets.dataset import dataset_recon, dataset_diff, dataset_complete
import argparse
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Training
def train(gen, dis, opt_gen, opt_dis, epoch, train_loader, train_diff_loader, train_rand_loader, writer):
    
    gen.train()
    dis.train()

    mse = nn.MSELoss(reduction = 'none').cuda(0)
    mrf = IDMRFLoss(device=1)

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_feat_cons_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    
    iter_train_diff_loader = iter(train_diff_loader)
    
    for batch_idx, (I_l, I_r, I_m) in enumerate(train_loader):
        
        if batch_idx % 10 == 0:
            print("train iter %d" %batch_idx)
            
        batchSize_rec = I_l.shape[0]
        imgSize = I_l.shape[2]
        
        (I1, I2) = next(iter_train_diff_loader) # I1 and I2 are inputs from different images
            
        I_rand = next(iter(train_rand_loader))
        
        I_l = torch.cat((I_l, I1), dim=0)
        I_r = torch.cat((I_r, I2), dim=0)

        batchSize = I_l.shape[0]
            
        I_l, I_r, I_m, I_rand = Variable(I_l).cuda(0), Variable(I_r).cuda(0), Variable(I_m).cuda(0), Variable(I_rand).cuda(0)
        
        # Generate Image
        I_pred, f_m, F_l, F_r = gen(I_l, I_r)
        f_m_gt = gen(I_m, only_encode=True) # gt for feature map of middle part
        I_pred_split = list(torch.split(I_pred, imgSize, dim=3))
        I_gt = torch.cat((I_l[:batchSize_rec],I_m,I_r[:batchSize_rec]),3)
        I_real = torch.cat((I_gt, I_rand),0)
        
        # Discriminator
        fake = dis(I_pred)
        real= dis(I_real)
        
        ## Compute losses        
        # Pixel Reconstruction Loss
        weight = gaussian_weight(batchSize_rec, imgSize, device=0)
        mask = weight + weight.flip(3)
        pixel_rec_loss = (mse(I_pred_split[0], I_l) + mse(I_pred_split[2], I_r)).mean() * batchSize + (mask * mse(I_pred_split[1][:batchSize_rec], I_m)).mean() * batchSize_rec
             
        # Texture Consistency Loss (IDMRF Loss)
        mrf_loss = mrf((I_pred_split[1][:batchSize_rec].cuda(1)+1)/2.0, (I_m.cuda(1)+1)/2.0) * 0.01
        
        # Feature Reconstruction Loss
        feat_rec_loss = mse(f_m[:batchSize_rec], f_m_gt.detach()).mean() * batchSize_rec
        
        # Feature Consistency Loss
        feat_cons_loss = (mse(F_l[0], F_r[0]) + mse(F_l[1], F_r[1]) + mse(F_l[2], F_r[2])).mean() * batchSize        
        
        # RaLSGAN Adversarial Loss
        real_label = torch.ones(batchSize,1).cuda(0)
        fake_label = torch.zeros(batchSize,1).cuda(0)
        gen_adv_loss = ((fake - real.mean(0, keepdim=True) - fake_label) ** 2).mean() * batchSize * 0.002 * 0.9
        dis_adv_loss = (((real - fake.mean(0, keepdim=True) - real_label) ** 2).mean() + ((fake - real.mean(0, keepdim=True) + real_label) ** 2).mean()) * batchSize
        
        gen_loss = pixel_rec_loss + mrf_loss.cuda(0) + feat_rec_loss + feat_cons_loss + gen_adv_loss
        dis_loss = dis_adv_loss
        acc_pixel_rec_loss += pixel_rec_loss.data
        acc_mrf_loss += mrf_loss.data
        acc_feat_rec_loss += feat_rec_loss.data
        acc_feat_cons_loss += feat_cons_loss.data
        acc_gen_adv_loss += gen_adv_loss.data
        acc_dis_adv_loss += dis_adv_loss.data
        
        ## Update Generator
        if (batch_idx % 3) != 0:
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()
        
        ## Update Discriminator
        if (batch_idx % 3) == 0:
            opt_dis.zero_grad()
            dis_loss.backward()
            opt_dis.step()        
    
    ## Tensor board
    writer.add_scalars('train/generator_loss', {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Feature Reconstruction Loss': acc_feat_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Feature Consistency Loss': acc_feat_cons_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(train_loader.dataset)}, epoch)

def test(gen, dis, epoch, test_loader, test_diff_loader, test_rand_loader, writer):

    gen.eval()
    dis.eval()
    
    mse = nn.MSELoss(reduction = 'none').cuda(0)
    mrf = IDMRFLoss(device=1)
    
    acc_pixel_rec_loss = 0
    acc_mrf_loss = 0
    acc_feat_rec_loss = 0
    acc_feat_cons_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    
    iter_test_diff_loader = iter(test_diff_loader)
    
    for batch_idx, (I_l, I_r, I_m) in enumerate(test_loader):
        
        if batch_idx % 10 == 0:
            print("test iter %d" %batch_idx)
            
        batchSize_rec = I_l.shape[0]
        imgSize = I_l.shape[2]
        
        I1, I2 = next(iter_test_diff_loader) # I1 and I2 are inputs from different images
            
        I_rand = next(iter(test_rand_loader))
        
        I_l = torch.cat((I_l, I1), dim=0)
        I_r = torch.cat((I_r, I2), dim=0)

        batchSize = I_l.shape[0]
            
        I_l, I_r, I_m, I_rand = Variable(I_l).cuda(0), Variable(I_r).cuda(0), Variable(I_m).cuda(0), Variable(I_rand).cuda(0)
        
        # Generate Image
        with torch.no_grad():
            I_pred, f_m, F_l, F_r = gen(I_l, I_r)
        with torch.no_grad():
            f_m_gt = gen(I_m, only_encode=True) # gt for feature map of middle part
        I_pred_split = list(torch.split(I_pred, imgSize, dim=3))
        I_gt = torch.cat((I_l[:batchSize_rec],I_m,I_r[:batchSize_rec]),3)
        I_real = torch.cat((I_gt, I_rand),0)
        
        # Discriminator
        with torch.no_grad():
            fake = dis(I_pred)
            real = dis(I_real)

        ## Compute losses        
        # Pixel Reconstruction Loss
        with torch.no_grad():
            weight = gaussian_weight(batchSize_rec, imgSize, device=0)
            mask = weight + weight.flip(3)
            pixel_rec_loss = (mse(I_pred_split[0], I_l) + mse(I_pred_split[2], I_r)).mean() * batchSize + (mask * mse(I_pred_split[1][:batchSize_rec], I_m)).mean() * batchSize_rec
             
        # Texture Consistency Loss (IDMRF Loss)
        with torch.no_grad():
            mrf_loss = mrf((I_pred_split[1][:batchSize_rec].cuda(1)+1)/2.0, (I_m.cuda(1)+1)/2.0) * 0.01
        
        # Feature Reconstruction Loss
        with torch.no_grad():
            feat_rec_loss = mse(f_m[:batchSize_rec], f_m_gt.detach()).mean() * batchSize_rec
        
        # Feature Consistency Loss
        with torch.no_grad():
            feat_cons_loss = (mse(F_l[0], F_r[0]) + mse(F_l[1], F_r[1]) + mse(F_l[2], F_r[2])).mean() * batchSize        

        # RaLSGAN Adversarial Loss
        real_label = torch.ones(batchSize,1).cuda(0)
        fake_label = torch.zeros(batchSize,1).cuda(0)
        with torch.no_grad():
            gen_adv_loss = ((fake - real.mean(0, keepdim=True) - fake_label) ** 2).mean() * batchSize * 0.002 * 0.9
            dis_adv_loss = (((real - fake.mean(0, keepdim=True) - real_label) ** 2).mean() + ((fake - real.mean(0, keepdim=True) + real_label) ** 2).mean()) * batchSize
        
        acc_pixel_rec_loss += pixel_rec_loss.data
        acc_mrf_loss += mrf_loss.data
        acc_feat_rec_loss += feat_rec_loss.data
        acc_feat_cons_loss += feat_cons_loss.data
        acc_gen_adv_loss += gen_adv_loss.data
        acc_dis_adv_loss += dis_adv_loss.data
    
    ## Tensor board
    writer.add_scalars('test/generator_loss', {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(test_loader.dataset)}, epoch)
    writer.add_scalars('test/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(test_loader.dataset)}, epoch)
    writer.add_scalars('test/generator_loss', {'Feature Reconstruction Loss': acc_feat_rec_loss / len(test_loader.dataset)}, epoch)
    writer.add_scalars('test/generator_loss', {'Feature Consistency Loss': acc_feat_cons_loss / len(test_loader.dataset)}, epoch)
    writer.add_scalars('test/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(test_loader.dataset)}, epoch)
    writer.add_scalars('test/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(test_loader.dataset)}, epoch)
        
if __name__ == '__main__':

    LOAD_WEIGHT_DIR = './weights/'
    SAVE_WEIGHT_DIR = './checkpoints/FT_Stage/'
    SAVE_LOG_DIR = './logs/'
    TRAIN_DATA_DIR = './data/scenery6000_split/train/'
    TEST_DATA_DIR = './data/scenery6000_split/test/'
    TRAIN_METRIC_PATH = './data/scenery6000_split/selected_records/train_LPIPS_256x256.pt'
    TEST_METRIC_PATH = './data/scenery6000_split/selected_records/test_LPIPS_256x256.pt'
    
    def get_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('--train_batch_size',type=int,help='batch size of training data',default=28)
        parser.add_argument('--test_batch_size',type=int,help='batch size of testing data',default=28)
        parser.add_argument('--epochs',type=int,help='number of epoches',default=200)
        parser.add_argument('--lr',type=float,help='learning rate',default=2e-3)
        parser.add_argument('--alpha',type=float,help='learning rate decay for discriminator',default=0.1)
        parser.add_argument('--decay_step',type=int,help='decay step for learning rate of both generator and discriminator',default=50)
        parser.add_argument('--gamma',type=float,help='decay rate for learning rate of both generator and discriminator',default=0.5)
        parser.add_argument('--load_pretrain',type=bool,help='load pretrain weight',default=False)
        parser.add_argument('--rand_pair',type=bool,help='pair training/testing data randomly',default=False)
        parser.add_argument('--test_flag',type=bool,help='testing while training', default=False)

        parser.add_argument('--skip_connection', type=int,help='layers with skip connection', nargs='+', default=[0,1,2,3,4])
        parser.add_argument('--attention', type=int,help='layers with attention mechanism applied on skip connection', nargs='+', default=[1])
        parser.add_argument('--select_k',type=int,help='number of images selected to pair with each image in training/testing data',default=3)

        parser.add_argument('--load_weight_dir',type=str,help='directory of pretrain model weights',default=LOAD_WEIGHT_DIR)
        parser.add_argument('--save_weight_dir',type=str,help='directory of saving model weights',default=SAVE_WEIGHT_DIR)
        parser.add_argument('--log_dir',type=str,help='directory of saving logs',default=SAVE_LOG_DIR)
        parser.add_argument('--train_data_dir',type=str,help='directory of training data',default=TRAIN_DATA_DIR)
        parser.add_argument('--test_data_dir',type=str,help='directory of testing data',default=TEST_DATA_DIR)
        parser.add_argument('--train_metric_path',type=str,help='file path of metric for pairing training data',default=TRAIN_METRIC_PATH)
        parser.add_argument('--test_metric_path',type=str,help='file path of metric for pairing testing data',default=TEST_METRIC_PATH)

        opts = parser.parse_args()
        return opts

    args = get_args()
    os.makedirs(args.save_weight_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(join(args.log_dir, 'FT_Stage_%s'%datetime.now().strftime("%Y%m%d-%H%M%S"))) 
    
    # Initialize the model
    print('Initializing model...')
    gen = Generator(skip=args.skip_connection, attention=args.attention).cuda(0)
    dis = Discriminator().cuda(0)
    
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=2e-5)
    opt_dis = optim.Adam(dis.parameters(), lr=args.lr*args.alpha, betas=(0.5, 0.9), weight_decay=2e-5)    

    scheduler_gen = StepLR(opt_gen, step_size=args.decay_step, gamma=args.gamma)
    scheduler_dis = StepLR(opt_dis, step_size=args.decay_step, gamma=args.gamma)

    # Load pre-trained weight
    if args.load_pretrain:
        print('Loading model weight...')
        gen.load_state_dict(torch.load(join(args.load_weight_dir, 'Gen')))
        dis.load_state_dict(torch.load(join(args.load_weight_dir, 'Dis')))
    
    # Load data
    print('Loading data...')
    transformations = transforms.Compose([ToTensor()])

    train_data = dataset_recon(root=args.train_data_dir, transforms=transformations, crop='rand', imgSize=256)
    if args.rand_pair:
        train_data_diff = dataset_diff(root=args.train_data_dir, transforms=transformations, crop='rand', imgSize=256, select_root=None, select_k=args.select_k)
    else:
        train_data_diff = dataset_diff(root=args.train_data_dir, transforms=transformations, crop='center', imgSize=256, select_root=args.train_metric_path, select_k=args.select_k)
    train_data_rand = dataset_complete(root=args.train_data_dir, transforms=transformations, crop='rand', imgSize=256, width=3)
    train_loader = DataLoader(train_data, batch_size=int(args.train_batch_size/2), shuffle=True)
    train_diff_loader = DataLoader(train_data_diff, batch_size=int(args.train_batch_size/2), shuffle=True)
    train_rand_loader = DataLoader(train_data_rand, batch_size=int(args.train_batch_size/2), shuffle=True)
    print('train data: %d images, %d pairs'%(len(train_loader.dataset), len(train_diff_loader.dataset)))
    
    if args.test_flag:
        test_data = dataset_recon(root=args.test_data_dir, transforms=transformations, crop='center', imgSize=256)
        if args.rand_pair:
            test_data_diff = dataset_diff(root=args.test_data_dir, transforms=transformations, crop='center', imgSize=256, select_root=None, select_k=args.select_k)
        else:
            test_data_diff = dataset_diff(root=args.test_data_dir, transforms=transformations, crop='center', imgSize=256, select_root=args.test_metric_path, select_k=args.select_k)
        test_data_rand = dataset_complete(root=args.test_data_dir, transforms=transformations, crop='rand', imgSize=256, width=3)
        test_loader = DataLoader(test_data, batch_size=int(args.test_batch_size/2), shuffle=False)
        test_diff_loader = DataLoader(test_data_diff, batch_size=int(args.test_batch_size/2), shuffle=False)
        test_rand_loader = DataLoader(test_data_rand, batch_size=int(args.test_batch_size/2), shuffle=True)
        print('test data: %d images, %d pairs'%(len(test_loader.dataset), len(test_diff_loader.dataset)))

    # Train & test the model
    for epoch in range(1, 1 + args.epochs):        
        scheduler_gen.step(epoch)
        scheduler_dis.step(epoch)
        print("----Start training[%d]----" %epoch)
        train(gen, dis, opt_gen, opt_dis, epoch, train_loader, train_diff_loader, train_rand_loader, writer)
        if args.test_flag:
            print("----Start testing[%d]----" %epoch)
            test(gen, dis, epoch, test_loader, test_diff_loader, test_rand_loader, writer)
        
        # Save the model weight
        torch.save(gen.state_dict(), join(args.save_weight_dir, 'Gen_%d'%epoch))
        torch.save(dis.state_dict(), join(args.save_weight_dir, 'Dis_%d'%epoch))

    writer.close()
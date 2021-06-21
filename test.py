import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
import os
from os.path import join
from models.Generator import Generator
from datasets.dataset import dataset_test
import argparse
import skimage

# Evaluate
def evaluate(gen, eval_loader, rand_pair, save_dir):

    gen.eval()
    
    os.makedirs(join(save_dir, 'result1-2'), exist_ok=True)
    os.makedirs(join(save_dir, 'result2-1'), exist_ok=True)
    if rand_pair:
        os.makedirs(join(save_dir, 'input1'), exist_ok=True)
        os.makedirs(join(save_dir, 'input2'), exist_ok=True)

    for batch_idx, (I1, I2, name) in enumerate(eval_loader):

        imgSize = I1.shape[2]
        
        I1, I2 = Variable(I1).cuda(), Variable(I2).cuda()
        with torch.no_grad():
            I_pred_1_2, _, _, _ = gen(I1, I2)
            I_pred_2_1, _, _, _ = gen(I2, I1)

        I_pred_1_2 = np.transpose(I_pred_1_2[0].data.cpu().numpy(), (1,2,0))
        I_pred_2_1 = np.transpose(I_pred_2_1[0].data.cpu().numpy(), (1,2,0))

        skimage.io.imsave(join(save_dir, 'result1-2', '%s.png'%(name[0])), skimage.img_as_ubyte(I_pred_1_2))
        skimage.io.imsave(join(save_dir, 'result2-1', '%s.png'%(name[0])), skimage.img_as_ubyte(I_pred_2_1))

        if rand_pair:
            I1 = np.transpose(I1[0].data.cpu().numpy(), (1,2,0))
            I2 = np.transpose(I2[0].data.cpu().numpy(), (1,2,0))

            skimage.io.imsave(join(save_dir, 'input1', '%s.png'%(name[0])), skimage.img_as_ubyte(I1))
            skimage.io.imsave(join(save_dir, 'input2', '%s.png'%(name[0])), skimage.img_as_ubyte(I2))
        
if __name__ == '__main__':

    LOAD_WEIGHT_DIR = './weights/'
    TEST_DATA_DIR_1 = './samples/input1/'
    TEST_DATA_DIR_2 = './samples/input2/'
    SAVE_DIR = './results/'
    
    def get_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('--rand_pair',type=bool,help='pair testing data randomly',default=False)

        parser.add_argument('--skip_connection', type=int,help='layers with skip connection', nargs='+', default=[0,1,2,3,4])
        parser.add_argument('--attention', type=int,help='layers with attention mechanism applied on skip connection', nargs='+', default=[1])

        parser.add_argument('--load_weight_dir',type=str,help='directory of pretrain model weights',default=LOAD_WEIGHT_DIR)
        parser.add_argument('--save_dir',type=str,help='directory of saving results',default=SAVE_DIR)
        parser.add_argument('--test_data_dir_1',type=str,help='directory of testing data 1',default=TEST_DATA_DIR_1)
        parser.add_argument('--test_data_dir_2',type=str,help='directory of testing data 2',default=TEST_DATA_DIR_2)

        opts = parser.parse_args()
        return opts

    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize the model
    print('Initializing model...')
    gen = Generator(skip=args.skip_connection, attention=args.attention).cuda()
    
    # Load pre-trained weight
    print('Loading model weight...')
    gen.load_state_dict(torch.load(join(args.load_weight_dir, 'Gen')))
    
    # Load data
    print('Loading data...')
    if args.rand_pair:
        transformations = transforms.Compose([ToTensor()])
        eval_data = dataset_test(root1=args.test_data_dir_1, root2=args.test_data_dir_2, transforms=transformations, crop='rand', rand_pair=True, imgSize=256)
    else:
        transformations = transforms.Compose([Resize((256,256)), ToTensor()])
        eval_data = dataset_test(root1=args.test_data_dir_1, root2=args.test_data_dir_2, transforms=transformations, crop='none', rand_pair=False)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)
    print('test data: %d image pairs'%(len(eval_loader.dataset)))

    # Evaluate
    evaluate(gen, eval_loader, args.rand_pair, args.save_dir)
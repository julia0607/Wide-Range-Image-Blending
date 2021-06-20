import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
from utils.utils import *
      

def make_layers(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if norm:
        layer.append(nn.InstanceNorm2d(out_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)  

def make_layers_transpose(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if norm:
        layer.append(nn.InstanceNorm2d(out_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)

class identity_block(nn.Module):
    def __init__(self, channels, norm=True, is_relu=False):
        super(identity_block, self).__init__()
        
        self.conv1 = make_layers(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=3, stride=1, padding=1, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv3 = make_layers(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=False)
        self.output = nn.ReLU() if is_relu else nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut
        x = self.output(x)
        return x  
    
class convolutional_block(nn.Module):
    def __init__(self, channels, norm=True, is_relu=False):
        super(convolutional_block, self).__init__()
        
        self.conv1 = make_layers(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv3 = make_layers(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=False)
        self.shortcut_path = make_layers(channels[0], channels[3], kernel_size=1, stride=2, padding=0, bias=False, norm=norm, activation=False)
        self.output = nn.ReLU() if is_relu else nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.shortcut_path(shortcut)
        x = x + shortcut
        x = self.output(x)
        return x    
    
class SHC(nn.Module):
    def __init__(self, channel, norm=True):
        super(SHC, self).__init__()

        self.conv1 = make_layers(channel*2, int(channel/2), kernel_size=1, stride=1, padding=0, norm=norm, activation=True, is_relu=True)
        self.conv2 = make_layers(int(channel/2), int(channel/2), kernel_size=3, stride=1, padding=1, norm=norm, activation=True, is_relu=True)
        self.conv3 = make_layers(int(channel/2), channel, kernel_size=1, stride=1, padding=0, norm=norm, activation=False)

    def forward(self, x, shortcut):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut
        return x
    
class GRB(nn.Module):
    def __init__(self, channel, dilation, norm=True):
        super(GRB, self).__init__()

        self.path1 = nn.Sequential(
            make_layers(channel, channel, kernel_size=(3,1), stride=1, padding=(dilation,0), dilation=dilation, norm=norm, activation=True, is_relu=True),
            make_layers(channel, channel, kernel_size=(1,7), stride=1, padding=(0,3*dilation), dilation=dilation, norm=norm, activation=False)
        )
        self.path2 = nn.Sequential(
            make_layers(channel, channel, kernel_size=(1,7), stride=1, padding=(0,3*dilation), dilation=dilation, norm=norm, activation=True, is_relu=True),
            make_layers(channel, channel, kernel_size=(3,1), stride=1, padding=(dilation,0), dilation=dilation, norm=norm, activation=False)
        )
        self.output = nn.ReLU()

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = x + x1 + x2
        x = self.output(x)
        return x

class LSTM_Encoder_Decoder(nn.Module):
    def __init__(self, size=[512,4,4], split=4, pred_step=4, device=0):
        # --PARAMS--
        # size: size for each input feature map [channel, height, width]
        # split: # of portions each input feature map should be split into
        # pred_step: # of steps for the prediction of the intermediate region 
        #            (the width of the resultant intermediate region will be the width of input portion * pred_step)

        super(LSTM_Encoder_Decoder, self).__init__()

        self.channel, self.height, self.width = size
        self.lstm_size = self.channel * self.height * int(self.width/split)
        self.LSTM_encoder = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.LSTM_decoder = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.output = make_layers(2*self.channel, self.channel, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)
        self.split = split
        self.pred_step = pred_step
        self.device = device

    def forward(self, x1, x2):
        init_hidden = (Variable(torch.zeros(2, x1.shape[0], self.lstm_size)).cuda(self.device), Variable(torch.zeros(2,x1.shape[0], self.lstm_size)).cuda(self.device))
        x1_out = x1
        x2_out = x2
        x1_split = torch.stack(torch.split(x1, int(self.width/self.split), dim=3)).view(self.split, -1, 1, self.lstm_size)
        x1_split_reversed = torch.stack(torch.split(x1, int(self.width/self.split), dim=3)).flip(dims=[4]).view(self.split, -1, 1, self.lstm_size)
        x2_split = torch.stack(torch.split(x2, int(self.width/self.split), dim=3)).view(self.split, -1, 1, self.lstm_size)
        x2_split_reversed = torch.stack(torch.split(x2, int(self.width/self.split), dim=3)).flip(dims=[4]).view(self.split, -1, 1, self.lstm_size)
        
        # Encode feature from x2 (left->right)
        en_hidden = init_hidden
        for i in range(self.split):
            en_out, en_hidden = self.LSTM_encoder(x2_split[i], en_hidden)
        hidden_x2 = en_hidden

        # Encode feature from x1 (right->left)
        en_hidden = init_hidden
        for i in reversed(range(self.split)):
            en_out, en_hidden = self.LSTM_encoder(x1_split_reversed[i], en_hidden)
        hidden_x1_reversed = en_hidden
        
        # Decode feature from x1 (left->right)
        de_hidden = hidden_x2
        for i in range(self.split):
            de_out, de_hidden = self.LSTM_decoder(x1_split[i], de_hidden) # f_1^2 ~ f_1^5
        x1_out = torch.cat((x1_out, de_out.view(-1, self.channel, self.height, int(self.width/self.split))), 3)
        for i in range(self.pred_step + self.split - 1):
            de_out, de_hidden = self.LSTM_decoder(de_out, de_hidden) # f_1^6 ~ f_1^12
            x1_out = torch.cat((x1_out, de_out.view(-1, self.channel, self.height, int(self.width/self.split))), 3)
        
        # Decode feature from x2 (right->left)
        de_hidden = hidden_x1_reversed
        for i in reversed(range(self.split)):
            de_out, de_hidden = self.LSTM_decoder(x2_split_reversed[i], de_hidden) # f_2^11' ~ f_2^8'
        x2_out = torch.cat((de_out.view(-1, self.channel, self.height, int(self.width/self.split)).flip(dims=[3]), x2_out), 3)
        for i in range(self.pred_step + self.split - 1):
            de_out, de_hidden = self.LSTM_decoder(de_out, de_hidden) # f_2^7' ~ # f_2^1'
            x2_out = torch.cat((de_out.view(-1, self.channel, self.height, int(self.width/self.split)).flip(dims=[3]), x2_out), 3)

        x1_out = (x1_out[:,:,:,:self.width], x1_out[:,:,:,self.width:-self.width], x1_out[:,:,:,-self.width:])
        x2_out = (x2_out[:,:,:,:self.width], x2_out[:,:,:,self.width:-self.width], x2_out[:,:,:,-self.width:])
        out = self.output(torch.cat((x1_out[1], x2_out[1]),1))
        
        return out, x1_out, x2_out
    
class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10, fuse=False, two_input=True, weight_func='cos', use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.two_input = two_input
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        if weight_func == 'cos':
            self.weight_func = cos_function_weight
        elif weight_func == 'gaussian':
            self.weight_func = gaussian_weight

    def forward(self, left, right, mid, shortcut, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            mid: Input feature to match (foreground).
            left & right: Input feature for match (background).
            shortcut: [0] for left, [1] for right, shortcut[0] = [LH, HL, HH, ...]
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        
        if self.two_input == False:
            left = torch.cat((left,right),3)
            for i in range(len(shortcut[0])):
                shortcut[0][i] = torch.cat((shortcut[0][i], shortcut[1][i]),3)
        
        # get shapes
        raw_int_ls = list(shortcut[0][0].size())   # b*c*h*w
        raw_int_ms = list(shortcut[1][0].size())   # b*c*h*w
        if self.two_input:
            raw_int_rs = list(shortcut[1][0].size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_l = [extract_image_patches(shortcut[0][i], ksizes=[kernel, kernel], strides=[self.rate*self.stride, self.rate*self.stride], rates=[1, 1], padding='same') for i in range(len(shortcut[0]))] # [N, C*k*k, L]
        if self.two_input:
            raw_r = [extract_image_patches(shortcut[1][i], ksizes=[kernel, kernel], strides=[self.rate*self.stride, self.rate*self.stride], rates=[1, 1], padding='same') for i in range(len(shortcut[1]))] # [N, C*k*k, L]
        
        # raw_shape: [N, C, k, k, L]
        raw_l = [raw_l[i].view(raw_int_ls[0], raw_int_ls[1], kernel, kernel, -1) for i in range(len(raw_l))]
        raw_l = [raw_l[i].permute(0, 4, 1, 2, 3) for i in range(len(raw_l))]    # raw_shape: [N, L, C, k, k]
        raw_l_groups = [torch.split(raw_l[i], 1, dim=0) for i in range(len(raw_l))]
        if self.two_input:
            raw_r = [raw_r[i].view(raw_int_rs[0], raw_int_rs[1], kernel, kernel, -1) for i in range(len(raw_r))]
            raw_r = [raw_r[i].permute(0, 4, 1, 2, 3) for i in range(len(raw_r))]    # raw_shape: [N, L, C, k, k]
            raw_r_groups = [torch.split(raw_r[i], 1, dim=0) for i in range(len(raw_r))]

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        left = F.interpolate(left, scale_factor=1./self.rate, mode='nearest')
        if self.two_input:
            right = F.interpolate(right, scale_factor=1./self.rate, mode='nearest')
        mid = F.interpolate(mid, scale_factor=1./self.rate, mode='nearest')
        int_ls = list(left.size())     # b*c*h*w
        if self.two_input:
            int_rs = list(right.size())
        int_mids = list(mid.size())
        mid_groups = torch.split(mid, 1, dim=0)  # split tensors along the batch dimension
        
        # w shape: [N, C*k*k, L]
        left = extract_image_patches(left, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        # w shape: [N, C, k, k, L]
        left = left.view(int_ls[0], int_ls[1], self.ksize, self.ksize, -1)
        left = left.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        l_groups = torch.split(left, 1, dim=0)
        if self.two_input:
            right = extract_image_patches(right, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
            # w shape: [N, C, k, k, L]
            right = right.view(int_rs[0], int_rs[1], self.ksize, self.ksize, -1)
            right = right.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
            r_groups = torch.split(right, 1, dim=0)

        batch = [i for i in range(raw_int_ls[0])]
        y_l = [[] for i in range(len(shortcut[0]))]
        y_r = [[] for i in range(len(shortcut[0]))]
        y = [[] for i in range(len(shortcut[0]))]

        weight = self.weight_func(raw_int_ls[0], raw_int_ls[2], device=self.device_ids)
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda(self.device_ids)
        if self.two_input == False:
            r_groups = l_groups

        for xi, li, ri, batch_idx in zip(mid_groups, l_groups, r_groups, batch):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda(self.device_ids)
            li = li[0]  # [L, C, k, k]
            max_li = torch.sqrt(reduce_sum(torch.pow(li, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            li_normed = li / max_li
            if self.two_input:
                ri = ri[0]  # [L, C, k, k]
                max_ri = torch.sqrt(reduce_sum(torch.pow(ri, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
                ri_normed = ri / max_ri

            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = []
            yi.append(F.conv2d(xi, li_normed, stride=1)) # left   # [1, L, H, W]
            if self.two_input:
                yi.append(F.conv2d(xi, ri_normed, stride=1)) # right  # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                for i in range(len(yi)):
                    yi[i] = yi[i].view(1, 1, int_ls[2]*int_ls[3], int_mids[2]*int_mids[3])  # (B=1, I=1, H=32*32, W=32*32)
                    yi[i] = same_padding(yi[i], [k, k], [1, 1], [1, 1])
                    yi[i] = F.conv2d(yi[i], fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                    yi[i] = yi[i].contiguous().view(1, int_ls[2], int_ls[3], int_mids[2], int_mids[3])  # (B=1, 32, 32, 32, 32)
                    yi[i] = yi[i].permute(0, 2, 1, 4, 3)
                    yi[i] = yi[i].contiguous().view(1, 1, int_ls[2]*int_ls[3], int_mids[2]*int_mids[3])
                    yi[i] = same_padding(yi[i], [k, k], [1, 1], [1, 1])
                    yi[i] = F.conv2d(yi[i], fuse_weight, stride=1)
                    yi[i] = yi[i].contiguous().view(1, int_ls[3], int_ls[2], int_mids[3], int_mids[2])
                    yi[i] = yi[i].permute(0, 2, 1, 4, 3).contiguous()
            yi = [yi[i].view(1, int_mids[2] * int_ls[3], int_mids[2], int_mids[3]) for i in range(len(yi))]  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = [F.softmax(yi[i]*scale, dim=1) for i in range(len(yi))]

            # deconv for patch pasting
            for i in range(len(shortcut[0])):
                li_center = raw_l_groups[i][batch_idx][0]
                if self.two_input:
                    ri_center = raw_r_groups[i][batch_idx][0]
                # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
                y_l[i].append(F.conv_transpose2d(yi[0], li_center, stride=self.rate, padding=1) / 4.) # (B=1, C=128, H=64, W=64)
                if self.two_input:
                    y_r[i].append(F.conv_transpose2d(yi[1], ri_center, stride=self.rate, padding=1) / 4.) # (B=1, C=128, H=64, W=64)

        for i in range(len(shortcut[0])):
            y_l[i] = torch.cat(y_l[i], dim=0).contiguous().view(raw_int_ms)  # back to the mini-batch
            if self.two_input:
                y_r[i] = torch.cat(y_r[i], dim=0).contiguous().view(raw_int_ms)
                y[i] = weight * y_l[i] + weight.flip(3) * y_r[i]
            else:
                y[i] = y_l[i]

        return y
    
class Generator(nn.Module):
    def __init__(self, device=0, skip=[0,1,2,3,4], attention=[0,1,2,3,4]):
        # --PARAMS--
        # skip: indicates the layers with skip connection
        # attention: indicates skip connections where the attention mechanism is applied

        super(Generator, self).__init__()

        # input size: 256x256x3
        self.skip = skip
        self.attention = attention
        self.CA = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True, two_input=False, use_cuda=True, device_ids=device)
        
        ## Stage 1
        # 256*256*3
        self.encoder_stage1_conv1 = make_layers(3, 64, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=False)

        # 128*128*64
        self.encoder_stage1_conv2 = make_layers(64, 128, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=False)
        
        ## Stage 2
        # 64*64*128
        self.encoder_stage2 = nn.Sequential(
            convolutional_block([128, 64, 64, 256], norm=False),
            identity_block([256, 64, 64, 256], norm=False),
            identity_block([256, 64, 64, 256], norm=False)
        )
        
        ## Stage 3
        # 32*32*256
        self.encoder_stage3 = nn.Sequential(
            convolutional_block([256, 128, 128, 512]),
            identity_block([512, 128, 128, 512]),
            identity_block([512, 128, 128, 512]),
            identity_block([512, 128, 128, 512])
        )
        
        ## Stage 4
        # 16*16*512
        self.encoder_stage4 = nn.Sequential(
            convolutional_block([512, 256, 256, 1024]),
            identity_block([1024, 256, 256, 1024]),
            identity_block([1024, 256, 256, 1024]),
            identity_block([1024, 256, 256, 1024])
        )
        
        ## Stage 5
        # 8*8*1024
        self.encoder_stage5 = nn.Sequential(
            convolutional_block([1024, 512, 512, 1024]),
            identity_block([1024, 512, 512, 1024]),
            identity_block([1024, 512, 512, 1024]),
            identity_block([1024, 512, 512, 1024]),
            identity_block([1024, 512, 512, 1024])
        )
        
        # 4*4*1024
        self.feature_in = make_layers(1024, 512, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)
        
        # 4*4*512
        self.BCT = LSTM_Encoder_Decoder(device=device)
        
        # 4*4*512
        self.feature_out = make_layers(512, 1024, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)
        
        ## Stage -5
        # 4*12*1024
        self.GRB5 = GRB(1024,1)
        self.decoder_stage5 = nn.Sequential(
            identity_block([1024, 512, 512, 1024], is_relu=True),
            identity_block([1024, 512, 512, 1024], is_relu=True),
            make_layers_transpose(1024, 1024, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )
        
        ## Stage -4
        # 8*24*1024
        self.SHC4 = SHC(1024)
        if 4 in self.skip:
            self.SHC4_mid = SHC(1024)
        self.skip4 = nn.Sequential(
            nn.InstanceNorm2d(1024, affine=True),
            nn.ReLU()
        )
        self.GRB4 = GRB(1024,2)
        self.decoder_stage4 = nn.Sequential(
            identity_block([1024, 256, 256, 1024], is_relu=True),
            identity_block([1024, 256, 256, 1024], is_relu=True),
            identity_block([1024, 256, 256, 1024], is_relu=True),
            make_layers_transpose(1024, 512, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )
        
        ## Stage -3
        # 16*48*512
        self.SHC3 = SHC(512)
        if 3 in self.skip:
            self.SHC3_mid = SHC(512)
        self.skip3 = nn.Sequential(
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU()
        )
        self.GRB3 = GRB(512,4)
        self.decoder_stage3 = nn.Sequential(
            identity_block([512, 128, 128, 512], is_relu=True),
            identity_block([512, 128, 128, 512], is_relu=True),
            identity_block([512, 128, 128, 512], is_relu=True),
            make_layers_transpose(512, 256, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )
        
        ## Stage -2
        # 32*96*256
        self.SHC2 = SHC(256, norm=False)
        if 2 in self.skip:
            self.SHC2_mid = SHC(256, norm=False)
        self.skip2 = nn.ReLU()
        self.GRB2 = GRB(256, 4, norm=False)
        self.decoder_stage2 = nn.Sequential(
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            make_layers_transpose(256, 128, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=True)
        )
        
        ## Stage -1
        # 64*192*128
        self.SHC1 = SHC(128, norm=False)
        if 1 in self.skip:
            self.SHC1_mid = SHC(128, norm=False)
        self.skip1 = nn.ReLU()
        self.decoder_stage1 = make_layers_transpose(128, 64, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=True)
        
        ## Stage -0
        # 128*384*64
        self.SHC0 = SHC(64, norm=False)
        if 0 in self.skip:
            self.SHC0_mid = SHC(64, norm=False)
        self.skip0 = nn.ReLU()
        self.decoder_stage0 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        # 256*768*3
        
    def encode(self, x):
        shortcut = []
        x = self.encoder_stage1_conv1(x)
        shortcut.append(x)
        x = self.encoder_stage1_conv2(x)
        shortcut.append(x)
        x = self.encoder_stage2(x)
        shortcut.append(x)
        x = self.encoder_stage3(x)
        shortcut.append(x)
        x = self.encoder_stage4(x)
        shortcut.append(x)
        x = self.encoder_stage5(x)
        shortcut.append(x)
        x = self.feature_in(x)
        
        return x, shortcut
    
    def decode(self, x, shortcut):

        out = self.GRB5(x)
        out = self.decoder_stage5(out)
        
        if 4 in self.skip:
            out = list(torch.split(out, 8, dim=3))
            if (4 in self.attention): 
                sc_l = [shortcut[4][0]]
                sc_r = [shortcut[4][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip4(self.SHC4_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip4(self.SHC4(torch.cat((out[0],shortcut[4][0]),1), shortcut[4][0]))
            out[2] = self.skip4(self.SHC4(torch.cat((out[2],shortcut[4][1]),1), shortcut[4][1]))
            out = torch.cat((out),3)
            out = self.GRB4(out)
        out = self.decoder_stage4(out)
        
        if 3 in self.skip:
            out = list(torch.split(out, 16, dim=3))
            if (3 in self.attention): 
                sc_l = [shortcut[3][0]]
                sc_r = [shortcut[3][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip3(self.SHC3_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip3(self.SHC3(torch.cat((out[0],shortcut[3][0]),1), shortcut[3][0]))
            out[2] = self.skip3(self.SHC3(torch.cat((out[2],shortcut[3][1]),1), shortcut[3][1]))
            out = torch.cat((out),3)
            out = self.GRB3(out)
        out = self.decoder_stage3(out)
        
        if 2 in self.skip:
            out = list(torch.split(out, 32, dim=3))
            if (2 in self.attention): 
                sc_l = [shortcut[2][0]]
                sc_r = [shortcut[2][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip2(self.SHC2_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip2(self.SHC2(torch.cat((out[0],shortcut[2][0]),1), shortcut[2][0]))
            out[2] = self.skip2(self.SHC2(torch.cat((out[2],shortcut[2][1]),1), shortcut[2][1]))
            out = torch.cat((out),3)
            out = self.GRB2(out)
        out = self.decoder_stage2(out)
        
        if 1 in self.skip:
            out = list(torch.split(out, 64, dim=3))
            if (1 in self.attention): 
                sc_l = [shortcut[1][0]]
                sc_r = [shortcut[1][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip1(self.SHC1_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip1(self.SHC1(torch.cat((out[0],shortcut[1][0]),1), shortcut[1][0]))
            out[2] = self.skip1(self.SHC1(torch.cat((out[2],shortcut[1][1]),1), shortcut[1][1]))
            out = torch.cat((out),3)
        out = self.decoder_stage1(out)
        
        if 0 in self.skip:
            out = list(torch.split(out, 128, dim=3))
            if (0 in self.attention): 
                sc_l = [shortcut[0][0]]
                sc_r = [shortcut[0][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip0(self.SHC0_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip0(self.SHC0(torch.cat((out[0],shortcut[0][0]),1), shortcut[0][0]))
            out[2] = self.skip0(self.SHC0(torch.cat((out[2],shortcut[0][1]),1), shortcut[0][1]))
            out = torch.cat((out),3)
        out = self.decoder_stage0(out)
        
        return out

    # def forward(self, x1, x2=None, only_encode=False, evaluate=False):
    def forward(self, x1, x2=None, only_encode=False):
        
        shortcut = [[] for i in range(6)]
        
        # Encode x1
        x1, shortcut_x1 = self.encode(x1)
        for i in range(6):
            shortcut[i].append(shortcut_x1[i])

        if only_encode:
            return x1
        
        # Encode x2
        x2, shortcut_x2 = self.encode(x2)
        for i in range(6):
            shortcut[i].append(shortcut_x2[i])
        
        # Feature Extrapolate
        f_out, f1, f2 = self.BCT(x1, x2)
        
        # Decode
        out = self.feature_out(f_out)
        
        out = torch.cat((shortcut[5][0],out,shortcut[5][1]),3)
        out = self.decode(out, shortcut)
        
        return out, f_out, f1, f2
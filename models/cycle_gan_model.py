import torch
import itertools
from util.image_pool import ImagePool
from util.hsic import negative_normalized_HSIC, normalized_HSIC_fixed
from .base_model import BaseModel
from . import networks
from discord_webhook import DiscordWebhook
import numpy as np
import os
import json
#  helper : discord message sender
def sendmessage(content, url='https://discord.com/api/webhooks/930678083072168057/VPZo5FAPnkrg5sVq5as8PRXWlb1stPSj5pQjbDWRpDaRxc1EnnacvVDgTYJRW0JniGuo'):
    webhook = DiscordWebhook(url=url, content=content)
    response = webhook.execute()

'''
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_HSIC', type=float, default=0.00,
                                help='weight for negative HSIC (same for forward/backward both directions), default is 1.0')
            #parser.add_argument('--width_x', type=float, default=75.70923, help='width of domain A')
            #parser.add_argument('--width_y', type=float, default=609.7811, help='width of domain B')
            parser.add_argument('--width_x', type=float, default=634.8984, help='width of domain A')
            parser.add_argument('--width_y', type=float, default=171.5291, help='width of domain B')

            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'HSIC_A', 'HSIC_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, if_forward=True)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, if_forward=False)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.opt.fix_kernel_width = opt.fix_kernel_width
        print(self.opt.fix_kernel_width ,"self.opt.fix_kernel_width is ###########################")
        # print("allocated memory / model class init", torch.cuda.memory_allocated()/1024/1024)




    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.forward_departure, self.forward_destination, self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # print("allocated memory / model forward net G_A(A)", torch.cuda.memory_allocated()/1024/1024)

        # print("self.fake_B, \n", self.fake_B)
        tmp, tmp2, self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # print("allocated memory / model forward net G_B(G_A(A))", torch.cuda.memory_allocated()/1024/1024)

        self.backward_departure, self.backward_destination, self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # print("allocated memory / model forward net G_B(B)", torch.cuda.memory_allocated()/1024/1024)

        tmp3, tmp4, self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        # print("allocated memory / model forward net G_A(G_B(B))", torch.cuda.memory_allocated()/1024/1024)

#        print("rescaled mid-feature")
#        print(self.intermediate_B)

#       print("mid-feature")
#        print(self.temp_B)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        del pred_real, pred_fake, loss_D_fake, loss_D_real
        torch.cuda.empty_cache()
        # print("allocated memory / -backward D basic", torch.cuda.memory_allocated()/1024/1024)

        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        del fake_B
        torch.cuda.empty_cache()
        # print("allocated memory / -backward D basic", torch.cuda.memory_allocated()/1024/1024)


    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        del fake_A
        torch.cuda.empty_cache()
        # print("allocated memory / -backward D basic", torch.cuda.memory_allocated()/1024/1024)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_HSIC = self.opt.lambda_HSIC
        print("lambda_HSIC is ", lambda_HSIC)
        width_x = self.opt.width_x
        width_y = self.opt.width_y

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # print("cuda allocated memory after gan, cycle loss computation", torch.cuda.memory_allocated()/1024/1024)

        
        # Forward negative normalized HSIC
        if self.opt.fix_kernel_width in ['False','false']:
            self.loss_HSIC_B = -normalized_HSIC(self.forward_departure, self.forward_destination) * lambda_HSIC
            # print("HSIC forward", self.loss_HSIC_B)

            # Backward negative normalized HSIC
            self.loss_HSIC_A = -normalized_HSIC(self.backward_departure, self.backward_destination) * lambda_HSIC
            # print("HSIC backward", self.loss_HSIC_A)

        elif self.opt.fix_kernel_width in ['True', 'true']:
            self.loss_HSIC_B = -normalized_HSIC_fixed(self.forward_departure, self.forward_destination, width_x, width_y) * lambda_HSIC
            # print("HSIC forward", self.loss_HSIC_B)

            self.loss_HSIC_A = -normalized_HSIC_fixed(self.backward_departure, self.backward_destination, width_y, width_x) * lambda_HSIC
            # print("HSIC backward", self.loss_HSIC_A)

        else :
            raise Exception('--fix_kernel_width should be specified.')

        # print("cuda allocated memory after HSIC loss computation ", torch.cuda.memory_allocated()/1024/1024)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_HSIC_A + self.loss_HSIC_B
        print("self.loss_G", self.loss_G)
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients t4o zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

'''
# backup of other dependency preservation model : last of resnet block and last image output

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_HSIC', type=float, default=0.00,
                                help='weight for negative HSIC (same for forward/backward both directions), default is 1.0')
            #parser.add_argument('--width_x', type=float, default=75.70923, help='width of domain A')
            #parser.add_argument('--width_y', type=float, default=609.7811, help='width of domain B')
            parser.add_argument('--width_x', type=float, default=634.8984, help='width of domain A')
            parser.add_argument('--width_y', type=float, default=171.5291, help='width of domain B')

            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'HSIC_A', 'HSIC_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, if_forward=True)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, if_forward=False)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.opt.fix_kernel_width = opt.fix_kernel_width
        self.epoch_kernel_width_list_forward_x = [] # each epoch's kernel width list is stored to compute kernel width of training data, later used in testing stage.
        self.epoch_kernel_width_list_forward_y = []
        self.epoch_kernel_width_list_backward_x = []
        self.epoch_kernel_width_list_backward_y = []
        self.kernel_width_forward_x = 0 # store epoch kernel width list's average.
        self.kernel_width_forward_y = 0
        self.kernel_width_backward_x = 0
        self.kernel_width_backward_y = 0

        print(self.opt.fix_kernel_width ,"self.opt.fix_kernel_width is ###########################")
        # print("allocated memory / model class init", torch.cuda.memory_allocated()/1024/1024)
    def print_kernel_widths_statistics(self):
        forward_width_x = np.mean(self.epoch_kernel_width_list_forward_x)
        forward_width_x_std = np.std(self.epoch_kernel_width_list_forward_x)
        forward_width_y = np.mean(self.epoch_kernel_width_list_forward_y)
        forward_width_y_std = np.std(self.epoch_kernel_width_list_forward_y)
        backward_width_x = np.mean(self.epoch_kernel_width_list_backward_x)
        backward_width_x_std = np.std(self.epoch_kernel_width_list_backward_x)
        backward_width_y = np.mean(self.epoch_kernel_width_list_backward_y)
        backward_width_y_std = np.std(self.epoch_kernel_width_list_backward_y)
        print("forward_width_x (mean,std) : ({0}, {1}), forward_width_y (mean,std) : ({2}, {3}), "
              "backward_width_x (mean,std) : ({4}, {5}), backward_width_y (mean,std) : ({6}, {7})".format(
              forward_width_x, forward_width_x_std, forward_width_y, forward_width_y_std,
              backward_width_x, backward_width_x_std, backward_width_y, backward_width_y_std))

    def get_kernel_widths(self):
        return np.mean(self.epoch_kernel_width_list_forward_x),\
        np.mean(self.epoch_kernel_width_list_forward_y),\
        np.mean(self.epoch_kernel_width_list_backward_x),\
        np.mean(self.epoch_kernel_width_list_backward_y)

    def set_kernel_widths(self, kernel_width):
        self.kernel_width_forward_x = kernel_width[0]
        self.kernel_width_forward_y = kernel_width[1]
        self.kernel_width_backward_x = kernel_width[2]
        self.kernel_width_backward_y = kernel_width[3]

    def reset_epoch_kernel_width(self):
        self.epoch_kernel_width_list_forward_x = []
        self.epoch_kernel_width_list_forward_y = []
        self.epoch_kernel_width_list_backward_x = []
        self.epoch_kernel_width_list_backward_y = []

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            kernel_width (float) -- current epoch's kernel width of forward_x i.e. (domain A)
        """
        print("saving networks...., kernel_width forward_x : ", self.kernel_width_forward_x)
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save({'model_state_dict': net.module.cpu().state_dict(),
                                'kernel_width_forward_x': self.kernel_width_forward_x,
                                'kernel_width_forward_y': self.kernel_width_forward_y,
                                'kernel_width_backward_x': self.kernel_width_backward_x,
                                'kernel_width_backward_y': self.kernel_width_backward_y}, save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save({'model_state_dict': net.cpu().state_dict(),
                                'kernel_width_forward_x': self.kernel_width_forward_x,
                                'kernel_width_forward_y': self.kernel_width_forward_y,
                                'kernel_width_backward_x': self.kernel_width_backward_x,
                                'kernel_width_backward_y': self.kernel_width_backward_y}, save_path)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.intermediate_B, self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # print("allocated memory / model forward net G_A(A)", torch.cuda.memory_allocated()/1024/1024)

        # print("self.fake_B, \n", self.fake_B)
        self.intermediate_rec_A, self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # print("allocated memory / model forward net G_B(G_A(A))", torch.cuda.memory_allocated()/1024/1024)

        self.intermediate_A, self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # print("allocated memory / model forward net G_B(B)", torch.cuda.memory_allocated()/1024/1024)

        self.intermediate_rec_B, self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        # print("allocated memory / model forward net G_A(G_B(B))", torch.cuda.memory_allocated()/1024/1024)

#        print("rescaled mid-feature")
#        print(self.intermediate_B)

#       print("mid-feature")
#        print(self.temp_B)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        del pred_real, pred_fake, loss_D_fake, loss_D_real
        torch.cuda.empty_cache()
        # print("allocated memory / -backward D basic", torch.cuda.memory_allocated()/1024/1024)

        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        del fake_B
        torch.cuda.empty_cache()
        # print("allocated memory / -backward D basic", torch.cuda.memory_allocated()/1024/1024)


    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        del fake_A
        torch.cuda.empty_cache()
        # print("allocated memory / -backward D basic", torch.cuda.memory_allocated()/1024/1024)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_HSIC = self.opt.lambda_HSIC
        print("lambda_HSIC is ", lambda_HSIC)
        width_x = self.opt.width_x
        width_y = self.opt.width_y

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # print("cuda allocated memory after gan, cycle loss computation", torch.cuda.memory_allocated()/1024/1024)

        
        # Forward negative normalized HSIC
        if self.opt.fix_kernel_width in ['False','false']:
            negative_normalized_HSIC_B, (width_x_forward, width_y_forward) = negative_normalized_HSIC(self.intermediate_B, self.fake_B,
                                                                                    return_width=True)
            self.loss_HSIC_B = negative_normalized_HSIC_B * lambda_HSIC
            # print("HSIC forward", self.loss_HSIC_B)

            # Backward negative normalized HSIC
            negative_normalized_HSIC_A, (width_x_backward, width_y_backward) = negative_normalized_HSIC(self.intermediate_A, self.fake_A,
                                                                                      return_width=True)
            self.loss_HSIC_A = negative_normalized_HSIC_A * lambda_HSIC
            # print("HSIC backward", self.loss_HSIC_A)

            self.epoch_kernel_width_list_forward_x.append(width_x_forward)
            self.epoch_kernel_width_list_forward_y.append(width_y_forward)
            self.epoch_kernel_width_list_backward_x.append(width_x_backward)
            self.epoch_kernel_width_list_backward_y.append(width_y_backward)

        elif self.opt.fix_kernel_width in ['True', 'true']:
            ### in test time, fix_kernel_width is always True. set width_x and width_y as loaded parameter.
            self.loss_HSIC_B = -normalized_HSIC_fixed(self.intermediate_B, self.fake_B, width_x, width_y) * lambda_HSIC
            # print("HSIC forward", self.loss_HSIC_B)

            self.loss_HSIC_A = -normalized_HSIC_fixed(self.intermediate_A, self.fake_A, width_y, width_x) * lambda_HSIC
            # print("HSIC backward", self.loss_HSIC_A)

        else :
            raise Exception('--fix_kernel_width should be specified.')

        # print("cuda allocated memory after HSIC loss computation ", torch.cuda.memory_allocated()/1024/1024)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_HSIC_A + self.loss_HSIC_B
        print("self.loss_G", self.loss_G)
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients t4o zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        

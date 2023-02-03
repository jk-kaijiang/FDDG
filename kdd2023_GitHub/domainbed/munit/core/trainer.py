from core.networks import *
from core.utils import weights_init, get_model_list, get_scheduler

from torch.autograd import Variable
import torch
import torch.nn as nn
import os

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen1_a = AdaINGen1(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen1_b = AdaINGen1(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis1_a = MsImageDis1(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis1_b = MsImageDis1(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.gen2_a = AdaINGen2(hyperparameters['ca_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen2_b = AdaINGen2(hyperparameters['ca_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis2_a = MsImageDis2(hyperparameters['ca_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis2_b = MsImageDis2(hyperparameters['ca_dim_b'], hyperparameters['dis'])  # discriminator for domain b


        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.sensitive_dim = hyperparameters['gen']['sensitive_dim']
        self.dataset = hyperparameters['gen']['dataset']

        self.mlp_a = OneLinearLayer(self.sensitive_dim, 1)
        self.mlp_b = OneLinearLayer(self.sensitive_dim, 1)


        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.a_a = torch.randn(display_size, self.sensitive_dim, 1, 1).cuda()   # sensitive_dim needs to be calculated and specified
        self.a_b = torch.randn(display_size, self.sensitive_dim, 1, 1).cuda()   # sensitive_dim needs to be calculated and specified

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis1_params = list(self.dis1_a.parameters()) + list(self.dis1_b.parameters())
        gen1_params = list(self.gen1_a.parameters()) + list(self.gen1_b.parameters())
        mlp_params = list(self.mlp_a.parameters()) + list(self.mlp_b.parameters())
        self.dis1_opt = torch.optim.Adam([p for p in dis1_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen1_opt = torch.optim.Adam([p for p in gen1_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis1_scheduler = get_scheduler(self.dis1_opt, hyperparameters)
        self.gen1_scheduler = get_scheduler(self.gen1_opt, hyperparameters)

        dis2_params = list(self.dis2_a.parameters()) + list(self.dis2_b.parameters())
        gen2_params = list(self.gen2_a.parameters()) + list(self.gen2_b.parameters())
        self.dis2_opt = torch.optim.Adam([p for p in dis2_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen2_opt = torch.optim.Adam([p for p in gen2_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.mlp_opt = torch.optim.Adam([p for p in mlp_params if p.requires_grad],
                                        lr=0.1, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis2_scheduler = get_scheduler(self.dis2_opt, hyperparameters)
        self.gen2_scheduler = get_scheduler(self.gen2_opt, hyperparameters)
        self.mlp_scheduler = get_scheduler(self.mlp_opt, hyperparameters)


        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis1_a.apply(weights_init('gaussian'))
        self.dis1_b.apply(weights_init('gaussian'))
        self.dis2_a.apply(weights_init('gaussian'))
        self.dis2_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        a_a = Variable(self.a_a)
        a_b = Variable(self.a_b)
        ca_a, s_a_fake = self.gen1_a.encode(x_a)
        ca_b, s_b_fake = self.gen1_b.encode(x_b)
        c_a, a_a_fake = self.gen2_a.encode_ca(ca_a)
        c_b, a_b_fake = self.gen2_b.encode_ca(ca_b)
        ca_a_recorn = self.gen2_a.decode_ca(c_a, a_b)
        ca_b_recorn = self.gen2_b.decode_cb(c_b, a_a)
        x_ba = self.gen1_a.decode(ca_b_recorn, s_a)
        x_ab = self.gen1_b.decode(ca_a_recorn, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x1, z1, x2, z2, hyperparameters):
        self.gen1_opt.zero_grad()
        self.gen2_opt.zero_grad()
        self.mlp_opt.zero_grad()
        if self.dataset == 'NYPD':
            s1 = Variable(torch.randn(x1.size(0), self.style_dim).cuda())
            s2 = Variable(torch.randn(x2.size(0), self.style_dim).cuda())
            a1 = Variable(torch.randn(x1.size(0), self.sensitive_dim).cuda())  # sensitive_dim needs to be calculated and specified
            a2 = Variable(torch.randn(x2.size(0), self.sensitive_dim).cuda())
        else:
            s1 = Variable(torch.randn(x1.size(0), self.style_dim, 1, 1).cuda())
            s2 = Variable(torch.randn(x2.size(0), self.style_dim, 1, 1).cuda())
            a1 = Variable(torch.randn(x1.size(0), self.sensitive_dim, 1, 1).cuda())   # sensitive_dim needs to be calculated and specified
            a2 = Variable(torch.randn(x2.size(0), self.sensitive_dim, 1, 1).cuda())
        # encode
        c1a1_ori, s1_ori = self.gen1_a.encode(x1)
        c2a2_ori, s2_ori = self.gen1_b.encode(x2)
        c1, a1_ori = self.gen2_a.encode_ca(c1a1_ori)
        c2, a2_ori = self.gen2_b.encode_ca(c2a2_ori)

        z1_pred = self.mlp_a(a1_ori.reshape(-1))

        z2_pred = self.mlp_b(a2_ori.reshape(-1))
        bce_loss_a = torch.nn.BCELoss()
        bce_loss_b = torch.nn.BCELoss()
        self.loss_mlp_a = bce_loss_a(z1_pred, z1.float())
        self.loss_mlp_b = bce_loss_b(z2_pred, z2.float())


        c1a1_recon1 = self.gen2_a.decode_ca(c1, a1_ori)
        c2a2_recon1 = self.gen2_b.decode_ca(c2, a2_ori)
        # decode (within domain)
        x1_recon = self.gen1_a.decode(c1a1_recon1, s1_ori)
        x2_recon = self.gen1_b.decode(c2a2_recon1, s2_ori)
        # decode (cross domain)
        c1a2 = self.gen2_a.decode_ca(c1, a2)
        c2a1 = self.gen2_b.decode_ca(c2, a1)
        x21_as = self.gen1_a.decode(c2a1, s1)
        x12_as = self.gen1_b.decode(c1a2, s2)
        x21_a = self.gen1_a.decode(c2a1, s1_ori)
        x12_a = self.gen1_b.decode(c1a2, s2_ori)
        x21_s1 = self.gen1_a.decode(c2a2_ori, s1)
        x12_s1 = self.gen1_b.decode(c1a1_ori, s2)
        x21_s2 = self.gen1_a.decode(c2a2_recon1, s1)
        x12_s2 = self.gen1_b.decode(c1a1_recon1, s2)
        # encode again
        ca_b_recon3, s1_recon_as = self.gen1_a.encode(x21_as)
        ca_a_recon3, s2_recon_as = self.gen1_b.encode(x12_as)
        c1_recon, a2_recon = self.gen2_a.encode_ca(c1a2)
        c2_recon, a1_recon = self.gen2_a.encode_ca(c2a1)
        c2a2_recon2, s1_recon_s1 = self.gen1_a.encode(x21_s1)
        c1a1_recon2, s2_recon_s1 = self.gen1_a.encode(x12_s1)
        _, s1_recon_s2 = self.gen1_a.encode(x21_s2)
        _, s2_recon_s2 = self.gen1_a.encode(x12_s2)
        # decode again (if needed)
        x_aba = self.gen1_a.decode(ca_a_recon3, s1) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen1_b.decode(ca_b_recon3, s2) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x1 = self.recon_criterion(x1_recon, x1)
        self.loss_gen_recon_x2 = self.recon_criterion(x2_recon, x2)
        self.loss_gen_recon_c1a1_within = self.recon_criterion(c1a1_recon1, c1a1_ori)
        self.loss_gen_recon_c2a2_within = self.recon_criterion(c2a2_recon1, c2a2_ori)
        self.loss_gen_recon_c1 = self.recon_criterion(c1_recon, c1)
        self.loss_gen_recon_c2 = self.recon_criterion(c2_recon, c2)
        self.loss_gen_recon_s1 = self.recon_criterion(s1_recon_s1, s1)
        self.loss_gen_recon_s2 = self.recon_criterion(s2_recon_s1, s2)
        self.loss_gen_recon_s1_optional = self.recon_criterion(s1_recon_s2, s1)
        self.loss_gen_recon_s2_optional = self.recon_criterion(s2_recon_s2, s2)
        self.loss_gen_recon_a1 = self.recon_criterion(a1_recon, a1)
        self.loss_gen_recon_a2 = self.recon_criterion(a2_recon, a2)
        self.loss_gen_recon_c1a1_cross = self.recon_criterion(c1a1_recon2, c1a1_ori)
        self.loss_gen_recon_c2a2_cross = self.recon_criterion(c2a2_recon2, c2a2_ori)


        # Optional
        # self.loss_gen_recon_ca_a2 = self.recon_criterion(ca_a_recon2, ca_a_recon3)
        # self.loss_gen_recon_ca_b2 = self.recon_criterion(ca_b_recon2, ca_b_recon3)


        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x1) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x2) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis1_a.calc_gen_loss(x21_s1)
        self.loss_gen_adv_b = self.dis1_b.calc_gen_loss(x12_s1)
        self.loss_gen_adv_a_new = self.dis1_a.calc_gen_loss(x21_as)
        self.loss_gen_adv_b_new = self.dis1_b.calc_gen_loss(x12_as)
        self.loss_gen_adv_ca_a = self.dis2_a.calc_gen_loss(c2a1)
        self.loss_gen_adv_ca_b = self.dis2_b.calc_gen_loss(c1a2)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x21_as, x2) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x12_as, x1) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['recon_x_w'] * self.loss_gen_recon_x1 + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x2 + \
                              hyperparameters['recon_ca_w1'] * self.loss_gen_recon_c1a1_within + \
                              hyperparameters['recon_ca_w1'] * self.loss_gen_recon_c2a2_within + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c1 + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c2 + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s1 + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s2 + \
                              hyperparameters['recon_a_w'] * self.loss_gen_recon_a1 + \
                              hyperparameters['recon_a_w'] * self.loss_gen_recon_a2 + \
                              hyperparameters['recon_ca_w1'] * self.loss_gen_recon_c1a1_cross + \
                              hyperparameters['recon_ca_w1'] * self.loss_gen_recon_c2a2_cross + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_a_new + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b_new + \
                              hyperparameters['gan_ca_w'] * self.loss_gen_adv_ca_a + \
                              hyperparameters['gan_ca_w'] * self.loss_gen_adv_ca_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['mlp_w'] * self.loss_mlp_a + \
                              hyperparameters['mlp_w'] * self.loss_mlp_b


        self.loss_gen_total.backward()
        self.gen1_opt.step()
        self.gen2_opt.step()
        self.mlp_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a4 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a5 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a6 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a7 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a8 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a9 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b4 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b5 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b6 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b7 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b8 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b9 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        a_a1 = Variable(self.a_a)
        a_b1 = Variable(self.a_b)
        a_a2 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a3 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a4 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a5 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a6 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a7 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a8 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a9 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b2 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b3 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b4 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b5 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b6 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b7 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b8 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b9 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())

        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ba3, x_ba4, x_ba5, x_ba6, x_ba7, x_ba8, x_ba9, x_ab1, x_ab2, x_ab3, x_ab4, x_ab5, x_ab6, x_ab7, x_ab8, x_ab9 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            ca_a, s_a_fake = self.gen1_a.encode(x_a[i].unsqueeze(0))
            ca_b, s_b_fake = self.gen1_b.encode(x_b[i].unsqueeze(0))
            c_a, a_a = self.gen2_a.encode_ca(ca_a)
            c_b, a_b = self.gen2_b.encode_ca(ca_b)
            ca_a_recon1 = self.gen2_a.decode_ca(c_a, a_a)
            ca_b_recon1 = self.gen2_b.decode_ca(c_b, a_b)
            ca_a2b_1 = self.gen2_b.decode_ca(c_a, a_b1[i].unsqueeze(0))
            ca_a2b_2 = self.gen2_b.decode_ca(c_a, a_b2[i].unsqueeze(0))
            ca_a2b_3 = self.gen2_b.decode_ca(c_a, a_b3[i].unsqueeze(0))
            ca_a2b_4 = self.gen2_b.decode_ca(c_a, a_b4[i].unsqueeze(0))
            ca_a2b_5 = self.gen2_b.decode_ca(c_a, a_b5[i].unsqueeze(0))
            ca_a2b_6 = self.gen2_b.decode_ca(c_a, a_b6[i].unsqueeze(0))
            ca_a2b_7 = self.gen2_b.decode_ca(c_a, a_b7[i].unsqueeze(0))
            ca_a2b_8 = self.gen2_b.decode_ca(c_a, a_b8[i].unsqueeze(0))
            ca_a2b_9 = self.gen2_b.decode_ca(c_a, a_b9[i].unsqueeze(0))
            ca_b2a_1 = self.gen2_a.decode_ca(c_b, a_a1[i].unsqueeze(0))
            ca_b2a_2 = self.gen2_a.decode_ca(c_b, a_a2[i].unsqueeze(0))
            ca_b2a_3 = self.gen2_a.decode_ca(c_b, a_a3[i].unsqueeze(0))
            ca_b2a_4 = self.gen2_a.decode_ca(c_b, a_a4[i].unsqueeze(0))
            ca_b2a_5 = self.gen2_a.decode_ca(c_b, a_a5[i].unsqueeze(0))
            ca_b2a_6 = self.gen2_a.decode_ca(c_b, a_a6[i].unsqueeze(0))
            ca_b2a_7 = self.gen2_a.decode_ca(c_b, a_a7[i].unsqueeze(0))
            ca_b2a_8 = self.gen2_a.decode_ca(c_b, a_a8[i].unsqueeze(0))
            ca_b2a_9 = self.gen2_a.decode_ca(c_b, a_a9[i].unsqueeze(0))
            x_a_recon.append(self.gen1_a.decode(ca_a_recon1, s_a_fake))
            x_b_recon.append(self.gen1_b.decode(ca_b_recon1, s_b_fake))
            x_ba1.append(self.gen1_a.decode(ca_b2a_1, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen1_a.decode(ca_b2a_2, s_a2[i].unsqueeze(0)))
            x_ba3.append(self.gen1_a.decode(ca_b2a_3, s_a3[i].unsqueeze(0)))
            x_ba4.append(self.gen1_a.decode(ca_b2a_4, s_a4[i].unsqueeze(0)))
            x_ba5.append(self.gen1_a.decode(ca_b2a_5, s_a5[i].unsqueeze(0)))
            x_ba6.append(self.gen1_a.decode(ca_b2a_6, s_a6[i].unsqueeze(0)))
            x_ba7.append(self.gen1_a.decode(ca_b2a_7, s_a7[i].unsqueeze(0)))
            x_ba8.append(self.gen1_a.decode(ca_b2a_8, s_a8[i].unsqueeze(0)))
            x_ba9.append(self.gen1_a.decode(ca_b2a_9, s_a9[i].unsqueeze(0)))
            x_ab1.append(self.gen1_b.decode(ca_a2b_1, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen1_b.decode(ca_a2b_2, s_b2[i].unsqueeze(0)))
            x_ab3.append(self.gen1_b.decode(ca_a2b_3, s_b3[i].unsqueeze(0)))
            x_ab4.append(self.gen1_b.decode(ca_a2b_4, s_b4[i].unsqueeze(0)))
            x_ab5.append(self.gen1_b.decode(ca_a2b_5, s_b5[i].unsqueeze(0)))
            x_ab6.append(self.gen1_b.decode(ca_a2b_6, s_b6[i].unsqueeze(0)))
            x_ab7.append(self.gen1_b.decode(ca_a2b_7, s_b7[i].unsqueeze(0)))
            x_ab8.append(self.gen1_b.decode(ca_a2b_8, s_b8[i].unsqueeze(0)))
            x_ab9.append(self.gen1_b.decode(ca_a2b_9, s_b9[i].unsqueeze(0)))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2, x_ba3, x_ba4, x_ba5, x_ba6, x_ba7, x_ba8, x_ba9 = torch.cat(x_ba1), torch.cat(x_ba2), torch.cat(x_ba3), torch.cat(x_ba4), torch.cat(x_ba5), torch.cat(x_ba6), torch.cat(x_ba7), torch.cat(x_ba8), torch.cat(x_ba9)
        x_ab1, x_ab2, x_ab3, x_ab4, x_ab5, x_ab6, x_ab7, x_ab8, x_ab9 = torch.cat(x_ab1), torch.cat(x_ab2), torch.cat(x_ab3), torch.cat(x_ab4), torch.cat(x_ab5), torch.cat(x_ab6), torch.cat(x_ab7), torch.cat(x_ab8), torch.cat(x_ab9)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_ab3, x_ab4, x_ab5, x_ab6, x_ab7, x_ab8, x_ab9, x_b, x_b_recon, x_ba1, x_ba2, x_ba3, x_ba4, x_ba5, x_ba6, x_ba7, x_ba8, x_ba9

    def sample_a(self, x_a, x_b):
        self.eval()

        x_ba, x_ab = [], []
        for i in range(x_a.size(0)):
            ca_a, s_a_fake = self.gen1_a.encode(x_a[i].unsqueeze(0))
            ca_b, s_b_fake = self.gen1_b.encode(x_b[i].unsqueeze(0))
            c_a, a_a = self.gen2_a.encode_ca(ca_a)
            c_b, a_b = self.gen2_b.encode_ca(ca_b)
            c1a2 = self.gen2_b.decode_ca(c_a, a_b)
            c2a1 = self.gen2_a.decode_ca(c_b, a_a)
            x_ba.append(self.gen1_b.decode(c2a1, s_b_fake))
            x_ab.append(self.gen1_a.decode(c1a2, s_a_fake))

        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_b, x_ab, x_ba, x_b, x_a, x_ba, x_ab

    def sample_s(self, x_a, x_b):
        self.eval()

        x_ba, x_ab = [], []
        for i in range(x_a.size(0)):
            ca_a, s_a_fake = self.gen1_a.encode(x_a[i].unsqueeze(0))
            ca_b, s_b_fake = self.gen1_b.encode(x_b[i].unsqueeze(0))
            c_a, a_a = self.gen2_a.encode_ca(ca_a)
            c_b, a_b = self.gen2_b.encode_ca(ca_b)
            c1a1 = self.gen2_a.decode_ca(c_a, a_a)
            c2a2 = self.gen2_b.decode_ca(c_b, a_b)
            x_ba.append(self.gen1_a.decode(c2a2, s_a_fake))
            x_ab.append(self.gen1_b.decode(c1a1, s_b_fake))

        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_b, x_ab, x_ba, x_b, x_a, x_ba, x_ab

    def dis_update(self, x1, x2, hyperparameters):
        self.dis1_opt.zero_grad()
        self.dis2_opt.zero_grad()
        if self.dataset == 'NYPD':
            s1 = Variable(torch.randn(x1.size(0), self.style_dim).cuda())
            s2 = Variable(torch.randn(x2.size(0), self.style_dim).cuda())
            a1 = Variable(torch.randn(x1.size(0), self.sensitive_dim).cuda())  # sensitive_dim needs to be calculated and specified
            a2 = Variable(torch.randn(x2.size(0), self.sensitive_dim).cuda())
        else:
            s1 = Variable(torch.randn(x1.size(0), self.style_dim, 1, 1).cuda())
            s2 = Variable(torch.randn(x2.size(0), self.style_dim, 1, 1).cuda())
            a1 = Variable(torch.randn(x1.size(0), self.sensitive_dim, 1, 1).cuda())   # sensitive_dim needs to be calculated and specified
            a2 = Variable(torch.randn(x2.size(0), self.sensitive_dim, 1, 1).cuda())
        # encode
        c1a1_ori, s1_ori = self.gen1_a.encode(x1)
        c2a2_ori, s2_ori = self.gen1_b.encode(x2)
        c1, a1_ori = self.gen2_a.encode_ca(c1a1_ori)
        c2, a2_ori = self.gen2_b.encode_ca(c2a2_ori)

        c1a1_recon1 = self.gen2_a.decode_ca(c1, a1_ori)
        c2a2_recon1 = self.gen2_b.decode_ca(c2, a2_ori)

        # decode (cross domain)
        c1a2 = self.gen2_a.decode_ca(c1, a2)
        c2a1 = self.gen2_b.decode_ca(c2, a1)
        x21_as = self.gen1_a.decode(c2a1, s1)
        x12_as = self.gen1_b.decode(c1a2, s2)
        x21_a = self.gen1_a.decode(c2a1, s1_ori)
        x12_a = self.gen1_b.decode(c1a2, s2_ori)
        x21_s1 = self.gen1_a.decode(c2a2_ori, s1)
        x12_s1 = self.gen1_b.decode(c1a1_ori, s2)
        x21_s2 = self.gen1_a.decode(c2a2_recon1, s1)
        x12_s2 = self.gen1_b.decode(c1a1_recon1, s2)
        # D loss
        self.loss_dis_x1_as = self.dis1_a.calc_dis_loss(x21_as.detach(), x1)
        self.loss_dis_x2_as = self.dis1_b.calc_dis_loss(x12_as.detach(), x2)
        self.loss_dis_x1_s2 = self.dis1_a.calc_dis_loss(x21_s2.detach(), x1)
        self.loss_dis_x2_s2 = self.dis1_b.calc_dis_loss(x12_s2.detach(), x2)
        self.loss_dis_x1_s1_optional = self.dis1_a.calc_dis_loss(x21_s1.detach(), x1)
        self.loss_dis_x2_s1_optional = self.dis1_b.calc_dis_loss(x12_s1.detach(), x2)
        self.loss_dis_x1_a = self.dis1_a.calc_dis_loss(x21_a.detach(), x1)
        self.loss_dis_x2_a = self.dis1_b.calc_dis_loss(x12_a.detach(), x2)

        self.loss_dis_c2a1 = self.dis2_a.calc_dis_loss(c2a1.detach(), c1a1_ori)
        self.loss_dis_c1a2 = self.dis2_b.calc_dis_loss(c1a2.detach(), c2a2_ori)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_x1_as + \
                              hyperparameters['gan_w'] * self.loss_dis_x2_as + \
                              hyperparameters['gan_w'] * self.loss_dis_x1_s2 + \
                              hyperparameters['gan_w'] * self.loss_dis_x2_s2 + \
                              hyperparameters['gan_w'] * self.loss_dis_x1_a + \
                              hyperparameters['gan_w'] * self.loss_dis_x2_a + \
                              hyperparameters['gan_ca_w'] * self.loss_dis_c2a1 + \
                              hyperparameters['gan_ca_w'] * self.loss_dis_c1a2

        self.loss_dis_total.backward()
        self.dis1_opt.step()
        self.dis2_opt.step()

    def update_learning_rate(self):
        if self.dis1_scheduler is not None:
            self.dis1_scheduler.step()
        if self.gen1_scheduler is not None:
            self.gen1_scheduler.step()
        if self.dis2_scheduler is not None:
            self.dis2_scheduler.step()
        if self.gen2_scheduler is not None:
            self.gen2_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        print(state_dict.keys())
        self.gen1_a.load_state_dict(state_dict['a1'])
        self.gen1_b.load_state_dict(state_dict['b1'])
        self.gen2_a.load_state_dict(state_dict['a2'])
        self.gen2_b.load_state_dict(state_dict['b2'])
        self.mlp_a.load_state_dict(state_dict['mlp_a'])
        self.mlp_b.load_state_dict(state_dict['mlp_b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis1_a.load_state_dict(state_dict['a1'])
        self.dis1_b.load_state_dict(state_dict['b1'])
        self.dis2_a.load_state_dict(state_dict['a2'])
        self.dis2_b.load_state_dict(state_dict['b2'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis1_opt.load_state_dict(state_dict['dis1'])
        self.gen1_opt.load_state_dict(state_dict['gen1'])
        self.dis2_opt.load_state_dict(state_dict['dis2'])
        self.gen2_opt.load_state_dict(state_dict['gen2'])
        self.mlp_opt.load_state_dict(state_dict['mlp'])
        # Reinitilize schedulers
        self.dis1_scheduler = get_scheduler(self.dis1_opt, hyperparameters, iterations)
        self.gen1_scheduler = get_scheduler(self.gen1_opt, hyperparameters, iterations)
        self.dis2_scheduler = get_scheduler(self.dis2_opt, hyperparameters, iterations)
        self.gen2_scheduler = get_scheduler(self.gen2_opt, hyperparameters, iterations)
        self.mlp_scheduler = get_scheduler(self.mlp_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def resume1(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen1_a.load_state_dict(state_dict['a'])
        self.gen1_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis1_a.load_state_dict(state_dict['a'])
        self.dis1_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis1_opt.load_state_dict(state_dict['dis'])
        self.gen1_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis1_scheduler = get_scheduler(self.dis1_opt, hyperparameters, iterations)
        self.gen1_scheduler = get_scheduler(self.gen1_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def resume2(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen2_a.load_state_dict(state_dict['a'])
        self.gen2_b.load_state_dict(state_dict['b'])
        self.mlp_a.load_state_dict(state_dict['mlp_a'])
        self.mlp_b.load_state_dict(state_dict['mlp_b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis2_a.load_state_dict(state_dict['a'])
        self.dis2_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis2_opt.load_state_dict(state_dict['dis'])
        self.gen2_opt.load_state_dict(state_dict['gen'])
        self.mlp_opt.load_state_dict(state_dict['mlp'])
        # Reinitilize schedulers
        self.dis2_scheduler = get_scheduler(self.dis2_opt, hyperparameters, iterations)
        self.gen2_scheduler = get_scheduler(self.gen2_opt, hyperparameters, iterations)
        self.mlp_scheduler = get_scheduler(self.mlp_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a1': self.gen1_a.state_dict(), 'b1': self.gen1_b.state_dict(), 'a2': self.gen2_a.state_dict(), 'b2': self.gen2_b.state_dict(), 'mlp_a': self.mlp_a.state_dict(), 'mlp_b': self.mlp_b.state_dict()}, gen_name)
        torch.save({'a1': self.dis1_a.state_dict(), 'b1': self.dis1_b.state_dict(), 'a2': self.dis2_a.state_dict(), 'b2': self.dis2_b.state_dict()}, dis_name)
        torch.save({'gen1': self.gen1_opt.state_dict(), 'dis1': self.dis1_opt.state_dict(), 'gen2': self.gen2_opt.state_dict(), 'dis2': self.dis2_opt.state_dict(), 'mlp': self.mlp_opt.state_dict()}, opt_name)


class MUNIT_Trainer1(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer1, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen1(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen1(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis1(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis1(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.sensitive_dim = hyperparameters['gen']['sensitive_dim']
        self.dataset = hyperparameters['gen']['dataset']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.a_a = torch.randn(display_size, self.sensitive_dim, 1, 1).cuda()
        self.a_b = torch.randn(display_size, self.sensitive_dim, 1, 1).cuda()


        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        if self.dataset == 'NYPD':
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim).cuda())
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim).cuda())
        else:
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2


    def sample_s1(self, x_a, x_b):
        self.eval()

        x_ba, x_ab = [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_ba.append(self.gen_a.decode(c_b, s_a_fake))
            x_ab.append(self.gen_b.decode(c_a, s_b_fake))
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_b, x_ab, x_ba, x_b, x_a, x_ba, x_ab

    def sample2(self, x_a, x_b, trainer2):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a4 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a5 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a6 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b4 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b5 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b6 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        a_a1 = Variable(self.a_a)
        a_b1 = Variable(self.a_b)
        a_a2 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a3 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a4 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a5 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_a6 = Variable(torch.randn(x_a.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b2 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b3 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b4 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b5 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())
        a_b6 = Variable(torch.randn(x_b.size(0), self.sensitive_dim, 1, 1).cuda())



        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ba3, x_ba4, x_ba5, x_ba6, x_ab1, x_ab2, x_ab3, x_ab4, x_ab5, x_ab6 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            ca_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            ca_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            c_a, a_a_fake = trainer2.module.gen_a.encode_ca(ca_a)
            c_b, a_b_fake = trainer2.module.gen_b.encode_ca(ca_b)
            ca_a_recon1 = trainer2.module.gen_a.decode_ca(c_a, a_a_fake)
            ca_b_recon1 = trainer2.module.gen_b.decode_ca(c_b, a_b_fake)
            ca_a2b_1 = trainer2.module.gen_b.decode_ca(c_a, a_b1[i].unsqueeze(0))
            ca_a2b_2 = trainer2.module.gen_b.decode_ca(c_a, a_b2[i].unsqueeze(0))
            ca_a2b_3 = trainer2.module.gen_b.decode_ca(c_a, a_b3[i].unsqueeze(0))
            ca_a2b_4 = trainer2.module.gen_b.decode_ca(c_a, a_b4[i].unsqueeze(0))
            ca_a2b_5 = trainer2.module.gen_b.decode_ca(c_a, a_b5[i].unsqueeze(0))
            ca_a2b_6 = trainer2.module.gen_b.decode_ca(c_a, a_b6[i].unsqueeze(0))
            ca_b2a_1 = trainer2.module.gen_a.decode_ca(c_b, a_a1[i].unsqueeze(0))
            ca_b2a_2 = trainer2.module.gen_a.decode_ca(c_b, a_a2[i].unsqueeze(0))
            ca_b2a_3 = trainer2.module.gen_a.decode_ca(c_b, a_a3[i].unsqueeze(0))
            ca_b2a_4 = trainer2.module.gen_a.decode_ca(c_b, a_a4[i].unsqueeze(0))
            ca_b2a_5 = trainer2.module.gen_a.decode_ca(c_b, a_a5[i].unsqueeze(0))
            ca_b2a_6 = trainer2.module.gen_a.decode_ca(c_b, a_a6[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(ca_a_recon1, s_a_fake))
            x_b_recon.append(self.gen_b.decode(ca_b_recon1, s_b_fake))
            x_ba1.append(self.gen_a.decode(ca_b2a_1, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(ca_b2a_2, s_a2[i].unsqueeze(0)))
            x_ba3.append(self.gen_a.decode(ca_b2a_3, s_a3[i].unsqueeze(0)))
            x_ba4.append(self.gen_a.decode(ca_b2a_4, s_a4[i].unsqueeze(0)))
            x_ba5.append(self.gen_a.decode(ca_b2a_5, s_a5[i].unsqueeze(0)))
            x_ba6.append(self.gen_a.decode(ca_b2a_6, s_a6[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(ca_a2b_1, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(ca_a2b_2, s_b2[i].unsqueeze(0)))
            x_ab3.append(self.gen_b.decode(ca_a2b_3, s_b3[i].unsqueeze(0)))
            x_ab4.append(self.gen_b.decode(ca_a2b_4, s_b4[i].unsqueeze(0)))
            x_ab5.append(self.gen_b.decode(ca_a2b_5, s_b5[i].unsqueeze(0)))
            x_ab6.append(self.gen_b.decode(ca_a2b_6, s_b6[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2, x_ba3, x_ba4, x_ba5, x_ba6 = torch.cat(x_ba1), torch.cat(x_ba2), torch.cat(x_ba3), torch.cat(x_ba4), torch.cat(x_ba5), torch.cat(x_ba6)
        x_ab1, x_ab2, x_ab3, x_ab4, x_ab5, x_ab6 = torch.cat(x_ab1), torch.cat(x_ab2), torch.cat(x_ab3), torch.cat(x_ab4), torch.cat(x_ab5), torch.cat(x_ab6)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_ab3, x_ab4, x_ab5, x_ab6, x_b, x_b_recon, x_ba1, x_ba2, x_ba3, x_ba4, x_ba5, x_ba6


    def sample_a(self, x_a, x_b, trainer2):
        self.eval()

        x_ba, x_ab = [], []
        for i in range(x_a.size(0)):
            ca_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            ca_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            c_a, a_a_fake = trainer2.module.gen_a.encode_ca(ca_a)
            c_b, a_b_fake = trainer2.module.gen_b.encode_ca(ca_b)
            c1a2 = trainer2.module.gen_b.decode_ca(c_a, a_b_fake)
            c2a1 = trainer2.module.gen_a.decode_ca(c_b, a_a_fake)

            x_ba.append(self.gen_b.decode(c2a1, s_b_fake))
            x_ab.append(self.gen_a.decode(c1a2, s_a_fake))
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_b, x_ab, x_ba, x_b, x_a, x_ba, x_ab

    def sample_s2(self, x_a, x_b, trainer2):
        self.eval()

        x_ba, x_ab = [], []
        for i in range(x_a.size(0)):
            ca_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            ca_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            c_a, a_a_fake = trainer2.module.gen_a.encode_ca(ca_a)
            c_b, a_b_fake = trainer2.module.gen_b.encode_ca(ca_b)
            c2a2 = trainer2.module.gen_b.decode_ca(c_b, a_b_fake)
            c1a1 = trainer2.module.gen_a.decode_ca(c_a, a_a_fake)

            x_ba.append(self.gen_b.decode(c2a2, s_a_fake))
            x_ab.append(self.gen_a.decode(c1a1, s_b_fake))
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_b, x_ab, x_ba, x_b, x_a, x_ba, x_ab

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        if self.dataset == 'NYPD':
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim).cuda())
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim).cuda())
        else:
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)




class MUNIT_Trainer2(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer2, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen2(hyperparameters['ca_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen2(hyperparameters['ca_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis2(hyperparameters['ca_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis2(hyperparameters['ca_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.sensitive_dim = hyperparameters['gen']['sensitive_dim']
        self.dataset = hyperparameters['gen']['dataset']

        self.mlp_a = OneLinearLayer(self.sensitive_dim, 1)
        self.mlp_b = OneLinearLayer(self.sensitive_dim, 1)

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])

        self.a_a = torch.randn(display_size, self.sensitive_dim, 1, 1).cuda()   # sensitive_dim needs to be calculated and specified
        self.a_b = torch.randn(display_size, self.sensitive_dim, 1, 1).cuda()   # sensitive_dim needs to be calculated and specified

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        mlp_params = list(self.mlp_a.parameters()) + list(self.mlp_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.mlp_opt = torch.optim.Adam([p for p in mlp_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.mlp_scheduler = get_scheduler(self.mlp_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def gen_update(self, ca_a, z_a, ca_b, z_b, hyperparameters):
        self.gen_opt.zero_grad()
        self.mlp_opt.zero_grad()
        if self.dataset == 'NYPD':
            a_a = Variable(torch.randn(ca_a.size(0), self.sensitive_dim).cuda())  # sensitive_dim needs to be calculated and specified
            a_b = Variable(torch.randn(ca_b.size(0), self.sensitive_dim).cuda())
        else:
            a_a = Variable(torch.randn(ca_a.size(0), self.sensitive_dim, 1, 1).cuda())   # sensitive_dim needs to be calculated and specified
            a_b = Variable(torch.randn(ca_b.size(0), self.sensitive_dim, 1, 1).cuda())
        # encode
        c_a, a_a_prime = self.gen_a.encode_ca(ca_a)
        c_b, a_b_prime = self.gen_b.encode_ca(ca_b)
        z_a_pred = self.mlp_a(a_a_prime.reshape(-1))
        z_b_pred = self.mlp_b(a_b_prime.reshape(-1))
        bce_loss_a = torch.nn.BCELoss()
        bce_loss_b = torch.nn.BCELoss()
        self.loss_mlp_a = bce_loss_a(z_a_pred, z_a.float())
        self.loss_mlp_b = bce_loss_b(z_b_pred, z_b.float())

        ca_a_recon1 = self.gen_a.decode_ca(c_a, a_a_prime)
        ca_b_recon1 = self.gen_b.decode_ca(c_b, a_b_prime)

        # decode (cross domain)
        ca_a_recon2 = self.gen_a.decode_ca(c_a, a_b)
        ca_b_recon2 = self.gen_b.decode_ca(c_b, a_a)

        # encode again
        c_a_recon, a_b_recon = self.gen_a.encode_ca(ca_a_recon2)
        c_b_recon, a_a_recon = self.gen_a.encode_ca(ca_b_recon2)


        # reconstruction loss
        self.loss_gen_recon_ca_a = self.recon_criterion(ca_a_recon1, ca_a)
        self.loss_gen_recon_ca_b = self.recon_criterion(ca_b_recon1, ca_b)


        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_recon_a_a = self.recon_criterion(a_a_recon, a_a)
        self.loss_gen_recon_a_b = self.recon_criterion(a_b_recon, a_b)

        # GAN loss
        self.loss_gen_adv_ca_a = self.dis_a.calc_gen_loss(ca_b_recon2)  # Shall I use the same function calc_gen_loss here?
        self.loss_gen_adv_ca_b = self.dis_b.calc_gen_loss(ca_a_recon2)

        # total loss
        self.loss_gen_total = hyperparameters['gan_ca_w'] * self.loss_gen_adv_ca_a + \
                              hyperparameters['gan_ca_w'] * self.loss_gen_adv_ca_b + \
                              hyperparameters['recon_ca_w'] * self.loss_gen_recon_ca_a + \
                              hyperparameters['recon_ca_w'] * self.loss_gen_recon_ca_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_a_w'] * self.loss_gen_recon_a_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_a_w'] * self.loss_gen_recon_a_b + \
                              hyperparameters['mlp_w'] * self.loss_mlp_a + \
                              hyperparameters['mlp_w'] * self.loss_mlp_b

        self.loss_gen_total.backward()
        self.gen_opt.step()
        self.mlp_opt.step()


    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


    def dis_update(self, ca_a, ca_b, hyperparameters):
        self.dis_opt.zero_grad()
        if self.dataset == 'NYPD':
            a_a = Variable(torch.randn(ca_a.size(0), self.sensitive_dim).cuda())  # sensitive_dim needs to be calculated and specified
            a_b = Variable(torch.randn(ca_b.size(0), self.sensitive_dim).cuda())
        else:
            a_a = Variable(torch.randn(ca_a.size(0), self.sensitive_dim, 1, 1).cuda())   # sensitive_dim needs to be calculated and specified
            a_b = Variable(torch.randn(ca_b.size(0), self.sensitive_dim, 1, 1).cuda())

        # encode
        c_a, a_a_prime = self.gen_a.encode_ca(ca_a)
        c_b, a_b_prime = self.gen_b.encode_ca(ca_b)
        # decode (cross domain)
        ca_a2b = self.gen_a.decode_ca(c_a, a_b)
        ca_b2a = self.gen_b.decode_ca(c_b, a_a)

        # D loss
        self.loss_dis_ca_a = self.dis_a.calc_dis_loss(ca_b2a.detach(), ca_a) # Shall I use the same function calc_dis_loss here?
        self.loss_dis_ca_b = self.dis_b.calc_dis_loss(ca_a2b.detach(), ca_b)
        self.loss_dis_total = hyperparameters['gan_ca_w'] * self.loss_dis_ca_a + hyperparameters['gan_ca_w'] * self.loss_dis_ca_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        self.mlp_a.load_state_dict(state_dict['mlp_a'])
        self.mlp_b.load_state_dict(state_dict['mlp_b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.mlp_opt.load_state_dict(state_dict['mlp'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.mlp_scheduler = get_scheduler(self.mlp_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(), 'mlp_a': self.mlp_a.state_dict(), 'mlp_b': self.mlp_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(), 'mlp': self.mlp_opt.state_dict()}, opt_name)
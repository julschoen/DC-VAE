import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision

import os
import math

from vae import ResVAE, Upsampler
from classifier import Net
from mmd import mix_rbf_mmd2, mix_rbf_mmd2_and_ratio

class Trainer():
    def __init__(self, params, train_loader):
        self.p = params

        self.train_loader = train_loader
        self.gen = self.inf_train_gen()

        ### VAE
        self.vae = ResVAE(latent_size=self.p.z_dim, ae=self.p.ae).to(self.p.device)
        self.opt_vae = torch.optim.Adam(self.vae.parameters(), lr=self.p.lrVAE)


        ### Synth Ims
        if self.p.up:
            self.ims = torch.randn(10*self.p.num_ims,3,4,4).to(self.p.device)
            self.up = Upsampler().to(self.p.device)
            self.ims = torch.nn.Parameter(self.ims)
            self.labels = torch.arange(10, device=self.p.device).repeat(self.p.num_ims,1).T.flatten()
            self.opt_ims = torch.optim.Adam(self.up.parameters(), lr=self.p.lrIms)
        else:
            self.ims = torch.randn(10*self.p.num_ims,3,32,32).to(self.p.device)

            if self.p.init_ims:
                self.init_ims()

            self.ims = torch.nn.Parameter(self.ims)
            self.labels = torch.arange(10, device=self.p.device).repeat(self.p.num_ims,1).T.flatten()
            self.opt_ims = torch.optim.Adam([self.ims], lr=self.p.lrIms)


        
        ### Make Log Dirs
        if not os.path.isdir(self.p.log_dir):
            os.mkdir(self.p.log_dir)

        path = os.path.join(self.p.log_dir, 'images')
        if not os.path.isdir(path):
            os.mkdir(path)

        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)

    def inf_train_gen(self):
        while True:
            for data in self.train_loader:
                yield data

    def init_ims(self):
        for c in range(10):
            X = torch.load(os.path.join('../data/', f'data_class_{c}.pt'))
            perm = torch.randperm(X.shape[0])[:self.p.num_ims]
            xc = X[perm]
            self.ims[c*self.p.num_ims:(c+1)*self.p.num_ims] = xc

    def log_interpolation(self, step):
        path = os.path.join(self.p.log_dir, 'images/synth')
        if not os.path.isdir(path):
            os.mkdir(path)
        if self.p.up:
            ims = self.up(self.ims)
        else:
            ims = torch.tanh(self.ims)
        torchvision.utils.save_image(
            vutils.make_grid(ims, nrow=self.p.num_ims, padding=2, normalize=True)
            , os.path.join(path, f'{step}.png'))

    def log_reconstructions(self, step, x, pred):
        path = os.path.join(self.p.log_dir, 'images/vae')
        if not os.path.isdir(path):
            os.mkdir(path)
        torchvision.utils.save_image(
            vutils.make_grid(pred, nrow=self.p.num_ims, padding=2, normalize=True)
            , os.path.join(path, f'rec_{step}.png'))
        torchvision.utils.save_image(
            vutils.make_grid(x, nrow=self.p.num_ims, padding=2, normalize=True)
            , os.path.join(path, f'real_{step}.png'))

    def shuffle(self):
        indices = torch.randperm(self.ims.shape[0])
        self.ims = torch.nn.Parameter(torch.index_select(self.ims, dim=0, index=indices.to(self.ims.device)))
        self.labels = torch.index_select(self.labels, dim=0, index=indices.to(self.labels.device))

    def save(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'data.pt')
        if self.p.up:
            ims = self.up(self.ims)
        else:
            ims = torch.tanh(self.ims)
        torch.save(ims.cpu(), file_name)

        file_name = os.path.join(path, 'labels.pt')
        torch.save(self.labels.cpu(), file_name)

    def load_ims(self):
        path = os.path.join(self.p.log_dir, 'checkpoints', 'data.pt')
        if os.path.exists(path):
            self.ims = torch.load(path)
            self.ims = torch.nn.Parameter(self.ims)
        return os.path.exists(path)

    def save_vae(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'vae.pt')
        torch.save(self.vae.state_dict(), file_name)

    def load_vae(self):
        path = os.path.join(self.p.log_dir, 'checkpoints', 'vae.pt')
        if os.path.exists(path):
            self.vae.load_state_dict(torch.load(path))
        return os.path.exists(path)

    def loss(self, x, pred, mu, logvar):
        recon_loss = F.mse_loss(pred, x,reduction='sum')/x.shape[0]
        mu_pow = torch.pow(mu,2)
        var = torch.exp(logvar)
        kl = -0.5 * (1 + logvar - mu_pow - var).sum()/x.shape[0]
        return recon_loss, kl

    def loss_ae(self, x, pred):
        return F.mse_loss(pred, x,reduction='sum')/x.shape[0]

    def train_vae(self):
        print('############## Training VAE ##############',flush=True)
        if self.p.load_vae and self.load_vae():
            print('Loaded existing checkpoint not training again',flush=True)
        else:
            for p in self.vae.parameters():
                p.requires_grad = True
            for t in range(self.p.niter_vae):
                with torch.autocast(device_type=self.p.device, dtype=torch.float16):
                    data, label = next(self.gen)
                    data = data.to(self.p.device)
                    self.vae.zero_grad()
                    if self.p.ae:
                        pred, z = self.vae(data,label)
                        loss = self.loss_ae(data, pred)
                    else:
                        pred, mu, logvar, z = self.vae(data, label)
                        rec, kl = self.loss(data, pred, mu, logvar)
                        loss = rec + self.p.beta * kl
                loss.backward()
                self.opt_vae.step()
                if (t%100) == 0:
                    if self.p.ae:
                        print('[{}|{}] Loss: {:.4f}'.format(t, self.p.niter_vae, loss.item()), flush=True)
                    else:
                        print('[{}|{}] Rec: {:.4f}, KLD: {:.4f}, Loss {:.4f}'.format(t, self.p.niter_vae, rec.item(), kl.item(), loss.item()),flush=True)
                    self.log_reconstructions(t, data, pred)
            self.save_vae()

            for p in self.vae.parameters():
                p.requires_grad = False

    def total_variation_loss(self, img, weight=1, four=True):
        bs_img, c_img, h_img, w_img = img.size()

        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()

        tv = weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

        if four:
            tv_h4 = torch.pow(img[:,:,:-1,:]-img[:,:,1:,:], 2).sum()
            tv_w4 = torch.pow(img[:,:,:,:-1]-img[:,:,:,1:], 2).sum()
            tv = tv + weight*(tv_h4+tv_w4)/(bs_img*c_img*h_img*w_img)
            tv = tv/2

        return tv

    def wasserstein_dist(self, X, Y):
        if X.shape != Y.shape:
            raise ValueError("Expecting equal shapes for X and Y!")
        X = X.squeeze()
        Y = Y.squeeze()
        # the linear algebra ops will need some extra precision -> convert to double
        X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
        mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
        n, b = X.shape
        fact = 1.0 if b < 2 else 1.0 / (b - 1)

        # Cov. Matrix
        E_X = X - mu_X
        E_Y = Y - mu_Y
        cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
        cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

        # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
        # The eigenvalues for M are real-valued.
        C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
        C_Y = E_Y * math.sqrt(fact)
        M_l = torch.matmul(C_X.t(), C_Y)
        M_r = torch.matmul(C_Y.t(), C_X)
        M = torch.matmul(M_l, M_r)
        S = torch.linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
        sq_tr_cov = S.sqrt().abs().sum()

        # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
        trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

        # |mu_X - mu_Y|^2
        diff = mu_X - mu_Y  # [n, 1]
        mean_term = torch.sum(torch.mul(diff, diff))  # scalar

        # put it together
        return (trace_term + mean_term).float()

    def train_ims_cw(self):
        print('############## Training Images ##############',flush=True)
        self.ims.requires_grad = True
        stats = []
        self.vae.eval()

        for p in self.vae.parameters():
                p.requires_grad = False

        for t in range(self.p.niter_ims):
            loss = torch.tensor(0.0).to(self.p.device)

            for c in range(10):
                data, labels = next(self.gen)

                d_c = data[labels == c]
                d_c = d_c[torch.randperm(d_c.shape[0])[:self.p.num_ims]]

                labels = torch.ones(d_c.shape[0], dtype=torch.long)*c
                ims = self.ims[c*self.p.num_ims:(c+1)*self.p.num_ims]

                ## VAE
                if self.p.ae:
                    _, encX = self.vae(d_c.to(self.p.device), labels)
                    encX = encX.detach()
                    rec, encY = self.vae(torch.tanh(ims), labels[:ims.shape[0]])
                    mmd, _, _ = mix_rbf_mmd2_and_ratio(encX, encY, [8, 16, 32, 64])
                    mmd = torch.sqrt(F.relu(mmd))
                    #mmd = torch.norm(encX.mean(dim=0)-encY.mean(dim=0))
                else:
                    _, muX, varX, encX = self.vae(d_c.to(self.p.device), labels)
                    encX = encX.detach()
                    muX, varX = muX.detach(), varX.detach()
                    rec, mu, logvar, encY = self.vae(torch.tanh(ims), labels)
                    mmdMu = mix_rbf_mmd2(muX, mu, [8, 16, 32, 64])
                    mmdMu = torch.sqrt(F.relu(mmdMu))
                    mmdVar = mix_rbf_mmd2(varX, logvar, [8, 16, 32, 64])
                    mmdVar = torch.sqrt(F.relu(mmdVar))
                    mmd = mmdMu + mmdVar

                if self.p.rec:
                    rec_loss, _ = self.loss(torch.tanh(ims), rec, mu, logvar)

                ## Correlation:
                if self.p.corr:
                    corr = self.total_variation_loss(torch.tanh(ims))
                else:
                    corr = torch.zeros(1)

                loss = loss + mmd

                if self.p.corr:
                    loss = loss + self.p.corr_coef*corr

                if self.p.rec:
                    loss = loss + self.p.rec_coef*rec

            self.opt_ims.zero_grad()
            loss.backward()
            self.opt_ims.step()
        
            if (t%100) == 0:
                s = '[{}|{}] Loss: {:.4f}, MMD: {:.4f}'.format(t, self.p.niter_ims, loss.item(), mmd.item())
                if self.p.corr:
                    s += ', Corr: {:.4f}'.format(corr.item())
                if self.p.rec:
                    s += ', Rec: {:.4f}'.format(rec.item())
                print(s,flush=True)
                self.log_interpolation(t)

        self.save()
        self.ims.requires_grad = False

    def train_ims(self):
        print('############## Training Images ##############',flush=True)
        self.ims.requires_grad = True
        stats = []
        self.vae.eval()

        for p in self.vae.parameters():
                p.requires_grad = False

        for t in range(self.p.niter_ims):
            loss = torch.tensor(0.0).to(self.p.device)
            data, labels = next(self.gen)

            perm = torch.randperm(data.shape[0])[:self.p.num_ims*10]
            data = data[perm]
            labels = labels[perm]

            ## VAE
            if self.p.ae:
                _, encX = self.vae(data.to(self.p.device), labels)
                rec, encY = self.vae(torch.tanh(self.ims), self.labels)
            else:
                _, _, _, encX = self.vae(data.to(self.p.device), labels)
                rec, _, _, encY = self.vae(torch.tanh(self.ims), self.labels)

            encX = encX.detach()
            mmd = mix_rbf_mmd2(encX, encY, [8, 16, 32, 64])
            mmd = torch.sqrt(F.relu(mmd))
            loss = loss + mmd

            if self.p.rec:
                rec_loss = self.loss_ae(torch.tanh(self.ims), rec)
                loss = loss + self.p.rec_coef*rec

            ## Correlation:
            if self.p.corr:
                corr = self.total_variation_loss(torch.tanh(self.ims))
                loss = loss + self.p.corr_coef*corr

            self.opt_ims.zero_grad()
            loss.backward()
            self.opt_ims.step()
        
            if (t%100) == 0:
                s = '[{}|{}] Loss: {:.4f}, MMD: {:.4f}'.format(t, self.p.niter_ims, loss.item(), mmd.item())
                if self.p.corr:
                    s += ', Corr: {:.4f}'.format(corr.item())
                if self.p.rec:
                    s += ', Rec: {:.4f}'.format(rec.item())
                print(s,flush=True)
                self.log_interpolation(t)

        self.save()
        self.ims.requires_grad = False
                
    def train_up(self):
        print('############## Training Images ##############',flush=True)
        for p in self.up.parameters():
                p.requires_grad = True
        stats = []
        self.vae.eval()
        self.cl_model.eval()

        for t in range(self.p.niter_ims):
            self.up.zero_grad()
            data, labels = next(self.gen)
            ims = self.up(self.ims)
            ## VAE
            if not self.p.kl:
                _, _, _, encX = self.vae(data.to(self.p.device))
                encX = encX.detach()
                rec, _, _, encY = self.vae(ims)
                mmd = mix_rbf_mmd2(encX, encY, [1, 2, 4, 8, 16, 24, 32, 64])
                mmd = torch.sqrt(F.relu(mmd))
            else:
                pred, mu, logvar, encY = self.vae(ims)
                _, kl = self.loss(ims, pred, mu, logvar)
                kl = kl*0.1

            if self.p.rec:
                pred, mu, logvar, encY = self.vae(ims)
                rec, _ = self.loss(ims, pred, mu, logvar)

            ## Alignment
            pred = self.cl_model(encY.squeeze())
            cl = self.cl_loss(pred,self.labels)


            ## Correlation:
            if self.p.corr:
                corr = self.total_variation_loss(ims)
            else:
                corr = torch.zeros(1)

            loss = cl
            if self.p.kl:
                loss = loss + kl
            else:
                loss = loss + mmd

            if self.p.corr:
                loss = loss + self.p.corr_coef*corr

            if self.p.rec:
                loss = loss + self.p.rec_coef*rec

            loss.backward()

            self.opt_ims.step()
        
            if (t%100) == 0:
                s = '[{}|{}] Loss: {:.4f}, CE: {:.4f}'.format(t, self.p.niter_ims, loss.item(), cl.item())
                if self.p.kl:
                    s += ', KLD: {:.4f}'.format(kl.item())
                else:
                    s += ', MMD: {:.4f}'.format(mmd.item())
                if self.p.corr:
                    s += ', Corr: {:.4f}'.format(corr.item())
                if self.p.rec:
                    s += ', Rec: {:.4f}'.format(rec.item())
                print(s,flush=True)
                self.log_interpolation(t)

        self.save()
        for p in self.up.parameters():
            p.requires_grad = False

    def train(self):
        for p in self.vae.parameters():
                    p.requires_grad = False
        self.ims.requires_grad = False

        self.train_vae()
        #self.train_classifier()
        if self.p.up:
            self.train_up()
        else:
            if self.p.class_wise:
                self.train_ims_cw()
            else:
                self.train_ims()

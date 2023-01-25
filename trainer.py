import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision

import os

from vae import ResVAE
from classifier import Net

class Trainer():
    def __init__(self, params, train_loader):
        self.p = params

        self.train_loader = train_loader
        self.gen = self.inf_train_gen()

        ### VAE
        self.vae = ResVAE(latent_size=self.p.z_dim).to(self.p.device)
        self.opt_vae = torch.optim.Adam(self.vae.parameters(), lr=self.p.lrVAE, weight_decay=1e-5)


        ### Classifier
        self.cl_model = Net(in_dim=self.p.z_dim).to(self.p.device)
        self.opt_cl = torch.optim.Adam(self.cl_model.parameters(), lr=self.p.lrCl)
        self.cl_loss = nn.CrossEntropyLoss()


        ### Synth Ims
        self.ims = torch.randn(10*self.p.num_ims,3,32,32).to(self.p.device)
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

    def log_interpolation(self, step):
        path = os.path.join(self.p.log_dir, 'images/synth')
        if not os.path.isdir(path):
            os.mkdir(path)
        torchvision.utils.save_image(
            vutils.make_grid(torch.tanh(self.ims), nrow=self.p.num_ims, padding=2, normalize=True)
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
        torch.save(torch.tanh(self.ims.cpu()), file_name)

        file_name = os.path.join(path, 'labels.pt')
        torch.save(self.labels.cpu(), file_name)

    def save_vae(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'vae.pt')
        torch.save(self.vae.state_dict(), file_name)

    def load_vae(self):
        path = os.path.join(self.p.log_dir, 'checkpoints', 'vae.pt')
        if os.path.exists(path):
            self.vae = self.vae.load_state_dict(torch.load(path))
        return os.path.exists(path)

    def save_classifier(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'classifier.pt')
        torch.save(self.cl_model.state_dict(), file_name)

    def load_classifier(self):
        path = os.path.join(self.p.log_dir, 'checkpoints', 'classifier.pt')
        if os.path.exists(path):
            self.cl_model = self.cl_model.load_state_dict(torch.load(path))
        return os.path.exists(path)

    def loss(self, x, pred, mu, logvar):
        recon_loss = F.mse_loss(pred, x,reduction='sum')/x.shape[0]
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()/x.shape[0]
        return recon_loss, kl

    def train_vae(self):
        print('############## Training VAE ##############')
        if self.load_vae():
            print('Loaded existing checkpoint not training again')
        else:
            for p in self.vae.parameters():
                p.requires_grad = True
            for t in range(self.p.niter_vae):
                data, _ = next(self.gen)
                data = data.to(self.p.device)
                self.vae.zero_grad()
                pred, mu, logvar, z = self.vae(data)
                rec, kl = self.loss(data, pred, mu, logvar)
                loss = rec + self.p.beta * kl
                loss.backward()
                self.opt_vae.step()
                if (t%100) == 0:
                    print('[{}|{}] Rec: {:.4f}, KLD: {:.4f}, Loss {:.4f}'.format(t, self.p.niter_vae, rec.item(), kl.item(), loss.item()))
                    self.log_reconstructions(t, data, pred)
            self.save_vae()

            for p in self.vae.parameters():
                p.requires_grad = False

    def train_classifier(self):
        print('############## Training Classifier ##############')
        if self.load_classifier():
            print('Loaded existing checkpoint not training again')
        else:
            for p in self.cl_model.parameters():
                p.requires_grad = True
            for t in range(self.p.niter_cl):
                data, label = next(self.gen)
                data = data.to(self.p.device)
                label = label.to(self.p.device)
                _, _, _, z = self.vae(data)
                self.cl_model.zero_grad()
                pred = self.cl_model(z.squeeze())
                loss = self.cl_loss(pred, label.long())
                loss.backward()
                self.opt_cl.step()
                pred = pred.argmax(dim=1, keepdim=True)
                correct = pred.eq(label.view_as(pred)).sum().item()

                if (t%100) == 0:
                    print('[{}|{}] CE: {:.4f}, Acc: {:.4f}'.format(t, self.p.niter_cl, loss.item(), 100. * correct / data.shape[0]))

            self.save_classifier()

            for p in self.cl_model.parameters():
                p.requires_grad = False

    def train_ims(self):
        print('############## Training Images ##############')
        self.ims.requires_grad = True
        stats = []
        with torch.no_grad():
            for i, (x,y) in enumerate(self.train_loader):
                _, _, _, z = self.vae(x.to(self.p.device))
                stats.append(z.squeeze())

            stats = torch.cat(stats)
            stats = stats.mean(dim=0)

        for t in range(self.p.niter_ims):
            self.opt_ims.zero_grad()
            _, _, _, z = self.vae(torch.tanh(self.ims))
            z = z.squeeze()
            pred = self.cl_model(z)
            mmd = torch.norm(stats - z.mean(dim=0))
            cl = self.cl_loss(pred,self.labels)
            loss = mmd + cl
            loss.backward()

            self.opt_ims.step()
        
            if (t%100) == 0:
                print('[{}|{}] Loss: {:.4f}, MMD: {:.4f}, CE: {:.4f}'.format(t, self.p.niter_ims, loss.item(), mmd.item(), cl.item()))
                self.log_interpolation(t)

        self.save()
        self.ims.requires_grad = False

    def train(self):
        for p in self.vae.parameters():
                    p.requires_grad = False
        for p in self.cl_model.parameters():
                    p.requires_grad = False
        self.ims.requires_grad = False

        self.train_vae()
        self.train_classifier()
        self.train_ims()

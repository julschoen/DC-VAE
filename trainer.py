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

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        
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
            self.vae.load_state_dict(torch.load(path))
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
            self.cl_model.load_state_dict(torch.load(path))
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

    def wasserstein_dist(self, X, Y):
        '''
        Calulates the two components of the 2-Wasserstein metric:
        The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
        For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
        this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
        Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
        Input shape: [b, n] (e.g. batch_size x num_features)
        Output shape: scalar
        '''

        if X.shape != Y.shape:
            raise ValueError("Expecting equal shapes for X and Y!")

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
        S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
        sq_tr_cov = S.sqrt().abs().sum()

        # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
        trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

        # |mu_X - mu_Y|^2
        diff = mu_X - mu_Y  # [n, 1]
        mean_term = torch.sum(torch.mul(diff, diff))  # scalar

        # put it together
        return (trace_term + mean_term).float()

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
        if not self.p.kl:
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
            if self.p.kl:
                # input should be a distribution in the log space
                zs = F.log_softmax(z, dim=1)
                # Sample a batch of distributions. Usually this would come from the dataset
                data, label = next(self.gen)
                data = data.to(self.p.device)
                label = label.to(self.p.device)
                _,_,_, d_z = self.vae(data)
                d_zs = F.softmax(d_z.squeeze(), dim=1)
                mmd = self.wasserstein_dist(zs, d_zs)
            else:
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

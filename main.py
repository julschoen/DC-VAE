import argparse
import torch
from torchvision import datasets, transforms
from trainer import Trainer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='DC-VAE')
    ### General
    parser.add_argument('--batch_size', type=int, default= 512, help='input batch size for training (default: 512)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--tsne', type=bool, default=False)

    ### VAE
    parser.add_argument('--niter_vae', type=int, default=3000)
    parser.add_argument('--lrVAE', type=float, default=1e-4)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--ae', type=bool, default=False)

    ### Synth Images
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--niter_ims', type=int, default=1000)
    parser.add_argument('--lrIms', type=float, default=1e-3)
    parser.add_argument('--rec', type=bool, default=False)
    parser.add_argument('--rec_coef', type=float, default=0.1)
    parser.add_argument('--corr', type=bool, default=True)
    parser.add_argument('--corr_coef', type=float, default=0.1)
    parser.add_argument('--class_wise', type=bool, default=False)
    parser.add_argument('--up', type=bool, default=False)

    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle':True}


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                        transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    trainer = Trainer(args, train_loader)
    trainer.train()
    if args.tsne:
        zs = []
        ys = []
        with torch.no_grad():
            for i, (x,y) in enumerate(train_loader):
                if args.ae:
                    _, z = trainer.vae(x.cuda(), y)
                else:
                    _, _, _, z = trainer.vae(x.cuda(), y)
                zs.append(z.detach().cpu())
                ys.append(y.detach().cpu())

        if args.ae:
            _, z_ims = trainer.vae(torch.tanh(trainer.ims), trainer.labels)
        else:
            _, _, _, z_ims = trainer.vae(torch.tanh(trainer.ims), trainer.labels)
        zs.append(z_ims.detach().cpu())
        ys.append(trainer.labels.detach().cpu())

        zs = torch.cat(zs).squeeze()
        ys = torch.cat(ys)
        tsne = TSNE(n_components=2, learning_rate='auto',
                 init='random', perplexity=30)

        X_embedded = tsne.fit_transform(zs.numpy())
        X_embedded.shape

        X, Y = X_embedded[:-100], ys[:-100]
        x, y = X_embedded[-100:], ys[-100:]

        fig = plt.figure(figsize=(16,14))
        plt.scatter(X[:,0], X[:,1], c=Y.numpy(), cmap='Set1')
        plt.scatter(x[:,0], x[:,1], cmap='black')
        plt.colorbar()
        plt.savefig('all_vae.pdf', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(40,30))
        for ind in range(10):
            plt.subplot(3,4,ind+1)
            plt.scatter(X[ind == Y,0], X[ind == Y,1], cmap='b', label='real')
            plt.scatter(x[ind == y,0], x[ind == y,1], cmap='r', label='embedded')
            plt.title(f'Class {ind}')
            plt.legend()
        plt.savefig('classes_vae.pdf', bbox_inches='tight')
        plt.close(fig)
    

if __name__ == '__main__':
    main()
import argparse
import torch
from torchvision import datasets, transforms
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='DC-VAE')
    ### General
    parser.add_argument('--batch_size', type=int, default= 512, help='input batch size for training (default: 512)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')

    ### VAE
    parser.add_argument('--niter_vae', type=int, default=15000)
    parser.add_argument('--lrVAE', type=float, default=1e-4)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1)

    ### Classifier
    parser.add_argument('--niter_cl', type=int, default=2000)
    parser.add_argument('--lrCl', type=float, default=1e-3)


    ### Synth Images
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--niter_ims', type=int, default=50000)
    parser.add_argument('--lrIms', type=float, default=1e-4)
    parser.add_argument('--kl', type=bool, default=False)
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
    

if __name__ == '__main__':
    main()
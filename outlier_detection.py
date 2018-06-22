"""
The script for doing outlier detection using different score
"""
import argparse
from model import VAE
from loss import VAELoss
from dataloader import load_vae_test_datasets
import os
import torch
from utilities import evaluateVAE

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--image_size', default=256, type=int,
                    help='transformed image size, has to be power of 2')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--print_freq', default=10)
args = parser.parse_args()

# load checkpoint
if not os.path.isfile(args.model_path):
    print('%s is not path to a file' % args.model_path)
    exit()
checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
print("checkpoint loaded!")
print("val loss: {}\tepoch: {}\t".format(checkpoint['val_loss'], checkpoint['epoch']))

# model and criterion
model = VAE(args.image_size)
model.load_state_dict(checkpoint['state_dict'])
criterion = VAELoss(size_average=True)

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# load data
test_loader = load_vae_test_datasets(args.image_size, args.data)

with torch.no_grad():
    avg_normal_loss, avg_abnormal_loss = evaluateVAE(test_loader=test_loader,
                                                     model=model,
                                                     criterion=criterion,
                                                     args=args)
print("test avg loss normal {}\ttest avg loss abnormal {}".format(avg_normal_loss, avg_abnormal_loss))


# define different way of computing score
def reconstruct_score(vae, image, L=5):
    """
    The reconstruct score for a single image
    :param image: [1, 3, 256, 256]
    """
    # Do a parallel run by repeat image for L times
    # [L, 3, 256, 256]
    image_batch = image.expand(L,
                               image.size(1),
                               image.size(2),
                               image.size(3))
    reconst, mu, logvar = vae.forward(image_batch)



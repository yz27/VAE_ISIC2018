import argparse
from model import VAE
from VAELoss import VAELoss
from dataloader import load_vae_train_datasets, load_vae_test_datasets
import os
import torch
from utilities import evaluateVAE, validateVAE

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
_, val_loader = load_vae_train_datasets(args.image_size, args.data, args.batch_size)
test_loader = load_vae_test_datasets(args.image_size, args.data)

with torch.no_grad():
    avg_normal_loss, avg_abnormal_loss = evaluateVAE(test_loader=test_loader,
                                                     model=model,
                                                     criterion=criterion,
                                                     args=args)
    val_loss, _, _ = validateVAE(val_loader=val_loader,
                                 model=model,
                                 criterion=criterion,
                                 args=args)
print("test avg loss normal {}\ttest avg loss abnormal {}\tval avg loss normal {}".format(
    avg_normal_loss, avg_abnormal_loss, val_loss))


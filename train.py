import os
import torch
import argparse
from VAELoss import VAELoss
from utilities import trainVAE, validateVAE, load_opt
from model import VAE
from dataloader import load_datasets
from tensorboardX import SummaryWriter

# import ipdb

parser = argparse.ArgumentParser(description='skin disease classification')
parser.add_argument('--data', metavar='DIR', help='path to dataset', type=str)

# for models
parser.add_argument('--image_size', default=256, type=int,
                    help='transformed image size, has to be power of 2')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

# for optimization
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--base_lr', default=1e-4, type=float)
parser.add_argument('--fine_tune', action='store_true',
                    help='if fine tune then use base_lr for the CNN tower.')
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--l2_decay', default=1e-4, type=float,
                    help="l2 weight decay")
parser.add_argument('--lr_decay', default=0.1, type=float,
                    help='learning rate decay')
parser.add_argument('--schedule', type=int, nargs='+', default=[45, ],
                    help='Decrease learning rate at these epochs.')

# for checkpoint loading
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--reset_opt', action='store_true',
                    help='if true then we do not load optimizer')

# for logging
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--out_dir', default='./result', type=str,
                    help='output result for tensorboard and model checkpoint')

args = parser.parse_args()
if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

model = VAE(args.image_size)

# load data
train_loader, val_loader = load_datasets(args.image_size, args)

# load criterion
criterion = VAELoss(size_average=False)
if args.cuda is True:
    model = model.cuda()
    criterion = criterion.cuda()


# load optimizer and scheduler
top_params = [p for p in model.parameters()]
base_params = []
for key, _ in (model._modules.items()):
    if key != 'decoder':
        base_params += [p for p in model._modules[key].parameters()]
opt, scheduler = load_opt(args, base_params, top_params)

# make output dir
if os.path.isdir(args.out_dir):
    print("{} already exists!".format(args.out_dir))
os.mkdir(args.out_dir)

# save args
args_dict = vars(args)
with open(os.path.join(args.out_dir, 'config.txt'), 'w') as f:
    for k in args_dict.keys():
        f.write("{}:{}\n".format(k, args_dict[k]))
writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'logs'))


def main():
    """
    Main Loop
    """
    # Set initial best loss
    best_loss = 30000

    for epoch in range(args.epochs):
        # train for one epoch
        scheduler.step()
        # ipdb.set_trace()
        trainVAE(train_loader, model, criterion, opt, epoch, writer, args)

        # evaluate on validation set
        with torch.no_grad():
            val_loss = validateVAE(val_loader, model, criterion, writer, args)

        # remember best acc and save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_dict = {'epoch' : epoch + 1,
                         'state_dict' : model.state_dict(),
                         'val_loss' : val_loss,
                         'optimizer' : opt.state_dict()}
            save_path = os.path.join(args.out_dir, 'best_model.pth.tar')
            torch.save(save_dict, save_path)


if __name__ == '__main__':
    main()

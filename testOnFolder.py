import torch
import os
import visdom
import argparse
from torchvision.utils import save_image
from model import VAE
from torchvision import transforms, datasets
from PIL import Image
from VAELoss import VAELoss

parser = argparse.ArgumentParser(description='skin disease classification')
parser.add_argument('--data', metavar='DIR', help='path to dataset', type=str)
args = parser.parse_args()

checkpoint = torch.load('./AKIEC_result/best_model.pth.tar')
input_size = 256
model = VAE(input_size)
criterion = VAELoss(size_average=False)

# top_params = [p for p in model._modules['fc4'].parameters()]
# base_params = []
# for key, _ in (model._modules.items()):
#     if (key != 'fc4'):
#         base_params += [p for p in model._modules[key].parameters()]

model.load_state_dict(checkpoint['state_dict'])
# device = torch.device("cuda")
device = torch.device("cpu")
vis = visdom.Visdom()
vis.env = 'vae'

# with torch.no_grad():
#     sample = torch.randn(64, 100, 1, 1).to(device)
#     sample = model.decode(sample).cpu()
#     save_image(sample.view(64, 3, 256, 256),
#                'sample.png')
valdir = os.path.join(args.data, 'vae_train/train/')

val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])

val_dataset = datasets.ImageFolder(valdir, val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             shuffle=True, num_workers=1,
                                             pin_memory=True)
true_win = None
recon_win = None
if __name__ == '__main__':
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output
            recon_batch, mu, logvar = model(input)
            recon_loss, kl_loss = criterion(recon_batch, input, mu, logvar)
            loss = recon_loss + kl_loss
            vis.image((input.data.view(3, 256, 256)+1)/2, win=true_win, opts=dict(title=f'true'))
            vis.image((recon_batch.data.view(3, 256, 256)+1)/2, win=recon_win, opts=dict(title=f'recon(loss:{loss})'))
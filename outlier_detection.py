"""
The script for doing outlier detection using different score
"""
import argparse
from model import VAE
from loss import VAELoss
from dataloader import load_vae_test_datasets
import os
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from itertools import product
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--cuda', action='store_true')
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

############################# ANOMALY SCORE DEF ##########################
def get_reconst_score(vae, image, L=5):
    """
    The reconstruct score for a single image
    :param image: [1, 3, 256, 256]
    :return scocre: the reconstruct score
    """
    # Do a parallel run by repeat image for L times
    # [L, 3, 256, 256]
    image_batch = image.expand(L,
                               image.size(1),
                               image.size(2),
                               image.size(3))
    reconst_batch, _, _ = vae.forward(image_batch)
    scores = torch.sum(0.5 * (reconst_batch - image_batch).pow(2),
                              dim=-1)
    return torch.mean(scores)

def get_vae_score(vae, image, L=5):
    """
    The vae score for a single image, which is basically the loss
    :param image: [1, 3, 256, 256]
    :return scocre: the reconstruct score
    """
    image_batch = image.expand(L,
                               image.size(1),
                               image.size(2),
                               image.size(3))
    reconst_batch, mu, logvar = vae.forward(image_batch)
    vae_loss, _ = criterion(reconst_batch, image_batch, mu, logvar)
    return vae_loss

def _log_mean_exp(x, dim):
    """
    A numerical stable version of log(mean(exp(x)))
    :param x: The input
    :param dim: The dimension along which to take mean with
    """
    # m [dim1, 1]
    m, _ = torch.max(x, dim=dim, keepdim=True)

    # x0 [dm1, dim2]
    x0 = x - m

    # m [dim1]
    m = m.squeeze(dim)

    return m + torch.log(torch.mean(torch.exp(x0),
                                    dim=dim))

def get_iwae_score(vae, image, L=5, K=5):
    """
    The vae score for a single image, which is basically the loss
    :param image: [1, 3, 256, 256]
    :return scocre: the reconstruct score
    """
    image_batch = image.expand(L*K,
                               image.size(1),
                               image.size(2),
                               image.size(3))
    reconst_batch, mu, logvar = vae.forward(image_batch)

    # [L*K]
    vae_losses = criterion.forward_without_reduce(reconst_batch,
                                                  image_batch,
                                                  mu,
                                                  logvar)
    # [L, K]
    vae_losses = vae_losses.reshape(L, K)

    # [L]
    scores = _log_mean_exp(vae_losses, dim=1)
    return torch.mean(scores)

############################# END OF ANOMALY SCORE ###########################

# Define the number of samples of each score
def compute_all_scores(vae, image):
    """
    Given an image compute all anomaly score
    return (reconst_score, vae_score, iwae_score)
    """
    result = {'reconst_score': get_reconst_score(vae=vae, image=image, L=4).item(),
              'vae_score': get_vae_score(vae=vae, image=image, L=4).item(),
              'iwae_score': get_iwae_score(vae=vae, image=image, L=2, K=2).item(),}
    return result


# MAIN LOOP
score_names = ['reconst_score', 'vae_score', 'iwae_score']
classes = test_loader.dataset.classes
scores = {(score_name, cls): [] for (score_name, cls) in product(score_names,
                                                                 classes)}
model.eval()
with torch.no_grad():
    for idx, (image, target) in tqdm(enumerate(test_loader)):
        cls = classes[target.item()]
        if args.cuda:
            image = image.cuda()

        score = compute_all_scores(vae=model, image=image)
        for name in score_names:
            scores[(name, cls)].append(score[name])

# get auc roc remove. remove the NV class
classes.remove('NV')
auc_result = np.zeros([len(score_names), len(classes)])
for (name, cls) in product(score_names, classes):
    normal_scores = scores[(name, 'NV')]
    abnormal_scores = scores[(name, cls)]
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_result[score_names.index(name), classes.index(cls)] = roc_auc_score(y_true, y_score)
df = pd.DataFrame(auc_result, index=score_names, columns=classes)
# display
print("###################### AUC ROC #####################")
print(df)
print("####################################################")










"""
The script for doing outlier detection using a fitted gaussian and z-score
"""
import argparse
from model import VAE
from loss import VAELoss
from dataloader import load_vae_test_datasets, load_vae_train_datasets
import os
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--batch_size', default=64, type=int)
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

# load val data
_, val_loader = load_vae_train_datasets(args.image_size,
                                        args.data,
                                        args.batch_size)


# get all loss
model.eval()
loss_avg = 0
loss_sq_avg = 0
total_num = float(len(val_loader))
with torch.no_grad():
    for (imgs, _) in val_loader:
        if args.cuda:
            imgs = imgs.cuda()

        recons, mu, logvar = model.forward(imgs)

        # [bsz]
        losses = criterion.forward_without_reduce(recons,
                                                  imgs,
                                                  mu,
                                                  logvar)
        loss_avg += torch.sum(losses).item() / total_num
        loss_sq_avg += torch.sum(losses.pow(2)).item() / total_num

sample_mean = loss_avg
sample_var = (total_num / (total_num - 1)) * (loss_sq_avg - loss_avg**2)
print("estimated done!")

# get through all test data
test_loader = load_vae_test_datasets(args.image_size, args.data)
classes = test_loader.dataset.classes
z_scores = {cls: [] for cls in classes}
model.eval()
with torch.no_grad():
    for idx, (img, target) in tqdm(enumerate(test_loader)):
        cls = classes[target.item()]

        if args.cuda:
            img = img.cuda()

        recons, mu, logvar = model.forward(img)
        loss, _ = criterion(recons, img, mu, logvar)

        z_score = (loss.item() - sample_mean)**2 / sample_var
        z_scores[cls].append(z_score)

# get auc roc remove. remove the NV class
classes.remove('NV')
auc_result = np.zeros([1, len(classes)])
for cls in classes:
    normal_scores = z_scores['NV']
    abnormal_scores = z_scores[cls]
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_result[0, classes.index(cls)] = roc_auc_score(y_true, y_score)
df = pd.DataFrame(auc_result, index=['z_score'], columns=classes)
# display
print("###################### AUC ROC #####################")
print(df)
print("####################################################")

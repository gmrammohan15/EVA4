
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    crit = nn.BCEWithLogitsLoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks , bgimg = batch['fgbg_image'], batch['mask'], batch['bg_image']
                        
            ###
            imgs = torch.cat((imgs, bgimg), dim=1)

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                # pred = torch.sigmoid(mask_pred)
                # pred = (pred > 0.5).float()
                # tot += dice_coeff(pred, true_masks).item()
                tot += crit(mask_pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val
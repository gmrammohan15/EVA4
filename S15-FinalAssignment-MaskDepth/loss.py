import torch
import torch.nn as nn
import torch.nn.functional as F
epsilon = 1e-6
epsilon_sqr = epsilon ** 2

def compute_reprojection_loss(pred, target, ssim=False):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    reprojection_loss = l1_loss
    if ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def compute_losses(inputs, outputs):
    """Compute the reprojection and smoothness losses for a minibatch
    """
    losses = {}
    total_loss = 0
    loss = 0
    #reprojection_losses = []
    target_mask = inputs["mask"]
    pred = outputs["predictive_mask"]["disp", 0]

    pred = F.interpolate(
        pred, [64, 64], mode="bilinear", align_corners=False)


    #outputs["predictive_mask"] = pred

    #reprojection_losses.append(compute_reprojection_loss(pred, target_mask))
    reprojection_losses = compute_reprojection_loss(pred, target_mask)
    #reprojection_losses = torch.cat(reprojection_losses, 1)

    mask = outputs["predictive_mask"]["disp", 0]

    mask = F.interpolate(mask, [64, 64], mode="bilinear", align_corners=False)
    # print(reprojection_losses.shape)
    # print(mask.shape)         
    reprojection_losses *= mask

    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
    loss += weighting_loss.mean()

    loss += reprojection_losses.mean()
    losses["loss"] = loss

    return losses

def SSIM(self, x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def alpha_prediction_loss(y_pred, y_true):
    mask = y_true[:, 1, :]
    diff = y_pred[:, 0, :] - y_true[:, 0, :]
    diff = diff * mask
    num_pixels = torch.sum(mask)
    return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / (num_pixels + epsilon)
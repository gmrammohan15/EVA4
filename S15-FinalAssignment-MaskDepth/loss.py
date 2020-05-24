import torch.nn.functional as F

def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    reprojection_loss = l1_loss
    # if self.opt.no_ssim:
    #     reprojection_loss = l1_loss
    # else:
    #     ssim_loss = self.ssim(pred, target).mean(1, True)
    #     reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

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
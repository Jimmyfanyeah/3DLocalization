# other loss
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np

# calculates the jaccard coefficient approximation using per-voxel probabilities
def jaccard_coeff(pred, target):
    """
    jaccard index = TP / (TP + FP + FN)
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # smoothing parameter
    smooth = 1e-6
    
    # number of examples in the batch
    N = pred.size(0)

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(N,-1)
    tflat = target.contiguous().view(N,-1)
    intersection = (iflat * tflat).sum(1)
    jacc_index = (intersection / (iflat.sum(1) + tflat.sum(1) - intersection + smooth)).mean()

    return jacc_index



def recall(pred, target):
    smooth = 1e-4
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - (intersection + smooth) / ( torch.sum(tflat) + smooth)


def recall_loss(pred, target):
    smooth = 1e-3
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    B_sum = torch.sum(tflat * tflat)

    return 1 - (intersection + smooth) / ( B_sum + smooth)

class calculate_loss(nn.Module):
    def __init__(self, scaling_factor,device):
        super(calculate_loss, self).__init__()
        self.scaling_factor = scaling_factor
        self.criterion = KDE_loss3D(self.scaling_factor)
        self.zero = torch.FloatTensor([0.0]).to(device)
        self.unit = torch.FloatTensor([1.0]).to(device)

    def forward(self, pred, target, metric, metrics):

        # only calc loss for 3d location, pred in [0,800] target in 0/1
        kde = self.criterion(pred, target)
        # dice = dice_loss(pred/self.scaling_factor,target)
    
        # Dice loss cannot be in back-propagation
        target_dice = torch.where(target > self.zero, self.unit, self.zero)
        dice = dice_loss(pred/self.scaling_factor,target_dice)
        reg_loss = reg(pred/self.scaling_factor, 1000)

        loss = kde

        metric['dice'] = dice.data.cpu().numpy()
        metric['kde'] = kde.detach().cpu().numpy()
        metric['reg_loss'] = reg_loss.detach().cpu().numpy()
        metric['loss'] = loss.data.cpu().numpy()

        metrics['Dice'] += dice.detach().clone() * target.size(0)
        metrics['KDE'] += kde.detach().clone() * target.size(0)
        metrics['Reg'] += reg_loss.detach().clone() * target.size(0)
        metrics['Loss'] += loss.detach().clone() * target.size(0)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()












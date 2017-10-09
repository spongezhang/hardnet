import torch
import torch.nn as nn

def pairwise_distance(x1, x2, p=2, eps=1e-6):
    r"""
    Computes the batchwise pairwise distance between vectors v1,v2:
        .. math ::
            \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}
        Args:
            x1: first input tensor
            x2: second input tensor
            p: the norm degree. Default: 2
        Shape:
            - Input: :math:`(N, D)` where `D = vector dimension`
            - Output: :math:`(N, 1)`
        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.pairwise_distance(input1, input2, p=2)
        >>> output.backward()
    """
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.sum(torch.pow(diff + eps, p),dim=1)
    return torch.pow(out, 1. / p)

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, anchor.size(0)) + torch.t(d2_sq.repeat(1, positive.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)
    #return torch.sqrt((d1_sq.repeat(1, anchor.size(0)) + torch.t(d2_sq.repeat(1, positive.size(0)))
    #                  - 2.0 * torch.mm(anchor, torch.t(positive)))+eps)

def loss_margin_min(anchor, positive, anchor_swap = False, anchor_ave = False, margin = 1.0, alpha = 1.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."

    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask

    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    if anchor_swap:
        min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
        min_neg = torch.min(min_neg,min_neg2)

    #min_neg = torch.t(min_neg)
    dist_hinge = torch.clamp(margin + pos - min_neg, min=0.0)

    if anchor_ave:
        min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
        #min_neg2 = torch.t(min_neg2).squeeze(0)
        dist_hinge2 = torch.clamp(1.0 + pos - min_neg2, min=0.0)
        dist_hinge = 0.5 * (dist_hinge2 + dist_hinge)

    loss = torch.mean(dist_hinge) + alpha*torch.mean(torch.pow(pos,2))
    return loss

def loss_margin_min_t(anchor, positive, negative, anchor_swap = False, anchor_ave = False, margin = 1.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."

    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask

    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    if anchor_swap:
        min_neg2 = torch.t(torch.min(dist_without_min_on_diag,0)[0])
        min_neg = torch.min(min_neg,min_neg2)
    min_neg = torch.t(min_neg).squeeze(0)
    dist_hinge = torch.clamp(margin + pos - min_neg, min=0.0)

    if anchor_ave:
        min_neg2 = torch.t(torch.min(dist_without_min_on_diag,0)[0])
        min_neg2 = torch.t(min_neg2).squeeze(0)
        dist_hinge2 = torch.clamp(1.0 + pos - min_neg2, min=0.0)
        dist_hinge = 0.5 * (dist_hinge2 + dist_hinge)

    neg_dis = torch.pow(torch.sum(torch.mul(anchor,negative),1),2)
    gor = torch.mean(neg_dis)

    loss = torch.mean(dist_hinge)

    return loss, gor


def loss_margin_min_gor(anchor, positive, negative, anchor_swap = False, anchor_ave = False, margin = 1.0, alpha = 1.0, beta = 0.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."

    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008)-1)*-1
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask

    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    if anchor_swap:
        min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
        min_neg = torch.min(min_neg,min_neg2)

    #min_neg = torch.t(min_neg)
    dist_hinge = torch.clamp(margin + pos - min_neg, min=0.0)

    if anchor_ave:
        min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
        #min_neg2 = torch.t(min_neg2).squeeze(0)
        dist_hinge2 = torch.clamp(1.0 + pos - min_neg2, min=0.0)
        dist_hinge = 0.5 * (dist_hinge2 + dist_hinge)

    neg_dis = torch.pow(torch.sum(torch.mul(anchor,negative),1),2)
    gor = torch.mean(neg_dis)

    loss = torch.mean(dist_hinge) + alpha*torch.mean(torch.pow(pos,2)) + beta*gor
    return loss, gor


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    d_p = pairwise_distance(anchor, positive, p, eps)
    d_n = pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)
    loss = torch.mean(dist_hinge)

    neg_dis = torch.pow(torch.sum(torch.mul(anchor,negative),1),2)
    gor = torch.mean(neg_dis)

    return loss


def triplet_margin_loss_gor(anchor, positive, negative, beta = 1.0, margin=1.0, p=2, eps=1e-6, swap=False):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    d_p = pairwise_distance(anchor, positive, p, eps)
    d_n = pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)

    neg_dis = torch.pow(torch.sum(torch.mul(anchor,negative),1),2)
    gor = torch.mean(neg_dis)
    
    loss = torch.mean(dist_hinge) + beta*gor
    
    return loss, gor


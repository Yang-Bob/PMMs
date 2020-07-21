
import torch.optim as optim
import numpy as np

def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    pretrain_weight_list = []
    pretrain_bias_list =[]
    for name,value in model.named_parameters():
        if 'model_res' in name or 'model_backbone' in name:
            if 'weight' in name:
                pretrain_weight_list.append(value)
            elif 'bias' in name:
                pretrain_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    opt = optim.SGD([{'params': pretrain_weight_list, 'lr':lr},
                     {'params': pretrain_bias_list, 'lr':lr*2},
                     {'params': weight_list, 'lr':lr*10},
                     {'params': bias_list, 'lr':lr*20}], momentum=0.90, weight_decay=0.0005) # momentum = 0.99

    return opt

def adjust_learning_rate_poly(args, optimizer, iter, power=0.9):
    base_lr = args.lr
    max_iter = args.max_steps
    reduce = ((1-float(iter)/max_iter)**(power))
    lr = base_lr * reduce
    optimizer.param_groups[0]['lr'] = lr * 1
    optimizer.param_groups[1]['lr'] = lr * 2
    optimizer.param_groups[2]['lr'] = lr * 10
    optimizer.param_groups[3]['lr'] = lr * 20


import os
import torch

def restore(args, model):

    group = args.group
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(group, args.num_folds))
    if args.restore_step =='best':
        filename = '%s.pth.tar' % (args.restore_step)
    else:
        filename='step_%d.pth.tar'%(args.restore_step)

    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s'%(snapshot))

def get_model_para_number(model):
    total_number = 0
    for para in model.parameters():
        total_number += torch.numel(para)

    return total_number

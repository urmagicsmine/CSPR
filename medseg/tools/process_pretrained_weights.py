import torch
import os
from collections import OrderedDict
"""
This script is used to convert imagenet/video pretrained weights to fit one-channel medical input.
And convert gpu state_dict to cpu mode.
Specifically, it convert (out, 3, (d), w, h) weights to (out, 1, (d), w, h) with sum in the first dims

"""

def main(model_path, process_name='conv1.weight'):
    checkpoint = torch.load(model_path, map_location=None)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # Remove module. prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # Sum 3 channel weights
    if state_dict['conv1.weight'].shape[1] == 3:
        shape_list = list(state_dict['conv1.weight'].shape)
        shape_list[1] = 1
        state_dict['conv1.weight'] = torch.sum(state_dict['conv1.weight'], dim=1).view(shape_list)
    # TO cpu
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        value = value.cpu()
        new_state_dict[key] = value
    file_name, ext_name = model_path.split('.')
    save_name = '.'.join([file_name + '_med1channel', ext_name])
    torch.save(new_state_dict, save_name)


if __name__ == '__main__':
    #model_path = '/lung_general_data/pretrained_model/kinetics/resnet-18-kinetics.pth'
    model_path = '/lung_general_data/pretrained_model/kinetics/r3d18_KM_200ep.pth'
    #model_path = 'resnet_18_23dataset.pth'
    main(model_path)


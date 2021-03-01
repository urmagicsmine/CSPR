import argparse
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument('--remove_backbone', help='remove \'backbone\' for item in checkpoint[\'state_dict\']', action='store_true')
    args = parser.parse_args()
    return args

def rm_backbone(checkpoint):
    state_dict = checkpoint['state_dict']
    state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if 'backbone' in k}
    checkpoint['state_dict'] = state_dict
    print(state_dict.keys())
    return checkpoint

def process_checkpoint(in_file, out_file, remove_backbone=False):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if remove_backbone:
        checkpoint = rm_backbone(checkpoint)
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file, args.remove_backbone)


if __name__ == '__main__':
    main()

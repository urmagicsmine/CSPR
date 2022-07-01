#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tempfile import NamedTemporaryFile
import torch
try:
    from sklearn.externals import joblib
except ImportError:
    # sklearn was 0.19.1, sklearn.externals is deprecated in 0.21 
    import joblib
import pickle
import sys
import traceback
import pdb


def torch_load_general(input_file, map_location=lambda storage, loc: storage):
    try:
        return torch.load(input_file, map_location=map_location)
    except UnicodeDecodeError:
        if int(torch.__version__.split(".")[1]) <= 1:
            import pickle
            from functools import partial
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(input_file, map_location=map_location, pickle_module=pickle)
            return checkpoint
        else:
            checkpoint = torch.load(input_file, map_location=map_location, encoding="bytes")
            return {k.decode():v for k,v in checkpoint.items()}
            


def clean_state_dict(state_dict, replace_torchvison=False, revise=False):

    def clean_key(key, replace_torchvison=False):
        if isinstance(key, bytes):
            key = key.decode()

        if replace_torchvison:
            k = key
            k = k.replace('norm.1', 'norm1')
            k = k.replace('relu.1', 'relu1')
            k = k.replace('conv.1', 'conv1')
            k = k.replace('norm.2', 'norm2')
            k = k.replace('relu.2', 'relu2')
            k = k.replace('conv.2', 'conv2')
            key = k

        # if key.startswith('module.feature_net'):
        #     prefix_len = len('module.feature_net')
        #     key = key[prefix_len:]
        if key.startswith('module.'):
            key = key.partition('module.')[2]
        return key

    pretrained_dict = {clean_key(k, replace_torchvison=replace_torchvison): v
                       for k, v in state_dict.items()}

    if revise:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if 'num_batches_tracked' not in k:
                new_state_dict[k] = v
        pretrained_dict = new_state_dict

    return pretrained_dict


def decode_seq(hex_seq):
    return bytes(hex_seq)


def LoadTorchWeights(input_file, gpu_id=-1, mode="pytorch"):
    key_string = b"Deepwise!@#$1234"
    iv_string = b"0000333311112222"
    gpu_id = int(gpu_id)
    input_file = str(input_file)
    try:
        if os.path.exists(input_file):
            # load original weights
            if mode is "sklearn":
                return joblib.load(input_file)
            elif mode is "pickle":
                with open(input_file, "rb") as f:
                    return pickle.load(f)
            else:
                if gpu_id >= 0:
                    return torch_load_general(input_file, map_location=lambda storage, loc: storage.cuda(int(gpu_id)))
                else:
                    return torch_load_general(input_file, map_location=torch.device('cpu'))

        else:
            from Crypto.Cipher import AES
            # load encrypted weights
            input_file += ".dw"
            decryptor = AES.new(key_string, AES.MODE_CBC, iv_string)
            tmp_file = None
            if int(sys.version[0]) < 3:
                tmp_file = NamedTemporaryFile(delete=False, mode="w")
                encrypted_seq = decode_seq(open(input_file).read())
                use_seq = decryptor.decrypt(encrypted_seq)
                tmp_file.write(use_seq)
            else:
                tmp_file = NamedTemporaryFile(delete=False, mode="wb")
                encrypted_seq = open(input_file, "rb").read()
                use_seq = decryptor.decrypt(encrypted_seq)
                tmp_file.write(use_seq)
            tmp_file.close()

            print("decrypt file:%s" % input_file)

            decrypted_dict = None
            if mode is "sklearn":
                decrypted_dict = joblib.load(tmp_file.name)
            elif mode is "pickle":
                with open(tmp_file.name, "rb") as f:
                    return pickle.load(f)
            else:
                if gpu_id >= 0:
                    decrypted_dict = torch_load_general(
                        tmp_file.name, map_location=lambda storage, loc: storage.cuda(gpu_id))
                else:
                    decrypted_dict = torch_load_general(
                        tmp_file.name, map_location=torch.device('cpu'))
            os.unlink(tmp_file.name)
            return decrypted_dict

    except Exception as ex:
        print(traceback.format_exc())
        return None
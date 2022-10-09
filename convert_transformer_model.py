import torch

state_dict = torch.load('F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/pytorch_model.bin')
state_dict_new = {}
for k, v in state_dict.items():
    if 'LayerNorm.gamma' in k:
        k = k.replace('LayerNorm.gamma', 'LayerNorm.weight')
        state_dict_new[k] = v
    elif 'LayerNorm.beta' in k:
        k = k.replace('LayerNorm.beta', 'LayerNorm.bias')
        state_dict_new[k] = v
    else:
        state_dict_new[k] = v
torch.save(state_dict_new, 'F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/bert4torch_pytorch_model.bin')

import os
import sys
import torch
import random
import argparse
import numpy as np
import time
import torch
from torchvision.models import resnet50	
# from thop import profile,clever_format	
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
# from torchstat import stat
import torchprof
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import get_config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

from torchinfo import summary

# thop 使用示例 （只能全模型统计，不能细粒度统计）
def thop_profile_example():
    model = resnet50()	
    input = torch.randn(1, 3, 224, 224)	
    flops, params = profile(model, inputs=(input, ),verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs)
    print(params)

# torch profile example 从prof.events()里面可以获得每层的详细信息
def torch_profile_example():
    model = resnet50().cuda()
    inputs = torch.randn(1, 3, 224, 224).cuda()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        with record_function("model_inference"):
            model(inputs)
    # print(prof.key_averages().table())
    print(model)
    # prof.export_stacks("profiler_stacks.txt")
    
def profile_model():
    model = resnet50().cuda()
    inputs = torch.randn(1, 3, 224, 224).cuda()
    with torchprof.Profile(model, use_cuda=True) as prof:
        model(inputs)
    print(prof.display(show_events=False))

def gpt2_profile_time(model_name='117M'):
    model_path = os.path.join('pretrained_models', model_name, 'model.bin')
    state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
    config = get_config(model_name)
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('model running device: ',device)
    model.to(device)
    model.eval()
    enc = get_encoder()
    sample_text = "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
    context_tokens = enc.encode(sample_text)
    # print(context_tokens)
    # print(len(context_tokens))
    # with torchprof.Profile(model, use_cuda=True) as prof:
        # with record_function("model_inference"):
    # time1 = time.time()
    out = sample_sequence(
        model=model, length=config.n_ctx // 2,
        context=context_tokens,
        start_token=None,
        batch_size=4,
        temperature=0.7, top_k=40, device=device
    )

    # time2 = time.time()
    # print(time2-time1)
    with open('time_profile1_2.txt','w') as f:
        print(prof.display(show_events=False), file=f)
    # trace, event_lists_dict = prof.raw()
    # with open('time_profile2.txt','w') as f:
    #     print(trace, file=f)
    # with open('time_profile3.txt','w') as f:
    #     print(event_lists_dict, file=f)

def gpt2_profile_size(model_name='117M'):
    model_path = os.path.join('pretrained_models', model_name, 'model.bin')
    state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
    config = get_config(model_name)
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('model running device: ',device)
    model.to(device)
    model.eval()
    summary(model,[(1, 64),(1,64)],dtypes=[torch.int, torch.int],depth=5)
    # summary(model,[(1, 64),(1,64)],dtypes=[torch.int, torch.int],depth=5,
    #         col_names=["kernel_size", "output_size", "num_params", "mult_adds"])


gpt2_profile_time('1542M')

# out = sample_sequence(
#     model=model, length=config.n_ctx // 2,
#     context=context_tokens,
#     start_token=None,
#     batch_size=4,
#     temperature=0.7, top_k=40, device=device
# )
# stat(model, )
# with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as prof:
#     # with record_function("model_inference"):
#     out = sample_sequence(
#         model=model, length=config.n_ctx // 2,
#         context=context_tokens,
#         start_token=None,
#         batch_size=4,
#         temperature=0.7, top_k=40, device=device
#     )




# print(prof.key_averages().table())
# # with open('profile.txt', 'w') as f:
# #     f.write(prof.events())
# print(prof.events()[20:22])
# print(len(prof.events()))
# out = out[:, len(context_tokens):].tolist()
# text = enc.decode(out[0])
# print(text)

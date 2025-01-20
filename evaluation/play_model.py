import os
import time
import csv
import math
from contextlib import nullcontext
from io import StringIO
import chess

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from ..net.model import HC3
from ..net.utility import HC3Config, ChessDataset, HC3DataHandler

@torch.no_grad()
def sample(max_time, inc, W_elo, B_elo): # make sure to put in eval mode and train mode after
    board = chess.Board()
    W_time = max_time
    B_time = max_time
    context = ['?', '?', '?', '?'] # list of previous moves made
    repetition = 'F'
    counter = 0
    while not board.is_game_over() and counter <= 100:
        counter += 1
        inp = handler.input_from_board(board, max_time, inc, W_time, B_time, W_elo, B_elo, context, repetition).to(device)

        with ctx:
            out, _ = model(inp) # yes or no context?

        legal_moves = [move.uci() for move in board.legal_moves]
        if not board.turn: # if black, then mirror legal moves
            legal_moves = [handler.mirror_move(m) for m in legal_moves]
        
        # print(legal_moves)
        move, speed, eval_str = handler.sample_output(out, is_white_move=board.turn, legal_moves=legal_moves, move_strategy='greedy')

        if board.turn == chess.WHITE:
            W_time = W_time - speed + inc 
            if W_time < inc: # you time out before getting increment
                print('white ran out of time')
                return '0-1'
        else: # black turn
            B_time = B_time - speed + inc 
            if B_time < inc:
                print('black ran out of time')
                return '1-0'
        
        chess_move = chess.Move.from_uci(move) # convert to correct format
        board.push(chess_move)
        print(counter, move, speed, eval_str)

        # update context
        context = [context[1], context[2], context[3], move]

        # repetition check
        repetition = 'F'
        for m in board.legal_moves:
            board.push(m)
            if board.is_repetition(3):
                repetition = 'T'
                board.pop()
                break
            board.pop()

    return board.result()

model_name = 'HC3_8.8.256.256.16.128' # HC3
out_dir = 'out'
eval_interval = 5000 # change to 5000 for official run
log_interval = 10 # change to 10 for official run
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'out/HC3_8.8.256.256.16.128_ckpt.pt'
# init_from = '/kaggle/input/chess-runs2/out/HC2_12_legal_epoch1.pt' 
# 'scratch', 'resume', or path of model to finetune
logging = 'log.csv'

# data
data_dir = '/home/ubuntu/chess-files/data/'
dataset_train = 'data_tr.csv'
dataset_val = 'data_val.csv' # same for now
gradient_accumulation_steps = 4 # used to simulate larger batch sizes
batch_size = 512 # if gradient_accumulation_steps > 1, this is the micro-batch size

# model
block_size = 64
move_size = 2048
speed_size = 8
input_dim = 80

n_layer = 8
n_head = 8
n_embd = 256
hl = 256
smolgen = True
sg_hidden1 = 8
sg_hidden2 = 128
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?


# adamw optimizer
learning_rate = 5e-4 # max learning rate
max_iters = 1000000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 1000000 # should be ~= max_iters per Chinchilla
min_lr = 5e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = "float16" # bfloat16 runs into errors with T4 GPU on colab during compilation
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


model_args = dict(block_size=block_size, move_size=move_size, speed_size=speed_size, input_dim=input_dim,
                  n_layer=n_layer, n_head=n_head, n_embd=n_embd, hl=hl, smolgen=smolgen, sg_hidden1=sg_hidden1, 
                  sg_hidden2=sg_hidden2, dropout=dropout, bias=bias)

# DataHandler
handler = HC3DataHandler()


print(f"Finetuning model {init_from}")
checkpoint = torch.load(init_from, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'move_size', 'hl']: # TODO: update
    model_args[k] = checkpoint_model_args[k]
# create the model
conf = HC3Config(**model_args)
model = HC3(conf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)


model.eval() 
print(sample(300, 0, 2100, 2100))
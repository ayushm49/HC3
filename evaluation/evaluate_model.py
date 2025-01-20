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

# -----------------------------------------------------------------------------
# default config values
# I/O
model_name = 'HC3_8.8.256.256.16.128' # HC3
out_dir = 'out'
eval_interval = 5000 # change to 5000 for official run
log_interval = 10 # change to 10 for official run
eval_iters = 5000
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'out/HC3_8.8.256.256.16.128_ckpt.pt'
# init_from = '/kaggle/input/chess-runs2/out/HC2_12_legal_epoch1.pt' 
# 'scratch', 'resume', or path of model to finetune
logging = 'eval_log.csv'

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
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = "float16" # bfloat16 runs into errors with T4 GPU on colab during compilation
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

compile = True # use PyTorch 2.0 to compile the model to be faster
sample = True # sample model during evaluation? NOTE: does not work with DDP training afaik because of randomness 
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# -----------------------------------------------------------------------------

# torch setup 
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading 
cached_x = None # global variable

def custom_collate_fn(batch): # batch is a list of x, y tuples
    global handler, cached_x
    x, y = zip(*batch)
    cached_x = x
    return handler.input_from_list(x), handler.output_from_list(y) # nonblocking stuff???? 


print('Setting up val data...')
data_val = ChessDataset(os.path.join(data_dir, dataset_val))

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
epoch = 0
best_val_loss = 1e9

# -----------------------------------------------------------------------------

# model init
model_args = dict(block_size=block_size, move_size=move_size, speed_size=speed_size, input_dim=input_dim,
                  n_layer=n_layer, n_head=n_head, n_embd=n_embd, hl=hl, smolgen=smolgen, sg_hidden1=sg_hidden1, 
                  sg_hidden2=sg_hidden2, dropout=dropout, bias=bias)

# DataHandler
handler = HC3DataHandler()

# Set up eval logging file with the header
with open(logging, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    header = ['game_id', 'move_num', 'player_to_move', 'max_time', 'increment', 'P1_elo', 'P2_elo', 
                'P1_time', 'P2_time', 'P1_material_diff', 'P1_material_count', 
                'move_agreement', 'speed_agreement', 'origin_agreement', 'target_agreement', 'outcome_agreement']
    csvwriter.writerow(header)

if init_from == 'scratch':
    # init a new model from scratch
    print("Evaluating an untrained model")
    conf = HC3Config(**model_args)
    model = HC3(conf)
else: # get a model but get rid of training metadata
    print(f"Evaluating model {init_from}")
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

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from != 'scratch':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# -----------------------------------------------------------------------------

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    model.eval()
    tloader_val = DataLoader(data_val, batch_size=batch_size, collate_fn=custom_collate_fn)
    tval_iter = iter(tloader_val) # temporary val data iterator
    val_losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        X, Y = next(tval_iter)
        if device.startswith('cuda'):
            X = X.pin_memory().to(device, non_blocking=True)
        else: 
            X = X.to(device)
        Y.to(device)
        with ctx:
            output, loss_tuple = model(X, Y)
        val_losses[k] = loss_tuple[0].item() # total loss
    model.train()
    return val_losses.mean()

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
        move, speed, eval_str = handler.sample_output(out, is_white_move=board.turn, legal_moves=legal_moves, move_strategy='random')

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

def get_accuracy(t1, t2): 
    # Both should be (B, N)
    if t1.shape != t2.shape:
        raise ValueError("Both tensors must have the same shape:", t1.shape, t2.shape)

    argmax1 = torch.argmax(t1, dim=1)
    argmax2 = torch.argmax(t2, dim=1)
    
    # Compare argmax results and count agreements
    agreement = (argmax1 == argmax2).sum().item()

    return agreement / t1.shape[0]

@torch.no_grad()
def full_eval(X, Y, output):
    # For each batch, log input conditions (eg. elo, time, etc.) and agreement on 5 tasks 
    
    # get input conditions in a list: []
    a = get_inp_conditions(cached_x, X) # (B, 11)
    # get agreement on the following: [move, speed, origin, target, outcome]
    move_agreement = (torch.argmax(Y.next_move, dim=1) == torch.argmax(output.next_move, dim=1)) # (B, 1)
    speed_agreement = (torch.argmax(Y.move_speed, dim=1) == torch.argmax(output.move_speed, dim=1)) # (B, 1)
    origin_agreement = (torch.argmax(Y.origin, dim=1) == torch.argmax(output.origin, dim=1)) # (B, 1)
    target_agreement = (torch.argmax(Y.target, dim=1) == torch.argmax(output.target, dim=1)) # (B, 1)
    outcome_agreement = (torch.argmax(Y.outcome, dim=1) == torch.argmax(output.outcome, dim=1)) # (B, 1)
    
    b = torch.stack([move_agreement, speed_agreement, origin_agreement, target_agreement, outcome_agreement], dim=1).cpu().numpy()
    
    result = np.concatenate((np.array(a), b), axis=1)
    return result


def get_inp_conditions(x, X):
    # given a list of csv rows (x) and processed input tensor (X)
    # [game_id, move_num, player_to_move, max_time, increment, P1_elo, P2_elo, P1_time, P2_time, P1_material_diff, P1_material_count ]
    out = []
    for k in range(len(x)): # num batches
        o = [0]*11 
        board = chess.Board(x[k][1]) # get board from fen
        o[0] = x[k][0] # game_id 
        o[1] = board.fullmove_number # move_num
        o[2] = 'W' if board.turn else 'B'
        o[3] = x[k][3]
        o[4] = x[k][4]
        if board.turn: # white is P1 
            o[5] = x[k][7] # elo 
            o[6] = x[k][8]
            o[7] = x[k][5] # time 
            o[8] = x[k][6]
        else: # black is P1
            o[5] = x[k][8] # elo
            o[6] = x[k][7]
            o[7] = x[k][6] # time 
            o[8] = x[k][5]
        o[9] = (X[k,0,19].item()*8 + X[k,0,20].item()*5 + X[k,0,21].item()*3 + 
                X[k,0,22].item()*3 + X[k,0,23].item()) # P1_material_diff
        o[10] = (X[k,0,25].item()*8 + X[k,0,26].item()*5 + X[k,0,27].item()*3 + 
                 X[k,0,28].item()*3 + X[k,0,29].item()) # P1_material count
        
        out.append(o)
    return out # B, 11
        

# -----------------------------------------------------------------------------

model.eval()

tloader_val = DataLoader(data_val, batch_size=batch_size, collate_fn=custom_collate_fn)
tval_iter = iter(tloader_val) # temporary val data iterator
val_losses = torch.zeros(eval_iters)
move_rates = torch.zeros(eval_iters) # move accuracy
speed_rates = torch.zeros(eval_iters) # speed accuracy
origin_rates = torch.zeros(eval_iters) # origin accuracy
target_rates = torch.zeros(eval_iters) # target accuracy
outcome_rates = torch.zeros(eval_iters) # outcome accuracy

print('Beginning Evaluation...')
with torch.no_grad():
    for k in range(eval_iters):
        if k%50 == 0: 
            print(k)
        X, Y = next(tval_iter)
        if device.startswith('cuda'):
            X = X.pin_memory().to(device, non_blocking=True)
        else: 
            X = X.to(device)
        Y.to(device)
        with ctx:
            output, loss_tuple = model(X, Y)
        
        rows = full_eval(X, Y, output)

        with open(logging, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)

        val_losses[k] = loss_tuple[0].item() # total loss
        move_rates[k] = get_accuracy(output.next_move, Y.next_move)
        speed_rates[k] = get_accuracy(output.move_speed, Y.move_speed)
        origin_rates[k] = get_accuracy(output.origin, Y.origin)
        target_rates[k] = get_accuracy(output.target, Y.target)
        outcome_rates[k] = get_accuracy(output.outcome, Y.outcome)


print('Total Loss:', val_losses.mean() )
print('Accuracy Rate for Move Choice:', move_rates.mean() )
print('Accuracy Rate for Move Speed:', speed_rates.mean() )
print('Accuracy Rate for Move Origin:', origin_rates.mean() )
print('Accuracy Rate for Move Target:', target_rates.mean() )
print('Accuracy Rate for Game Outcome:', outcome_rates.mean() )

print("Evaluation Done!")


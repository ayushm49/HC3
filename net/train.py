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

from model import HC3
from utility import HC3Config, ChessDataset, HC3DataHandler

# -----------------------------------------------------------------------------
# default config values
# I/O
model_name = 'HC3_8.8.256.256.16.128' # HC3
out_dir = 'out'
eval_interval = 5000 # change to 5000 for official run
log_interval = 10 # change to 10 for official run
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch'
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

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# -----------------------------------------------------------------------------

# torch setup 
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# logging setup
log_path = os.path.join(out_dir, logging)

# data loading 
def custom_collate_fn(batch): # batch is a list of x, y tuples
    global handler
    x, y = zip(*batch)
    return handler.input_from_list(x), handler.output_from_list(y) # nonblocking stuff???? 

if ddp:
    torch.distributed.barrier()

print('Setting up train data...')
data_tr = ChessDataset(os.path.join(data_dir, dataset_train))
print('Setting up val data...')
data_val = ChessDataset(os.path.join(data_dir, dataset_val))

print("Setting up loaders...")
if ddp:
    train_sampler = DistributedSampler(data_tr, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
    loader = DataLoader(
        dataset=data_tr,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
else:
    train_sampler = None  # No sampler for single-GPU training
    loader = DataLoader(
        dataset=data_tr,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False, 
        num_workers=4
    )

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

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    conf = HC3Config(**model_args)
    model = HC3(conf)
    if ddp:
        train_sampler.set_epoch(0)
    data_iter = iter(loader)
    if master_process: 
        with open(log_path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['epoch','iter','total_loss', 'next_move_loss', 'legal_moves_loss', 'origin_loss', 
                        'target_loss', 'move_speed_loss', 'outcome_loss', 'time', 'mfu'])
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, model_name + '_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'move_size', 'hl']:
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
    iter_num = checkpoint['iter_num']
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    ep = math.ceil(len(data_tr) / (ddp_world_size * batch_size * gradient_accumulation_steps))
    if ddp:
        train_sampler.set_epoch(epoch)
    else:
        pass
    start = iter_num % ep # calculate how far into data_iter we were 
    data_iter = iter(loader)
    for _ in range(start): 
        next(data_iter) # roll through iter until we reach where we were in the epoch 
    # not exactly a continuation, but maintains the appropriate epoch

else: # get a model but get rid of training metadata
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
    if ddp:
        train_sampler.set_epoch(0)
    data_iter = iter(loader)

ep = math.ceil(len(data_tr) / (ddp_world_size * batch_size * gradient_accumulation_steps)) # number of iterations per epoch
print("Epoch length:", ep, "iters") 

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

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    tloader_tr = DataLoader(data_tr, batch_size=batch_size, collate_fn=custom_collate_fn)
    tloader_val = DataLoader(data_val, batch_size=batch_size, collate_fn=custom_collate_fn)
    ttr_iter = iter(tloader_tr) # temporary train data iterator
    tval_iter = iter(tloader_val) # temporary val data iterator
    tr_losses = torch.zeros(eval_iters)
    val_losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        X, Y = next(ttr_iter)
        if device.startswith('cuda'):
            X = X.pin_memory().to(device, non_blocking=True)
        else: 
            X = X.to(device)
        Y.to(device)
        with ctx:
            output, loss_tuple = model(X, Y)
        tr_losses[k] = loss_tuple[0].item() # total loss

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
    return tr_losses.mean(), val_losses.mean()

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# saves model and training state 
def save_state(name: str, best_val_loss): 
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'config': config
    }
    print(f"saving checkpoint to {os.path.join(out_dir, name)}")
    torch.save(checkpoint, os.path.join(out_dir, name))

# gets batch for train split
def get_batch():
    global data_iter, epoch
    try:
        X, Y = next(data_iter)
    except StopIteration:
        epoch += 1
        if ddp:
            train_sampler.set_epoch(epoch)  # Ensure the sampler shuffles consistently across processes
            torch.distributed.barrier()  # Ensure all processes synchronize
        if master_process:
            save_state(model_name + f'_epoch{epoch}.pt', estimate_loss()[0])
        data_iter = iter(loader)
        X, Y = next(data_iter)
    if device.startswith('cuda'):
        X = X.pin_memory().to(device, non_blocking=True)
    else: 
        X = X.to(device)
    Y.to(device)
    return X, Y

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

# -----------------------------------------------------------------------------

X, Y = get_batch()
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

print("Beginning training loop:")

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        tr_loss, val_loss = estimate_loss()
        print(f"epoch {epoch}, step {iter_num}: tr_loss {tr_loss:.4f}, " + 
              f"val_loss {val_loss:.4f}")
        if sample: 
            model.eval()
            print(sample(300, 0, 2100, 2100)) # print result of game as well
            model.train()
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss # this is the priority 
            if iter_num > 0:
                save_state(model_name + '_ckpt.pt', best_val_loss) 
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            output, loss_tuple = model(X, Y)

            loss = loss_tuple[0] / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch()
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if master_process: 
        # log to file 
        l1 = [epoch, iter_num, loss.item()]
        l2 = [t.item() for t in loss_tuple[1]]
        l3 = [dt*1000, running_mfu*100]
        l = l1 + l2 + l3
        with open(log_path, 'a') as f:
            w = csv.writer(f)
            w.writerow(l)
    
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        l = loss_tuple[1]
        print(f"epoch {epoch}, iter {iter_num}: loss {lossf:.4f}, next_move {l[0].item():.4f}, " + 
              f"legal_moves {l[1].item():.4f}, origin {l[2].item():.4f}, " + 
              f"target {l[3].item():.4f}, move_speed {l[4].item():.4f}, " + 
              f"outcome {l[5].item():.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

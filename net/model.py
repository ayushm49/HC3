import math
import inspect
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utility import HC3Output, HC3Config

### Model Building Blocks
class LayerNorm(nn.Module):
    # LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, config: HC3Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            #                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att += mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config: HC3Config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.hl, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.hl, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config: HC3Config):
        super().__init__()
        self.config = config
        self.smolgen = config.smolgen
        self.n_head = config.n_head

        if self.smolgen:
            self.sg_lin1 = nn.Linear(config.n_embd, config.sg_hidden1, bias=False) # (B, T, n_embd) -> (B, T, sg_hidden1)
            self.sg_lin2 = nn.Linear(config.sg_hidden1 * config.block_size, config.sg_hidden2, bias=True) # (B, T * sg_hidden1) -> (B, sg_hidden2)
            self.sg_ln1 = LayerNorm(config.sg_hidden2, bias=config.bias) # compressed board representation
            self.sg_lin3 = nn.Linear(config.sg_hidden2, config.sg_hidden2 * config.n_head, bias=True) # (B, sg_hidden2) -> (B, n_head * sg_hidden2)
            self.sg_ln2 = LayerNorm(config.sg_hidden2, bias=config.bias) # done for each head separately
            self.sg_lin4 = nn.Linear(config.sg_hidden2, config.block_size ** 2, bias=True) # (B, sg_hidden2) -> (B, block_size^2) # share across all layers
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        B, T, C = x.shape
        if self.smolgen:
            sg = self.sg_lin1(x) # (B, T, sg_hidden1)
            sg = sg.view(B, -1) # (B, T * sg_hidden1)
            sg = self.sg_lin2(sg) # (B, sg_hidden2)
            sg = self.sg_ln1(sg) # (B, sg_hidden2)
            sg = self.sg_lin3(sg) # (B, n_head * sg_hidden2)
            sg = sg.view(B, self.n_head, -1) # (B, n_head, sg_hidden2)
            sg = self.sg_ln2(sg) # (B, n_head, sg_hidden2)
            sg = self.sg_lin4(sg) # (B, n_head, block_size^2), attention biases
            sg = sg.view(B, self.n_head, self.config.block_size, self.config.block_size) # (B, n_head, block_size, block_size)
        else: 
            sg = None
        x_norm = self.ln_1(x)
        x = x + self.attn(x_norm, sg)
        x_norm = self.ln_2(x)
        x = x + self.mlp(x_norm)
        return x


class OutputHead(nn.Module):
    def __init__(self, config: HC3Config, output_dim, bias=None):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, config.block_size)) # (1, block_size)
        self.linear = nn.Linear(config.n_embd, output_dim, bias=bias)

    def forward(self, x):
        B, T, C = x.shape
        weights = F.softmax(self.weights, dim=-1)  # Normalize weights, (1, block_size) 
        weighted_avg = torch.bmm(weights.unsqueeze(0).expand(B, -1, -1), x).squeeze(1)  # B, C
        output = self.linear(weighted_avg)  # B, output_dim
        return output



### Model
class HC3(nn.Module):

    def __init__(self, config: HC3Config):
        super().__init__()
        self.config = config

        # embedding
        self.square_embed = nn.Linear(config.input_dim, config.n_embd, bias=True)
        self.pos_embed = nn.Embedding(64, config.n_embd)

        # blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias) # final layer norm

        # output heads 
        self.move_head = OutputHead(config, config.move_size, bias=False) # predict next move
        self.legal_head = OutputHead(config, config.move_size, bias=False) # predict legal moves
        self.origin_head = OutputHead(config, 64, bias=False) # predict origin of next move
        self.target_head = OutputHead(config, 64, bias=False) # predict target of next move
        self.speed_head = OutputHead(config, config.speed_size, bias=False) # predict move speed
        self.outcome_head = OutputHead(config, 3, bias=False) # predict outcome

        # init weights
        self.apply(self._init_weights)

        # manually tie smolgen final layer weights to the first block
        for block in self.blocks:
            if hasattr(block, 'sg_lin4'):
                block.sg_lin4.weight = self.blocks[0].sg_lin4.weight
                block.sg_lin4.bias = self.blocks[0].sg_lin4.bias


        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, inp: torch.Tensor, targets: HC3Output = None): 
        # embedding
        B, T, C = inp.shape # B, 64, 80

        x = self.square_embed(inp) # (B, 64, n_embd)
        p = torch.arange(0, 64, dtype=torch.long, device=inp.device) # 0 to 63
        pos = self.pos_embed(p)
        x = x + pos # absolute PE for chess squares

        # transformer blocks 
        for block in self.blocks: 
            x = block(x)
        x = self.ln_f(x) # (B, 64, n_embd)

        next_move = self.move_head(x) # (B, move_size)
        legal_moves = self.legal_head(x) # (B, move_size)
        origin = self.origin_head(x) # (B, 64)
        target = self.target_head(x) # (B, 64)
        move_speed = self.speed_head(x) # (B, 8)
        outcome = self.outcome_head(x) # (B, 3)

        output = HC3Output(next_move, origin, target, legal_moves, outcome, move_speed)

        loss_tuple = None if targets is None else self.get_loss(output, targets)

        return output, loss_tuple

    def get_num_params(self):
        # Return the number of parameters in the model, including embedding.
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_loss(self, model_out: HC3Output, target_out: HC3Output):
        # next_move 
        model_logprobs = F.log_softmax(model_out.next_move, dim=-1)
        target_probs = F.softmax(target_out.next_move, dim=-1)
        next_move_loss = F.kl_div(model_logprobs, target_probs, reduction='batchmean')  # KL divergence loss

        # legal_moves
        model_logprobs = F.log_softmax(model_out.legal_moves, dim=-1)
        target_probs = F.softmax(target_out.legal_moves, dim=-1)
        legal_moves_loss = F.kl_div(model_logprobs, target_probs, reduction='batchmean')  # KL divergence loss

        # origin
        model_logprobs = F.log_softmax(model_out.origin, dim=-1)
        target_probs = F.softmax(target_out.origin, dim=-1)
        origin_loss = F.kl_div(model_logprobs, target_probs, reduction='batchmean')  # KL divergence loss

        # target
        model_logprobs = F.log_softmax(model_out.target, dim=-1)
        target_probs = F.softmax(target_out.target, dim=-1)
        target_loss = F.kl_div(model_logprobs, target_probs, reduction='batchmean')  # KL divergence loss

        # move_speed
        model_logprobs = F.log_softmax(model_out.move_speed, dim=-1)
        target_probs = F.softmax(target_out.move_speed, dim=-1)
        move_speed_loss = F.kl_div(model_logprobs, target_probs, reduction='batchmean')  # KL divergence loss

        # outcome
        model_logprobs = F.log_softmax(model_out.outcome, dim=-1)
        target_probs = F.softmax(target_out.outcome, dim=-1)
        outcome_loss = F.kl_div(model_logprobs, target_probs, reduction='batchmean')  # KL divergence loss

        total_loss = next_move_loss + legal_moves_loss + origin_loss + target_loss + move_speed_loss + outcome_loss

        return total_loss, [next_move_loss, legal_moves_loss, origin_loss, target_loss, move_speed_loss, outcome_loss]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
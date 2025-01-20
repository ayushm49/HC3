# class for homemade engine in lichess-bot 

# add these imports to homemade.py
import torch
import time
import sys

sys.path.append("/Users/ayushmishra/Documents/HC3") # whatever the location of the HC3 folder is...
from net.model import HC3
from net.utility import HC3DataHandler, HC3Config

# add this class to homemade.py
class HC3Engine(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.handler = HC3DataHandler()
        self.context = ['?', '?', '?', '?']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        init_from = '/Users/ayushmishra/documents/HC3/out/HC3_8.8.256.256.16.128_ckpt.pt' # saved state file
        checkpoint = torch.load(init_from, map_location=self.device)
        
        model_args = dict(block_size=64, move_size=2048, speed_size=8, input_dim=80,
                  n_layer=8, n_head=8, n_embd=256, hl=256, smolgen=True, sg_hidden1=8, 
                  sg_hidden2=128, dropout=0.0, bias=False)
        conf = HC3Config(**model_args)
        self.model = HC3(conf)
        
        state_dict = checkpoint['model']

        unwanted_prefix = '_orig_mod.' # fix the keys of the state dictionary
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval() 

        self.W_elo = 2100
        self.B_elo = 2100

        self.default_time = 300
        self.max_time = self.default_time
        self.inc = 0

        print("Initialization Done.")
    
    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> PlayResult:

        # print(time_limit)
        t0 = time.time()

        last_4_moves = list(board.move_stack)[-4:]
        self.context = ['?']*4 + [move.uci() for move in last_4_moves]
        self.context = self.context[-4:] # crop out the '?'

        if not board.turn: # black to move
            self.context = [self.handler.mirror_move(m) for m in self.context] # mirror context


        print(self.context)

        if isinstance(time_limit.time, float):
            W_time = time_limit.time
            B_time = time_limit.time

            self.max_time = self.default_time
            self.inc = 0
            timing = False
        else:
            timing = True
            if len(board.move_stack) < 4:
                self.max_time = time_limit.white_clock
            W_time = time_limit.white_clock if isinstance(time_limit.white_clock, float) else self.default_time
            B_time = time_limit.white_clock if isinstance(time_limit.black_clock, float) else self.default_time
            self.inc = time_limit.white_inc if isinstance(time_limit.white_inc, float) else 0

        repetition = 'F' # CURRENTLY DOES NOT HANDLE REPETITION!!!

        inp = self.handler.input_from_board(board, self.max_time, self.inc, W_time, B_time, self.W_elo, self.B_elo, self.context, repetition).to(self.device)

        # forward pass
        out, _ = self.model(inp)
        move_final, speed_final, eval_str = self.handler.sample_output(out, is_white_move=board.turn)
        move = chess.Move.from_uci(move_final)
        if move not in board.legal_moves:
            print("Illegal Move Attempted:", move) 
            move = chess.Move.from_uci( move_final )

        # eval 
        print(eval_str)
        # timing
        t1 = time.time()
        # dt = t1 - t0 # in ms
        if timing:
            time_to_stall = speed_final - (time.time() - t0)
            if time_to_stall > 0: 
                print('Waiting', time_to_stall)
                time.sleep(time_to_stall)
            
        
        return PlayResult(move=move,ponder=None)
        

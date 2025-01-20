import os
from dataclasses import dataclass
import chess
import mmap
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass
class HC3Config:
    block_size: int = 64 # num tokens in attention. 64
    move_size: int = 2048 # num possible moves
    speed_size: int = 8 # num possible move speeds {[-0.5,0.5), [0.5,1.5),  [1.5,2.5), [2.5,4.5), [4.5,9.5), [9.5,19.5), [19.5, 59.5), [60, inf) }
    input_dim: int = 80 # input dimensionality of each square
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    hl: int = 256 # hidden layer in FFN
    smolgen: bool = True # use smolgen?
    sg_hidden1: int = 16 # first compression dimension, ignored if smolgen is False
    sg_hidden2: int = 128 # final compression dimension, ignored if smolgen is False
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms


@dataclass
class HC3Output: 
    next_move: torch.Tensor # shape = (B, move_size)
    origin: torch.Tensor # shape = (B, 64)
    target: torch.Tensor # shape = (B, 64)
    legal_moves: torch.Tensor # shape = (B, move_size)
    outcome: torch.Tensor # shape = (B, 3)
    move_speed: torch.Tensor # shape = (B, 8)

    def to(self, device: str):
        # send to specified device
        if device.startswith('cuda'):
            self.next_move = self.next_move.pin_memory().to(device, non_blocking=True)
            self.origin = self.origin.pin_memory().to(device, non_blocking=True)
            self.target = self.target.pin_memory().to(device, non_blocking=True)
            self.legal_moves = self.legal_moves.pin_memory().to(device, non_blocking=True)
            self.outcome = self.outcome.pin_memory().to(device, non_blocking=True)
            self.move_speed = self.move_speed.pin_memory().to(device, non_blocking=True)
        else:
            self.next_move = self.next_move.to(device)
            self.origin = self.origin.to(device)
            self.target = self.target.to(device)
            self.legal_moves = self.legal_moves.to(device)
            self.outcome = self.outcome.to(device)
            self.move_speed = self.move_speed.to(device)



class HC3DataHandler: 
    # Handles inputs and outputs for HC3 model (eg. conversions from list to tensor)
    def __init__(self):
        moves = self.get_all_moves()
        squares = self.get_squares() 
        self.init_move_dicts(moves)
        self.init_square_dicts(squares)

    def input_from_list(self, inp: list[list]):
        B = len(inp)
        ts = []
        for i in range(B):
            t1, is_white_move = self.from_fen(inp[i][1]) # array of shape (64, 31)
            t2 = self.context_to_tensor(inp[i][9:13], is_white_move) # array of shape (64, 8)
            t3 = np.zeros((64,41))

            # repetition
            if inp[i][2] == 'T':
                t3[:,0] = 1
            else:
                t3[:,0] = 0

            # bins 
            max_time_bins = [65, 125, 185, 305, 605] # 6
            inc_bins = [0.5, 1.5, 3.5] # 4
            elo_bins = [1200, 1400, 1600, 1800, 2000, 2200] # 7 * 2 = 14
            time_bins = [5, 10, 20, 30, 60, 120, 240] # 8 * 2 = 16
            if is_white_move:
                P1_elo = np.digitize(inp[i][7], bins=elo_bins)
                P2_elo = np.digitize(inp[i][8], bins=elo_bins)
                P1_time = np.digitize(inp[i][5], bins=time_bins)
                P2_time = np.digitize(inp[i][6], bins=time_bins)
            else:
                P1_elo = np.digitize(inp[i][8], bins=elo_bins)
                P2_elo = np.digitize(inp[i][7], bins=elo_bins)
                P1_time = np.digitize(inp[i][6], bins=time_bins)
                P2_time = np.digitize(inp[i][5], bins=time_bins)
            timing = np.digitize(inp[i][3], bins=max_time_bins)
            inc = np.digitize(inp[i][4], bins=inc_bins)

            # order: timing, inc, P1_time, P2_time, P1_elo, P2_elo
            one_hot = np.concatenate([
                np.eye(len(max_time_bins) + 1)[timing],
                np.eye(len(inc_bins) + 1)[inc],
                np.eye(len(time_bins) + 1)[P1_time],
                np.eye(len(time_bins) + 1)[P2_time],
                np.eye(len(elo_bins) + 1)[P1_elo],
                np.eye(len(elo_bins) + 1)[P2_elo]
            ], axis=-1)  # Concatenate one-hot encodings

            one_hot_broadcasted = np.broadcast_to(one_hot, (64, one_hot.shape[-1]))  # Broadcast to (64, X)
            t3[:, 1:] = one_hot_broadcasted  # Fill the rest of t3 with the broadcasted array

            t = np.concatenate([t1, t2, t3], axis=-1)  # Concatenate t1, t2, t3 along the last dimension
            # print(t1.shape, t2.shape, t3.shape, t.shape)
            ts.append(torch.tensor(t, dtype=torch.float32))  # Append to list of tensors
        
        tensor = torch.stack(ts, dim=0) # (B, 64, 80)
        return tensor

    def output_from_list(self, inp: list[list]):
        # output from a list [13, 14, 15]
        B = len(inp)
        next_move = torch.full((B, 2048), float('-inf'))
        origin = torch.full((B, 64), float('-inf'))
        target = torch.full((B, 64), float('-inf'))
        legal_moves = torch.full((B, 2048), float('-inf'))
        outcome = torch.full((B, 3), float('-inf')) # P1 perspective, so [0, 1, 2] = [P1 win, draw, P1 loss]
        move_speed = torch.full((B, 8), float('-inf'))
        time_taken_bins = [0.5, 1.5, 2.5, 4.5, 9.5, 19.5, 59.5] # 8 bins
        for i in range(B):
            board = chess.Board(inp[i][1]) # get board from fen
            legal = [move.uci() for move in board.legal_moves]
            if board.turn: # P1 is white
                next_move[i, self.move_stoi(inp[i][13])] = 1
                for m in legal: 
                    legal_moves[i, self.move_stoi(m)] = 1
                if inp[i][13] != '?': # blank tensor if move is not recognized
                    origin[i, self.square_stoi(inp[i][13][:2])] = 1
                    target[i, self.square_stoi(inp[i][13][2:4])] = 1
                
                if inp[i][15].strip() == 'W':
                    outcome[i, 0] = 1
                elif inp[i][15].strip() == 'D':
                    outcome[i, 1] = 1
                else:
                    outcome[i, 2] = 1
                time_taken = float(inp[i][5]) - float(inp[i][14]) + float(inp[i][4]) # W_time - time_rem + increment 
                move_speed[i, np.digitize(time_taken, bins=time_taken_bins)] = 1
            else: # P1 is black
                next_move[i, self.move_stoi(self.mirror_move(inp[i][13]))] = 1
                for m in legal: 
                    legal_moves[i, self.move_stoi(self.mirror_move(m))] = 1
                if inp[i][13] != '?': # blank tensor if move is not recognized
                    origin[i, self.square_stoi(self.mirror_square(inp[i][13][:2]))] = 1
                    target[i, self.square_stoi(self.mirror_square(inp[i][13][2:4]))] = 1
                if inp[i][15].strip() == 'B':
                    outcome[i, 0] = 1
                elif inp[i][15].strip() == 'D':
                    outcome[i, 1] = 1
                else:
                    outcome[i, 2] = 1
                time_taken = float(inp[i][6]) - float(inp[i][14]) + float(inp[i][4]) # B_time - time_rem + increment 
                move_speed[i, np.digitize(time_taken, bins=time_taken_bins)] = 1

        return HC3Output(next_move, origin, target, legal_moves, outcome, move_speed)

    # not very good, but should work??
    def input_from_board(self, board: chess.Board, max_time, inc, W_time, B_time, W_elo, B_elo, context, repetition):
        # input from a chess board directly, outputs a tensor of shape (1, 64, 80)

        t1, is_white_move = self.from_fen(board.fen()) # array of shape (64, 31)
        t2 = self.context_to_tensor(context, is_white_move) # array of shape (64, 8)
        t3 = np.zeros((64,41))
        # repetition
        if repetition == 'T':
            t3[:,0] = 1
        else:
            t3[:,0] = 0

        # bins 
        max_time_bins = [65, 125, 185, 305, 605] # 6
        inc_bins = [0.5, 1.5, 3.5] # 4
        elo_bins = [1200, 1400, 1600, 1800, 2000, 2200] # 7 * 2 = 14
        time_bins = [5, 10, 20, 30, 60, 120, 240] # 8 * 2 = 16
        if is_white_move:
            P1_elo = np.digitize(W_elo, bins=elo_bins)
            P2_elo = np.digitize(B_elo, bins=elo_bins)
            P1_time = np.digitize(W_time, bins=time_bins)
            P2_time = np.digitize(B_time, bins=time_bins)
        else:
            P1_elo = np.digitize(B_elo, bins=elo_bins)
            P2_elo = np.digitize(W_elo, bins=elo_bins)
            P1_time = np.digitize(B_time, bins=time_bins)
            P2_time = np.digitize(W_time, bins=time_bins)
        timing = np.digitize(max_time, bins=max_time_bins)
        inc = np.digitize(inc, bins=inc_bins)

        # order: timing, inc, P1_time, P2_time, P1_elo, P2_elo
        one_hot = np.concatenate([
            np.eye(len(max_time_bins) + 1)[timing],
            np.eye(len(inc_bins) + 1)[inc],
            np.eye(len(time_bins) + 1)[P1_time],
            np.eye(len(time_bins) + 1)[P2_time],
            np.eye(len(elo_bins) + 1)[P1_elo],
            np.eye(len(elo_bins) + 1)[P2_elo]
        ], axis=-1)  # Concatenate one-hot encodings

        one_hot_broadcasted = np.broadcast_to(one_hot, (64, one_hot.shape[-1]))  # Broadcast to (64, X)
        t3[:, 1:] = one_hot_broadcasted  # Fill the rest of t3 with the broadcasted array

        t = np.concatenate([t1, t2, t3], axis=-1)  # Concatenate t1, t2, t3 along the last dimension
        
        return torch.tensor(t, dtype=torch.float32).unsqueeze(0) # (1, 64, 80)
    
    @torch.no_grad()
    def sample_output(self, output: HC3Output, is_white_move: bool, batch_idx: int = 0, move_strategy: str = 'greedy', 
                      speed_strategy: str = 'simple', legal_moves=None): 
        # output to be sampled from
        # batch_idx = which batch to sample? 0 if batch_size = 1
        # move_strategy = greedy, random; how to sample next move
        # speed_strategy = default, complex; default chooses one bin
        # legal_moves = list of legal moves in uci format. If None, may play illegal move 
        new_logits = output.next_move[batch_idx].clone()
        if legal_moves is not None:
            legal_indices = [self.move_stoi(m) for m in legal_moves]
            for idx in range(new_logits.shape[-1]):
                if idx not in legal_indices:
                    new_logits[idx] = float('-inf')
        
        # evaluation
        evaluation = F.softmax(output.outcome[batch_idx], dim=-1).detach().cpu().to(torch.float32).numpy() # [P1 win, draw, P1 loss]

        # next move 
        if move_strategy == 'greedy':
            move = new_logits.argmax().item() # index of best move 
        else: # softmax, then sample 
            move_probs = F.softmax(new_logits, dim=-1)
            move = torch.multinomial(move_probs, num_samples=1).squeeze().item() # index of sampled move 

        # move speed, simple implementation
        speed_probs = F.softmax(output.move_speed[batch_idx], dim=-1)
        speed = torch.multinomial(speed_probs, num_samples=1).squeeze().item() # index of sampled move 
        # speed = output.move_speed[batch_idx].argmax().item() # index of best speed bin
        
        move_final = self.move_itos(move) # str

        if not is_white_move:
            move_final = self.mirror_move(move_final)
        
        speed_final = self.speed_itof(speed, speed_strategy) # float
        eval_str = f'P1 win: {evaluation[0]:.2f}, draw: {evaluation[1]:.2f}, P1 loss: {evaluation[2]:.2f}'

        return move_final, speed_final, eval_str
        
    def get_squares(self) -> list:
        squares = []
        for i in range(8):
            for j in range(8):
                squares.append(chr(i+97) + str(j+1))
        return squares

    def get_all_moves(self) -> list:
        # get all possible moves in lan format from any legal board position
        # exclude geometrically impossible moves

        moves = ['?'] # unknown token, in case a move isn't recognized
        letter = lambda n: chr(n + 96)
        for i in range(1, 9): # 1 to 8, letter(i) is a to h
            for j in range(1, 9): # 1 to 8
                # letter(i) + str(j) = board position 
                curr_pos = letter(i) + str(j)

                # handle rook-like movement
                val = j 
                while 1 < val: # down 
                    val -= 1
                    moves.append(curr_pos + letter(i) + str(val))
                val = j 
                while val < 8: # up 
                    val += 1
                    moves.append(curr_pos + letter(i) + str(val))
                val = i
                while 1 < val: # left
                    val -= 1
                    moves.append(curr_pos + letter(val) + str(j))
                val = i
                while val < 8: # right
                    val += 1
                    moves.append(curr_pos + letter(val) + str(j))

                # handle bishop-like movement
                val1 = i 
                val2 = j
                while 1 < val1 and 1 < val2: # to bottom-left
                    val1 -= 1
                    val2 -= 1
                    moves.append(curr_pos + letter(val1) + str(val2))
                val1 = i 
                val2 = j
                while val1 < 8 and val2 < 8: # to top-right
                    val1 += 1
                    val2 += 1
                    moves.append(curr_pos + letter(val1) + str(val2))
                val1 = i 
                val2 = j
                while 1 < val1 and val2 < 8: # to top-left
                    val1 -= 1
                    val2 += 1
                    moves.append(curr_pos + letter(val1) + str(val2))
                val1 = i 
                val2 = j
                while val1 < 8 and 1 < val2: # to bottom-right
                    val1 += 1
                    val2 -= 1
                    moves.append(curr_pos + letter(val1) + str(val2))

                # handle knight movement 
                if 1 <= i+2 <= 8 and 1 <= j+1 <= 8: 
                    moves.append(curr_pos + letter(i+2) + str(j+1))
                if 1 <= i+1 <= 8 and 1 <= j+2 <= 8: 
                    moves.append(curr_pos + letter(i+1) + str(j+2))
                if 1 <= i-1 <= 8 and 1 <= j+2 <= 8: 
                    moves.append(curr_pos + letter(i-1) + str(j+2))
                if 1 <= i-2 <= 8 and 1 <= j+1 <= 8: 
                    moves.append(curr_pos + letter(i-2) + str(j+1))
                if 1 <= i+2 <= 8 and 1 <= j-1 <= 8: 
                    moves.append(curr_pos + letter(i+2) + str(j-1))
                if 1 <= i+1 <= 8 and 1 <= j-2 <= 8: 
                    moves.append(curr_pos + letter(i+1) + str(j-2))
                if 1 <= i-2 <= 8 and 1 <= j-1 <= 8: 
                    moves.append(curr_pos + letter(i-2) + str(j-1))
                if 1 <= i-1 <= 8 and 1 <= j-2 <= 8: 
                    moves.append(curr_pos + letter(i-1) + str(j-2))

        # handle promotion 
        for i in range(1, 9): # 1 to 8
            curr_pos = letter(i) + '7'
            s = 'qrbn' # promotion possibilities 
            for j in [-1, 0, 1]: # capture left, push, or capture right to promote
                for c in s:
                    if 1 <= i+j <= 8: # eg. can't capture left on a-file 
                        moves.append(curr_pos + letter(i+j) + '8' + c)

        # return list 
        return moves
    
    def speed_itof(self, val: int, strategy = 'simple') -> float:
        # given a speed bin index, return a move speed as a float
        if strategy == 'simple':
            if val == 0: # 0 to 0.5
                return 0.5
            elif val == 1: # 0.5 to 1.5
                return 1
            elif val == 2: # 1.5 to 2.5
                return 2
            elif val == 3: # 2.5 to 4.5
                return 3.5
            elif val == 4: # 4.5 to 9.5
                return 7
            elif val == 5: # 9.5 to 19.5
                return 12
            elif val == 6: # 19.5 to 59.5
                return 25
            else:
                return 60
        else:
            raise NotImplementedError

    def init_square_dicts(self, squares: list): #
        self._stoi_square = { ch:i for i,ch in enumerate(squares) }
        self._itos_square = { i:ch for i,ch in enumerate(squares) }

    def init_move_dicts(self, moves: list): # 
        self._stoi_move = { ch:i for i,ch in enumerate(moves) }
        self._itos_move = { i:ch for i,ch in enumerate(moves) }

    def move_stoi(self, move: str) -> int:
        return self._stoi_move[move] if move in self._stoi_move else 0

    def move_itos(self, val: int) -> str:
        return self._itos_move[val] if val in self._itos_move else '?'

    def square_stoi(self, square: str) -> int:
        return self._stoi_square[square]
    
    def square_itos(self, val: int) -> str:
        return self._itos_square[val]

    def mirror_square(self, square: str) -> str:
        # given a square, return the mirror square
        return square[0] + str(9 - int(square[1]))

    def mirror_move(self, move: str) -> str:
        # given a move, return the mirror move
        if move == '?': 
            return '?'
        elif len(move) > 4: # promotion
            return self.mirror_square(move[:2]) + self.mirror_square(move[2:4]) + move[4]
        else: # regular move 
            return self.mirror_square(move[:2]) + self.mirror_square(move[2:])
    
    def context_to_tensor(self, moves: list, is_white_move: bool): # list of past moves 
        # Board squares in correct order
        if is_white_move:
            # a1->h1,..., a8->h8
            squares = list(chess.SQUARES)
        else:
            # a8->h8,..., a1->h1
            squares = [
                chess.square(file, rank)
                for rank in range(7, -1, -1)
                for file in range(8)
            ]
        
        # Initialize 64 x 8 tensor
        tensor = np.zeros((64, len(moves)*2), dtype=np.float32)

        # Process up to 4 moves
        for i in range(len(moves)):
            move_str = moves[i]
            if move_str != '?': 
                origin_str = move_str[:2]  # e.g. 'e2'
                target_str = move_str[2:4]  # e.g. 'e4'
            else:
                continue

            # Convert to square indexes (0..63)
            origin_sq = chess.parse_square(origin_str)
            target_sq = chess.parse_square(target_str)

            # Find their indices in our squares list
            origin_idx = squares.index(origin_sq)
            target_idx = squares.index(target_sq)

            # Mark the appropriate channels
            # channel 2*i => origin, 2*i + 1 => target
            tensor[origin_idx, 2*i] = 1.0
            tensor[target_idx, 2*i + 1] = 1.0

        return tensor

    def from_fen(self, fen: str = None, board: chess.Board = None): # given a fen string, return a tensor of shape (64, 31)
        # Channels:
        # 0–5   : P1 Pieces   (K, Q, R, B, N, P)
        # 6–11  : P2 Pieces   (K, Q, R, B, N, P)
        # 12    : EP Capture
        # 13–14 : P1 Castling (Kingside, Queenside) 
        # 15–16 : P2 Castling (Kingside, Queenside)
        # 17    : P1 Piece Mask
        # 18    : P2 Piece Mask
        # 19–23 : P1 Material Difference (Q, R, B, N, P)
        # 24    : Checkers (squares occupied by checking P2 pieces)
        # 25–29 : P1 Material Count (Q, R, B, N, P)
        # 30    : 50-move rule (halfmove clock)
        if fen is not None:
            board = chess.Board(fen)
        # Identify P1 color (the side to move) and P2 color (the other side)
        p1_color = board.turn  # True for White, False for Black

        # Decide square iteration order (White perspective or Black perspective)
        if p1_color == chess.WHITE:
            # a1->h1, a2->h2, ... a8->h8
            squares = list(chess.SQUARES)
        else:
            # a8->h8, a7->h7, ... a1->h1
            squares = [
                chess.square(file, rank)
                for rank in range(7, -1, -1)  # 7 -> 0
                for file in range(8)         # 0 -> 7
            ]

        # Create the 64 x 31 output
        tensor = np.zeros((64, 31), dtype=np.float32)

        # --- 1) Fill piece channels & piece masks ---
        # Map each piece symbol to its channel index for P1 vs P2
        p1_map = {'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5}
        p2_map = {'K': 6, 'Q': 7, 'R': 8, 'B': 9, 'N': 10, 'P': 11}

        for i, sq in enumerate(squares):
            piece = board.piece_at(sq)
            if piece:
                symbol = piece.symbol().upper()  # K, Q, R, B, N, P
                color_is_white = piece.color
                # Check if this piece is P1’s or P2’s
                is_p1 = (color_is_white == p1_color)
                if is_p1 and symbol in p1_map:
                    tensor[i, p1_map[symbol]] = 1
                    tensor[i, 17] = 1  # P1 piece mask
                elif not is_p1 and symbol in p2_map:
                    tensor[i, p2_map[symbol]] = 1
                    tensor[i, 18] = 1  # P2 piece mask

        # --- 2) En passant channel (12) ---
        if board.ep_square is not None:
            ep_index = squares.index(board.ep_square)  # find which of the 64 we’re at
            tensor[ep_index, 12] = 1

        # --- 3) Castling rights (13–14 for P1, 15–16 for P2) ---
        # For P1
        if p1_color == chess.WHITE:
            tensor[:, 13] = int(board.has_kingside_castling_rights(chess.WHITE))
            tensor[:, 14] = int(board.has_queenside_castling_rights(chess.WHITE))
            tensor[:, 15] = int(board.has_kingside_castling_rights(chess.BLACK))
            tensor[:, 16] = int(board.has_queenside_castling_rights(chess.BLACK))
        else: # For P2
            tensor[:, 13] = int(board.has_kingside_castling_rights(chess.BLACK))
            tensor[:, 14] = int(board.has_queenside_castling_rights(chess.BLACK))
            tensor[:, 15] = int(board.has_kingside_castling_rights(chess.WHITE))
            tensor[:, 16] = int(board.has_queenside_castling_rights(chess.WHITE))

        # --- 4) Material difference (19–23) & P1 material count (25–29) ---
        # Count pieces on board
        # Only Q, R, B, N, P matter for difference & count
        piece_order = ['Q', 'R', 'B', 'N', 'P']
        p1_counts = {typ: 0 for typ in piece_order}
        p2_counts = {typ: 0 for typ in piece_order}

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                symbol = piece.symbol().upper()
                if symbol in piece_order:
                    if piece.color == p1_color:
                        p1_counts[symbol] += 1
                    else:
                        p2_counts[symbol] += 1

        # Fill difference channels
        for idx, typ in enumerate(piece_order):
            diff = p1_counts[typ] - p2_counts[typ]
            tensor[:, 19 + idx] = diff

        # Fill material count channels
        for idx, typ in enumerate(piece_order):
            tensor[:, 25 + idx] = p1_counts[typ]

        # --- 5) Checkers channel (24) ---
        # Identify if the P1 king is in check, and which P2 pieces deliver the check
        if board.is_check():
            # If P1 is in check, find squares that deliver it
            king_sq = board.king(p1_color)
            attackers = board.attackers(not p1_color, king_sq)
            # Mark each attacker’s square
            for sq_attacker in attackers:
                if board.piece_at(sq_attacker):
                    # locate sq_attacker in the squares list
                    idx = squares.index(sq_attacker)
                    tensor[idx, 24] = 1
        
            # --- 6) 50move rule (halfmove clock) channel (30) ---
            halfmove_clock = board.halfmove_clock
            tensor[:, 30] = halfmove_clock / 50 # Normalize to [0, 1] range

        return tensor, board.turn


### Dataset Handling
class ChessDataset(Dataset):
    def __init__(self, csv_file: str, index_file: str = None):
        self.csv_file = csv_file
        self.index_file = index_file or f"{csv_file}.index"

        # Build or load the index
        # self._build_index() 
        if not os.path.exists(self.index_file):
            self._build_index()

        with open(self.index_file, 'r') as f:
            self.offsets = [int(line.strip()) for line in f]

        self.total_rows = len(self.offsets) - 2  
        # Last offset is EOF, 2nd to last is blank

        # Memory-map the CSV file in read-only mode
        self._file_obj = open(self.csv_file, 'rb')  # keep file handle open
        self.mmap_obj = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self): 
        return self.total_rows

    def __del__(self):
        # Clean up the mmap and file handle
        if hasattr(self, 'mmap_obj'):
            self.mmap_obj.close()
        if hasattr(self, '_file_obj'):
            self._file_obj.close()

    def __getitem__(self, idx): 
        # Load the specific row as needed, minimizing memory usage.
        start_offset = self.offsets[idx]
        end_offset = self.offsets[idx + 1]

        # Slice the memory map for this row's bytes, strip the newline, decode to string
        row_bytes = self.mmap_obj[start_offset:end_offset].rstrip(b'\n')
        row_str = row_bytes.decode('utf-8')

        # Split the row into a list and process using HCInput
        row_list = row_str.split(',')
        x = row_list[:72] 
        y = row_list # variable length if legal is true
        return x, y

    def _build_index(self):
        # Build an index file mapping each row to its byte offset.
        with open(self.csv_file, 'r', encoding='utf-8') as f, open(self.index_file, 'w') as index_f:
            l = f.readline() # skip header
            while l:
                offset = f.tell()
                l = f.readline()
                index_f.write(f"{offset}\n")

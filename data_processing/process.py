"""
With no augmentation, 1M positions takes about 420 seconds, 122 MB of space, 
and about 15000 games. To reach 10 million games, or about 650M positions, 
it should take about 76 hours, and 80 GB of space.

With augmentation, 1M position takes about 830 seconds, 122 MB of space, about 
17.5k games, and 260k games processed. To reach 10 million games, or about 570M 
positions, it should take about 130 hours, 70 GB of space, and 150M games processed. 
"""
import chess
import chess.pgn
import io
import re
import os
import csv
import time
import random
import numpy as np

def parse_pgn_to_rows(pgn_string: str):
    """
    Parses a single PGN string representing one chess game and returns
    a list of rows (lists) for each position with the specified columns.

    Returns:
        rows (list of list): Each sublist has:
            [
              game_id,
              move_count,
              player_to_move,
              fen,
              repetition,
              progress_count,
              max_time,
              increment,
              W_time,
              B_time,
              W_elo,
              B_elo,
              past_move1,
              past_move2,
              past_move3,
              past_move4,
              next_move,
              W_time_rem,
              B_time_rem,
              outcome
            ]
    """

    # Helper Functions
    
    def get_time_from_comment(comment):
        """
        Scan a move comment for something like [%clk 0:00:29]
        and return the time in seconds if found, else None.
        """
        clk_match = re.search(r'\[%clk\s+(\d:\d{2}:\d{2})\]', comment)
        if not clk_match:
            return None
        else: 
            match = re.match(r'(\d+):(\d+):(\d+)', clk_match.group(1))
            if not match:
                return None
            hours = int(match.group(1))
            mins = int(match.group(2))
            secs = int(match.group(3))
            return int(hours * 3600 + mins * 60 + secs)

    def check_if_repetition_is_possible(board: chess.Board) -> str:
        """
        Check if there is any legal move that would lead to
        a threefold repetition if that move is played.
        """
        for move in board.legal_moves:
            board.push(move)
            # If the resulting position is already repeated twice, 
            # then one more occurrence would allow a claim of threefold repetition
            # or if directly is_repetition(3) is True, it means 
            # the position has now occurred 3 times in total.
            if board.is_repetition(3):
                board.pop()
                return 'T'
            board.pop()
        return 'F'

    # Open PGN in memory
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    
    # If there's no game, return empty
    if game is None:
        return []

    # Set up a board from the initial position
    board = game.board()

    ply_count = sum(1 for _ in game.mainline_moves())

    if ply_count < 30: 
        return []

    headers = game.headers

    # Game ID from the 'Site' field. E.g. https://lichess.org/PpwPOZMq => PpwPOZMq
    site_value = headers.get("Site", "")
    if site_value == "": 
        return []
    game_id = site_value.rsplit('/', 1)[-1]

    # Time control parsing, e.g. "30+0"
    time_control = headers.get("TimeControl", "")
    if time_control == "": 
        return []

    if '+' in time_control:
        max_time_str, increment_str = time_control.split('+', 1)
        try:
            max_time = int(max_time_str)  # in seconds 
            increment = int(increment_str) # in seconds
        except ValueError:
            return []
    else:
        return [] 

    if max_time > 1205: # no games longer than 20 minutes
        return []

    w_elo = int(headers.get("WhiteElo", "-1"))
    b_elo = int(headers.get("BlackElo", "-1"))

    if w_elo == '-1' or b_elo == '-1':
        return []
    elif w_elo < 1000 or b_elo < 1000: # no beginners :(
        return []

    result_str = headers.get("Result", "*")  # e.g. "1-0", "0-1", "1/2-1/2"
    if result_str == "1-0":
        outcome = 'W'
    elif result_str == "1/2-1/2":
        outcome = 'D'
    elif result_str == "0-1":
        outcome = 'B'
    else:
        return [] # unknown result 


    ### Game-level sampling augmentation here

    r = random.randint(1, 1000)
    # time control 
    if (50 < max_time < 70 or 125 < max_time < 310) and r > 350: # roughly, undersample 1 minute and 3-5 minute games
        return []

    r = random.randint(1, 1000) # reset random
    r2 = random.randint(1, 1000) 
    # elo match
    elo_bins = [990,1200,1400,1600,1800,2000,2200,5000]
    w_dig = np.digitize(w_elo, elo_bins)
    b_dig = np.digitize(b_elo, elo_bins)

    if w_dig == b_dig: 
        if r > 125: # same bin, 1/8 
            return []
        if w_dig == 2 and r2 > 800: # 0.8x 
            return []
        elif w_dig == 3 and r2 > 550: # 0.55x
            return []
        elif (w_dig == 4 or w_dig == 5) and r2 > 500: # 0.5x 
            return []
        elif w_dig == 6 and r2 > 700: # 0.7x 
            return []

    elif abs(w_dig - b_dig) == 1: # one off, 1/2
        if r > 500:
            return []

        if w_dig == 2 and r2 > 800: # 0.8x 
            return []
        elif w_dig == 3 and r2 > 550: # 0.55x
            return []
        elif (w_dig == 4 or w_dig == 5) and r2 > 500: # 0.5x 
            return []
        elif w_dig == 6 and r2 > 700: # 0.7x 
            return []

    # We'll traverse the moves. 
    rows = []

    # Initialize them to max_time (in seconds)
    white_time = max_time
    black_time = max_time

    # For storing the last 4 moves in UCI
    past_moves_uci = []

    # We’ll define a small function to append a row given the current board
    def append_position_row():
        # game_id
        # fen
        # repetition: boolean if next move can produce a threefold repetition
        # max_time, increment
        # W_time, B_time (time on the clock when position shows up)
        # W_elo, B_elo
        # past_move1..4 (from oldest to newest if fewer than 4, use '0')
        # next_move
        # time_rem (time left after next_move is played)
        # outcome 

        # Build past_moveX from the tail of past_moves_uci
        # If we have fewer than 4, fill with '0'
        pm = past_moves_uci[-4:]  # up to 4 last moves
        pm = ['?'] * (4 - len(pm)) + pm  # left-pad with '0' if needed

        # Next move in UCI
        nxt_move = str(move)

        # Check repetition possibility
        repetition = check_if_repetition_is_possible(board)

        # Build the row
        row = [
            game_id,
            board.fen(),
            repetition,
            max_time,
            increment,
            white_time,
            black_time,
            w_elo,
            b_elo,
            pm[0], pm[1], pm[2], pm[3],  # past_move1..4
            nxt_move,                    # next_move
            time_rem,                    # for whoever's move it just was
            outcome
        ]
        rows.append(row)

    # Now we iterate through moves
    node = game
    while node.variations:
        move = node.variations[0].move
        move_comment = node.variations[0].comment
        time_rem = get_time_from_comment(move_comment)

        num = 2*board.fullmove_number + abs(board.turn-1) - 1 # ply count 

        r = random.randint(1,1000)
        if num < 17 and r > 1.5**num: 
            pass  
        else:
            append_position_row()

        # update clock
        if board.turn: # white's move
            white_time = time_rem
        else: # black's move
            black_time = time_rem

        # Make the move on the board
        board.push(move)

        # Now the new position is reached. 
        # Let’s store that move in past_moves_uci
        past_moves_uci.append(move.uci())

        # Advance to the next move in the PGN
        node = node.variations[0]

    return rows


def process_file(inp_file, out_file):
    header = ['game_id', 'fen', 'repetition', 'max_time', 'increment',
              'W_time', 'B_time', 'W_elo', 'B_elo', 'past_move1',
              'past_move2', 'past_move3', 'past_move4', 'next_move',
              'time_rem', 'outcome']

    with open(out_file, "w") as f: # w or a 
        w = csv.writer(f)
        w.writerow(header)

    game_lines = []
    games = 0 # total games looked at 
    games_added = 0 # games that weren't filtered
    positions = 0
    ls = [] # look at 500 games before updating file

    print("Number of games processed so far:")

    with open(inp_file, 'r') as file:
        for line_count, line in enumerate(file):

            # Check if the line starts a new game
            if line.startswith("[Event") and game_lines:
                games += 1 # increment total games looked at

                # process the previous game
                s = ''.join(game_lines)
                l = parse_pgn_to_rows(s) # list of lists or []
                if len(l) != 0:
                    games_added += 1

                positions += len(l)
                ls.extend(l)

                if games % 10000 == 0:
                    with open(out_file, 'a') as f:
                        w = csv.writer(f)
                        w.writerows(ls)
                    ls = []
                    print(f"\r{games}   {games_added}   {positions}   {line_count}")
                
                # Reset the current game lines
                game_lines = []

            # Add the current line to the game
            game_lines.append(line)

    print("\nDone!")
    print("Final Numbers:")
    print(f"{games}   {games_added}   {positions}")


if __name__ == "__main__":
    t0 = time.time() 
    inp_dir = r'D:/'
    out_file = r'C:/Users/srita/Documents/data/data.csv'

    files = os.listdir(inp_dir)
    print(files)

    for file in files:
        if not file.endswith('.pgn'): # only .pgn files in folder
            continue
        if file.startswith('.'):
            continue 
        print("Processing:", file)
        process_file(os.path.join(inp_dir, file), out_file)
    t1 = time.time()
    dt = t1 - t0
    print(dt, 'seconds')

    # testing 
    pgn_input = """[Event "Rated Bullet tournament https://lichess.org/tournament/yc1WW2Ox"]
[Site "https://lichess.org/PpwPOZMq"]
[Date "2017.04.01"]
[Round "-"]
[White "Abbot"]
[Black "Costello"]
[Result "0-1"]
[UTCDate "2017.04.01"]
[UTCTime "11:32:01"]
[WhiteElo "2100"]
[BlackElo "2000"]
[WhiteRatingDiff "-4"]
[BlackRatingDiff "+1"]
[WhiteTitle "FM"]
[ECO "B30"]
[Opening "Sicilian Defense: Old Sicilian"]
[TimeControl "30+0"]
[Termination "Time forfeit"]

1. e4 { [%eval 0.17] [%clk 0:00:30] } 1... c5 { [%eval 0.19] [%clk 0:00:30] }
2. Nf3 { [%eval 0.25] [%clk 0:00:29] } 2... Nc6 { [%eval 0.33] [%clk 0:00:30] }
3. Bc4 { [%eval -0.13] [%clk 0:00:28] } 3... e6 { [%eval -0.04] [%clk 0:00:30] }
4. c3 { [%eval -0.4] [%clk 0:00:27] } 4... b5? { [%eval 1.18] [%clk 0:00:30] }
5. Bb3?! { [%eval 0.21] [%clk 0:00:26] } 5... c4 { [%eval 0.32] [%clk 0:00:29] }
6. Bc2 { [%eval 0.2] [%clk 0:00:25] } 6... a5 { [%eval 0.6] [%clk 0:00:29] }
7. d4 { [%eval 0.29] [%clk 0:00:23] } 7... cxd3 { [%eval 0.6] [%clk 0:00:27] }
8. Qxd3 { [%eval 0.12] [%clk 0:00:22] } 8... Nf6 { [%eval 0.52] [%clk 0:00:26] }
9. e5 { [%eval 0.39] [%clk 0:00:21] } 9... Nd5 { [%eval 0.45] [%clk 0:00:25] }
10. Bg5?! { [%eval -0.44] [%clk 0:00:18] } 10... Qc7 { [%eval -0.12] [%clk 0:00:23] }
11. Nbd2?? { [%eval -3.15] [%clk 0:00:14] } 11... h6 { [%eval -2.99] [%clk 0:00:23] }
12. Bh4 { [%eval -3.0] [%clk 0:00:11] } 12... Ba6? { [%eval -0.12] [%clk 0:00:23] }
13. b3?? { [%eval -4.14] [%clk 0:00:02] } 13... Nf4? { [%eval -2.73] [%clk 0:00:21] } 0-1"""

    #rows = parse_pgn_to_rows(pgn_input)
    
    # Print out each row nicely
    #for r in rows:
    #    print(r)
import copy

# Inicialización del tablero y contador de fichas
board = [[None, None, None] for _ in range(3)]
pieces = {'X': 0, 'O': 0}

def print_board(b):
    print("\n  0 1 2")
    for i, row in enumerate(b):
        print(i, end=" ")
        print(" ".join(c if c else "." for c in row))
    print()

def check_winner(b):
    lines = b + list(map(list, zip(*b))) + [[b[i][i] for i in range(3)]] + [[b[i][2-i] for i in range(3)]]
    for line in lines:
        if line[0] and all(c == line[0] for c in line):
            return line[0]
    return None

def get_possible_moves(b, player, local_pieces):
    moves = []
    if local_pieces[player] < 3:
        for i in range(3):
            for j in range(3):
                if b[i][j] is None:
                    moves.append(('place', i, j))
    else:
        for i in range(3):
            for j in range(3):
                if b[i][j] == player:
                    for x in range(3):
                        for y in range(3):
                            if b[x][y] is None:
                                moves.append(('move', i, j, x, y))
    return moves

def apply_move(b, move, player, real=True, local_pieces=None):
    if real:
        global pieces
        if move[0] == 'place':
            _, i, j = move
            b[i][j] = player
            pieces[player] += 1
        elif move[0] == 'move':
            _, i, j, x, y = move
            b[i][j] = None
            b[x][y] = player
    else:
        # Usado para minimax: no afecta contador global
        if move[0] == 'place':
            _, i, j = move
            b[i][j] = player
            local_pieces[player] += 1
        elif move[0] == 'move':
            _, i, j, x, y = move
            b[i][j] = None
            b[x][y] = player
    return b

def minimax(b, player, depth, maximizing, local_pieces):
    winner = check_winner(b)
    if winner == 'X':
        return 1
    elif winner == 'O':
        return -1
    elif depth == 0:
        return 0

    if maximizing:
        max_eval = -float('inf')
        for move in get_possible_moves(b, 'X', local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, 'X', real=False, local_pieces=new_pieces)
            eval = minimax(new_board, 'O', depth-1, False, new_pieces)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_possible_moves(b, 'O', local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, 'O', real=False, local_pieces=new_pieces)
            eval = minimax(new_board, 'X', depth-1, True, new_pieces)
            min_eval = min(min_eval, eval)
        return min_eval

def best_move(b, player):
    moves = get_possible_moves(b, player, pieces)
    best_val = -float('inf') if player == 'X' else float('inf')
    best_m = None
    for move in moves:
        new_board = copy.deepcopy(b)
        new_pieces = copy.deepcopy(pieces)
        apply_move(new_board, move, player, real=False, local_pieces=new_pieces)
        val = minimax(new_board, 'O' if player=='X' else 'X', 3, player=='O', new_pieces)
        if (player=='X' and val > best_val) or (player=='O' and val < best_val):
            best_val = val
            best_m = move
    return best_m

def player_turn(player):
    print(f"Turno de {player}")
    moves = get_possible_moves(board, player, pieces)
    if pieces[player] < 3:
        print("Introduce fila y columna para colocar tu ficha (ej: 0 2):")
        while True:
            try:
                line = input()
                i, j = map(int, line.strip().split())
                if ('place', i, j) in moves:
                    return ('place', i, j)
            except:
                pass
            print("Movimiento inválido. Intenta otra vez.")
    else:
        print("Tienes 3 fichas. Movimientos posibles:")
        for m in moves:
            print(f"De ({m[1]},{m[2]}) a ({m[3]},{m[4]})")
        print("Formato: fila_origen col_origen fila_destino col_destino")
        while True:
            try:
                line = input()
                i, j, x, y = map(int, line.strip().split())
                if ('move', i, j, x, y) in moves:
                    return ('move', i, j, x, y)
            except:
                pass
            print("Movimiento inválido. Intenta otra vez.")

def play_game():
    current_player = 'X'  # Humano
    ai_player = 'O'

    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"¡{winner} ha ganado!")
            break

        if current_player == 'X':
            move = player_turn(current_player)
        else:
            move = best_move(board, current_player)
            print(f"IA ({current_player}) juega:", move)

        apply_move(board, move, current_player)
        current_player = 'O' if current_player == 'X' else 'X'

play_game()

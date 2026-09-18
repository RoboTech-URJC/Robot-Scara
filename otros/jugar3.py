import copy
import serial
import time
import cv2
import numpy as np
import sys

# ===== CONFIGURACIÓN GLOBAL =====
CAM_ID = 0  # ID de la cámara (cámbialo si no detecta la correcta)
PLAYER_HUMAN = 'X'  # Humano (fichas azules)
PLAYER_AI = 'O'     # IA (fichas rojas)
MAX_DEPTH = 6       # Profundidad para el algoritmo Minimax

ROWS, COLS = 6, 7   # Tablero estándar de 4 en raya

# ===== INICIALIZACIÓN DEL TABLERO =====
def create_board():
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def print_board(board):
    for row in board:
        print('|'.join(row))
    print('-' * 13)

def is_valid_location(board, col):
    return board[0][col] == ' '

def get_next_open_row(board, col):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == ' ':
            return r
    return None

def drop_piece(board, row, col, piece):
    board[row][col] = piece

# ===== DETECCIÓN DE VICTORIA =====
def winning_move(board, piece):
    # Horizontal
    for c in range(COLS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Diagonal positiva
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    # Diagonal negativa
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_HUMAN if piece == PLAYER_AI else PLAYER_AI

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(' ') == 1:
        score += 10
    elif window.count(piece) == 2 and window.count(' ') == 2:
        score += 5

    if window.count(opp_piece) == 3 and window.count(' ') == 1:
        score -= 80

    return score

def score_position(board, piece):
    score = 0
    # Columna central
    center_array = [board[r][COLS//2] for r in range(ROWS)]
    score += center_array.count(piece) * 6
    # Horizontales
    for r in range(ROWS):
        row_array = board[r]
        for c in range(COLS-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)
    # Verticales
    for c in range(COLS):
        col_array = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS-3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)
    # Diagonales positivas
    for r in range(ROWS-3):
        for c in range(COLS-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    # Diagonales negativas
    for r in range(3, ROWS):
        for c in range(COLS-3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    return score

def get_valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]

def is_terminal_node(board):
    return winning_move(board, PLAYER_HUMAN) or winning_move(board, PLAYER_AI) or len(get_valid_locations(board)) == 0

# ===== MINIMAX CON PODA ALFA-BETA =====
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)
    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, PLAYER_AI):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_HUMAN):
                return (None, -10000000000000)
            else:  # Empate
                return (None, 0)
        else:
            return (None, score_position(board, PLAYER_AI))
    if maximizingPlayer:
        value = -np.inf
        best_col = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, PLAYER_AI)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = np.inf
        best_col = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, PLAYER_HUMAN)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

# ===== DETECCIÓN DE FICHAS CON OPENCV =====
def detectar_movimiento_humano(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Rango azul (humano)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            col = int(x // (frame.shape[1] / COLS))
            return col
    return None

# ===== MAIN =====
def main():
    board = create_board()
    game_over = False
    turn = 0  # 0 = humano, 1 = IA

    cap = cv2.VideoCapture(CAM_ID)

    while not game_over:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer de la cámara")
            break

        cv2.imshow("4 en raya - Cámara", frame)

        if turn == 0:
            col = detectar_movimiento_humano(frame)
            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_HUMAN)

                if winning_move(board, PLAYER_HUMAN):
                    print("¡Ganaste!")
                    game_over = True

                turn = 1
                print_board(board)

        else:
            col, minimax_score = minimax(board, MAX_DEPTH, -np.inf, np.inf, True)
            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_AI)

                if winning_move(board, PLAYER_AI):
                    print("La IA ganó.")
                    game_over = True

                turn = 0
                print_board(board)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
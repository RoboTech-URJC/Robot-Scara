import copy
import serial
import time
import cv2
import numpy as np
import tkinter as tk
from threading import Thread

# ===============================
#   MANEJADOR DE CÁMARA (OO)
# ===============================
class CameraManager:
    def __init__(self, cam_index=0, max_cams=3):
        self.cam_index = cam_index
        self.max_cams = max_cams
        self.cap = None
        self.open_camera(self.cam_index)

    def open_camera(self, index):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {index}")
            return False
        print(f"✅ Cámara {index} abierta correctamente")
        self.cam_index = index
        return True

    def next_camera(self):
        new_index = (self.cam_index + 1) % self.max_cams
        self.open_camera(new_index)

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

# ===============================
#   CONFIGURACIÓN SERIAL
# ===============================
try:
    arduino = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
    time.sleep(2)  # Espera a que Arduino se inicialice
except serial.SerialException as e:
    print(f"❌ Error al conectar con Arduino: {e}")
    arduino = None

# ===============================
#   TABLERO Y CONTADOR
# ===============================
board = [[None, None, None] for _ in range(3)]
pieces = {'X': 0, 'O': 0}

ORIGIN = (25, 0, 0)  # Donde están las piezas sin colocar
COORDS = {
    (0, 0): (750, -990, -820),
    (0, 1): (861, -1178.8, -820),
    (0, 2): (1011, -1328, -820),
    (1, 0): (851, -824.22, -820),
    (1, 1): (1027.5, -1030.96, -820),
    (1, 2): (1210, -1128, -820),
    (2, 0): (893, -615, -820),
    (2, 1): (1069, -789, -820),
    (2, 2): (1219, -875, -820),
}

# ===============================
#   FUNCIONES GENERALES
# ===============================
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
    return None if any(None in row for row in b) else 'draw'

# ===============================
#   IA (MINIMAX)
# ===============================
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
        if move[0] == 'place':
            _, i, j = move
            b[i][j] = player
            local_pieces[player] += 1
        elif move[0] == 'move':
            _, i, j, x, y = move
            b[i][j] = None
            b[x][y] = player
    return b

def minimax(b, player, depth, maximizing, alpha, beta, local_pieces):
    winner = check_winner(b)
    if winner == 'X':
        return 1
    elif winner == 'O':
        return -1
    elif winner == 'draw' or depth == 0:
        return 0

    if maximizing:
        max_eval = -float('inf')
        for move in get_possible_moves(b, 'X', local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, 'X', real=False, local_pieces=new_pieces)
            eval = minimax(new_board, 'O', depth-1, False, alpha, beta, new_pieces)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_possible_moves(b, 'O', local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, 'O', real=False, local_pieces=new_pieces)
            eval = minimax(new_board, 'X', depth-1, True, alpha, beta, new_pieces)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def best_move(b, player):
    moves = get_possible_moves(b, player, pieces)
    if not moves:
        return None
    best_val = -float('inf') if player == 'X' else float('inf')
    best_m = moves[0]
    for move in moves:
        new_board = copy.deepcopy(b)
        new_pieces = copy.deepcopy(pieces)
        apply_move(new_board, move, player, real=False, local_pieces=new_pieces)
        val = minimax(new_board, 'O' if player == 'X' else 'X', 5, player == 'O', -float('inf'), float('inf'), new_pieces)
        if (player == 'X' and val > best_val) or (player == 'O' and val < best_val):
            best_val = val
            best_m = move
    return best_m

# ===============================
#   VISIÓN POR COMPUTADORA
# ===============================
def find_pieces_on_board(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    azul_bajo = np.array([100, 100, 100])
    azul_alto = np.array([140, 255, 255])
    mask_azul = cv2.inRange(hsv, azul_bajo, azul_alto)

    rojo_bajo1 = np.array([0, 100, 70])
    rojo_alto1 = np.array([15, 255, 255])
    rojo_bajo2 = np.array([165, 100, 70])
    rojo_alto2 = np.array([180, 255, 255])
    mask_rojo = cv2.inRange(hsv, rojo_bajo1, rojo_alto1) | cv2.inRange(hsv, rojo_bajo2, rojo_alto2)

    contours_rojo, _ = cv2.findContours(mask_rojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rojo:
        return None, []
    c = max(contours_rojo, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    board_roi = (x, y, w, h)

    contours_azul, _ = cv2.findContours(mask_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_piece_coords = []
    cell_w = w // 3
    cell_h = h // 3

    for cnt in contours_azul:
        area = cv2.contourArea(cnt)
        if 500 < area < 5000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if x <= cx <= x+w and y <= cy <= y+h:
                    i = (cy - y) // cell_h
                    j = (cx - x) // cell_w
                    blue_piece_coords.append((i, j))
    return board_roi, blue_piece_coords

# ===============================
#   TURNOS
# ===============================
def player_turn(player, cam_manager):
    global board, pieces
    print(f"\nTurno de {player}")

    while True:
        ret, frame = cam_manager.read()
        if not ret:
            print("❌ Error al capturar imagen")
            break

        board_roi, detected_pieces = find_pieces_on_board(frame)
        if board_roi:
            x, y, w, h = board_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cell_w = w // 3
            cell_h = h // 3
            for i, j in detected_pieces:
                cx = x + cell_w//2 + j*cell_w
                cy = y + cell_h//2 + i*cell_h
                cv2.circle(frame, (cx, cy), 15, (255, 255, 0), -1)

        cv2.imshow("Coloca tu ficha azul", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            pieces['X'] += 1
            break
        elif key == ord('q'):
            cam_manager.release()
            cv2.destroyAllWindows()
            exit()

    return None

# ===============================
#   JUEGO
# ===============================
def play_game(cam_manager):
    current_player = 'X'
    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"¡{'Empate' if winner == 'draw' else winner + ' ha ganado'}!")
            break
        if current_player == 'X':
            player_turn(current_player, cam_manager)
        else:
            move = best_move(board, current_player)
            if move:
                print(f"IA juega: {move}")
                apply_move(board, move, current_player)
            else:
                print("No hay movimientos válidos para la IA")
                break
        current_player = 'O' if current_player == 'X' else 'X'

# ===============================
#   INTERFAZ TKINTER
# ===============================
def start_gui(cam_manager):
    root = tk.Tk()
    root.title("Control de Cámara")

    def cambiar_cam():
        cam_manager.next_camera()

    btn = tk.Button(root, text="Cambiar cámara", command=cambiar_cam, font=("Arial", 14))
    btn.pack(padx=20, pady=20)

    # ejecutar el juego en hilo aparte
    Thread(target=play_game, args=(cam_manager,), daemon=True).start()

    root.mainloop()

# ===============================
#   MAIN
# ===============================
if __name__ == "__main__":
    cam_manager = CameraManager(0, 3)
    try:
        start_gui(cam_manager)
    finally:
        if arduino is not None:
            arduino.close()
        cam_manager.release()
        cv2.destroyAllWindows()

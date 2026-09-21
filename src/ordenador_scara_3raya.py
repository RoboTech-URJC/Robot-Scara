"""
Developers:
            Adrián Manzanares ->  Github: Amanza17
            Justo Darío ->  Github: JustoDario
            Marcos Moreno -> Github: Marcox300

"""

import copy
import serial
import time
import cv2
import numpy as np

CAM = 2

# ===== CONFIGURACIÓN SERIAL =====
try:
    arduino = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
    time.sleep(2)  # Espera a que Arduino se inicialice
except serial.SerialException as e:
    print(f"❌ Error al conectar con Arduino: {e}")
    arduino = None

# ===== TABLERO Y CONTADOR =====
board = [[None, None, None] for _ in range(3)]
pieces = {'X': 0, 'O': 0}

# ===== POSICIONES FÍSICAS =====
ORIGIN = (25, 0, 0)  # Donde están las piezas sin colocar

# Mapa de coordenadas por casilla: fila, columna
COORDS = {
    (0, 0): (-675, 1050, -820),
    (0, 1): (-810, 1290, -820),
    (0, 2): (-910, 1530, -820),
    (1, 0): (-750, 1000, -820),
    (1, 1): (-880, 1200, -820),
    (1, 2): (-1000, 1350, -820),
    (2, 0): (-790, 710, -820),
    (2, 1): (-960, 900, -820),
    (2, 2): (-1130, 1020, -820),
}

# ===== CONSTANTES DEL ELECTROIMÁN =====
# Estos son los strings que tus becarios deben añadir en el "case" del Arduino
CMD_IMAN_ON = "Y ENCENDER Y"  
CMD_IMAN_OFF = "Z APAGAR Z"

# Alturas Z para la rutina
Z_SEGURO = 0      # Altura para moverse lateralmente sin chocar
Z_ABAJO = -820    # Altura del tablero/fichas para coger o dejar

# ===== FUNCIONES GENERALES =====
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

# ===== IA (MINIMAX CON ALPHA-BETA PRUNING) =====
def minimax(b, player, depth, maximizing, alpha, beta, local_pieces):
    winner = check_winner(b)
    if winner == 'X': return 1
    elif winner == 'O': return -1
    elif winner == 'draw' or depth == 0: return 0

    if maximizing:
        max_eval = -float('inf')
        for move in get_possible_moves(b, 'X', local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, 'X', real=False, local_pieces=new_pieces)
            eval = minimax(new_board, 'O', depth-1, False, alpha, beta, new_pieces)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_possible_moves(b, 'O', local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, 'O', real=False, local_pieces=new_pieces)
            eval = minimax(new_board, 'X', depth-1, True, alpha, beta, new_pieces)
            min_eval = min(min_eval, eval)
            beta = min(min_eval, eval)
            if beta <= alpha: break
        return min_eval

def best_move(b, player):
    moves = get_possible_moves(b, player, pieces)
    if not moves: return None
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

# ===== COMUNICACIÓN SERIAL MODIFICADA =====
def send_to_arduino(data):
    if arduino is None:
        print(f"❌ Arduino no conectado, omitiendo envío: {data}")
        return
    try:
        # Detecta si es una tupla de coordenadas o un comando string
        if isinstance(data, (tuple, list)):
            linea = f"{data[0]} {data[1]} {data[2]}"
        else:
            linea = str(data)
            
        print(f"[SERIAL] Enviando: {linea}")
        arduino.write((linea + '\n').encode())
        
        # Mantenemos la pausa de Python para dar tiempo al movimiento físico
        time.sleep(2) 
    except serial.SerialException as e:
        print(f"❌ Error al enviar a Arduino: {e}")

# ===== VISIÓN POR COMPUTADORA =====
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

def mouse_callback(event, x, y, flags, param):
    global confirm_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        button_x, button_y, button_w, button_h = param
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            confirm_clicked = True

# ===== JUEGO =====
def player_turn(player):
    global board, pieces, confirm_clicked
    print(f"\nTurno de {player}")
    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return None

    old_board = copy.deepcopy(board)
    print("Coloca o mueve tu ficha azul. Presiona Enter o haz clic en 'Confirmar' cuando hayas terminado.")

    start_time = time.time()
    timeout = 60
    confirm_clicked = False
    window_name = "Coloca tu ficha azul"
    cv2.namedWindow(window_name)

    button_x, button_y, button_w, button_h = 10, 10, 100, 40
    cv2.setMouseCallback(window_name, mouse_callback, (button_x, button_y, button_w, button_h))

    while True:
        ret, frame = cap.read()
        if not ret: break
        board_roi, detected_pieces = find_pieces_on_board(frame)
        
        if board_roi:
            x, y, w, h = board_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cell_w, cell_h = w // 3, h // 3
            for i, j in detected_pieces:
                cx = x + cell_w // 2 + j * cell_w
                cy = y + cell_h // 2 + i * cell_h
                cv2.circle(frame, (cx, cy), 15, (255, 255, 0), -1)

        cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 255, 0), -1)
        cv2.putText(frame, "Confirmar", (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10)
        if key == 13 or confirm_clicked:
            confirm_clicked = False
            ret, frame = cap.read()
            if not ret: break

            _, final_detected_pieces = find_pieces_on_board(frame)
            new_board = [[None, None, None] for _ in range(3)]
            for (i, j) in final_detected_pieces:
                if 0 <= i < 3 and 0 <= j < 3: new_board[i][j] = 'X'

            placed_at, removed_from = [], []
            for r in range(3):
                for c in range(3):
                    if old_board[r][c] is None and new_board[r][c] == 'X': placed_at.append((r, c))
                    elif old_board[r][c] == 'X' and new_board[r][c] is None: removed_from.append((r, c))

            if not placed_at and not removed_from:
                print("❌ No se detectó ningún cambio en el tablero. Turno saltado.")
                break

            if pieces['X'] < 3:
                if len(placed_at) == 1 and not removed_from:
                    i, j = placed_at[0]
                    if board[i][j] is None:
                        board[i][j] = 'X'
                        pieces['X'] += 1
                        print(f"✅ Ficha colocada en ({i},{j}). Total de fichas: {pieces['X']}")
                        break
            else:
                if len(placed_at) == 1 and len(removed_from) == 1:
                    i_removed, j_removed = removed_from[0]
                    i_placed, j_placed = placed_at[0]
                    if board[i_placed][j_placed] is None:
                        board[i_removed][j_removed] = None
                        board[i_placed][j_placed] = 'X'
                        print(f"✅ Ficha movida de ({i_removed},{j_removed}) a ({i_placed},{j_placed}).")
                        break

            break
        elif key == ord('q'): exit()
        if time.time() - start_time > timeout: break

    cap.release()
    cv2.destroyAllWindows()
    return None

# ===== RUTINA DE PICK & PLACE VERDADERA =====
def execute_pick_and_place(move):
    # Separamos el ORIGIN para usar sus X e Y, pero poder controlar la altura (Z) libremente
    origen_x, origen_y, _ = ORIGIN

    if move[0] == 'place':
        _, i, j = move
        dest = COORDS[(i, j)]
        print(f"[PICK&PLACE] IA coge nueva ficha y coloca en {i},{j}")
        
        # --- COGER FICHA ---
        send_to_arduino((origen_x, origen_y, Z_SEGURO))   # 1. Posicionarse sobre el origen
        send_to_arduino((origen_x, origen_y, Z_ABAJO))    # 2. Bajar
        send_to_arduino(CMD_IMAN_ON)                      # 3. Encender imán
        send_to_arduino("X 1000 X")                       # *. Pausa en Arduino de 1 seg para asegurar el agarre
        send_to_arduino((origen_x, origen_y, Z_SEGURO))   # 4. Subir con la ficha

        # --- DEJAR FICHA ---
        send_to_arduino((dest[0], dest[1], Z_SEGURO))     # 5. Moverse sobre el destino
        send_to_arduino((dest[0], dest[1], Z_ABAJO))      # 6. Bajar al tablero
        send_to_arduino(CMD_IMAN_OFF)                     # 7. Apagar imán
        send_to_arduino("X 1000 X")                       # *. Pausa en Arduino para asegurar que la suelta
        send_to_arduino((dest[0], dest[1], Z_SEGURO))     # 8. Subir

        # --- RETORNO ---
        send_to_arduino((origen_x, origen_y, Z_SEGURO))   # 9. Volver a standby

    elif move[0] == 'move':
        _, i, j, x, y = move
        start = COORDS[(i, j)]
        dest = COORDS[(x, y)]
        print(f"[PICK&PLACE] IA mueve ficha de ({i},{j}) a ({x},{y})")
        
        # --- COGER FICHA DEL TABLERO ---
        send_to_arduino((start[0], start[1], Z_SEGURO))   # 1. Posicionarse sobre la ficha a mover
        send_to_arduino((start[0], start[1], Z_ABAJO))    # 2. Bajar
        send_to_arduino(CMD_IMAN_ON)                      # 3. Encender imán
        send_to_arduino("X 1000 X")                       # *. Pausa en Arduino
        send_to_arduino((start[0], start[1], Z_SEGURO))   # 4. Subir con la ficha

        # --- DEJAR FICHA EN NUEVA CASILLA ---
        send_to_arduino((dest[0], dest[1], Z_SEGURO))     # 5. Moverse sobre el destino
        send_to_arduino((dest[0], dest[1], Z_ABAJO))      # 6. Bajar al tablero
        send_to_arduino(CMD_IMAN_OFF)                     # 7. Apagar imán
        send_to_arduino("X 1000 X")                       # *. Pausa en Arduino
        send_to_arduino((dest[0], dest[1], Z_SEGURO))     # 8. Subir

        # --- RETORNO ---
        send_to_arduino((origen_x, origen_y, Z_SEGURO))   # 9. Volver a standby

def play_game():
    current_player = 'X'  # Humano
    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"¡{'Empate' if winner == 'draw' else winner + ' ha ganado'}!")
            break

        if current_player == 'X':
            player_turn(current_player)
        else:
            move = best_move(board, current_player)
            if move:
                print(f"IA ({current_player}) juega: {move}")
                execute_pick_and_place(move)
                apply_move(board, move, current_player)
            else:
                print("No hay movimientos válidos para la IA")
                break

        current_player = 'O' if current_player == 'X' else 'X'

if __name__ == "__main__":
    try:
        play_game()
    finally:
        if arduino is not None:
            arduino.close()
        cv2.destroyAllWindows()

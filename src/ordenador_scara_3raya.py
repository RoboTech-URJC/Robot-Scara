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

CAM = 0

# ===== CONFIGURACIÓN DE VISIÓN (MODIFICADA PARA ROBUSTEZ) =====

RED_HSV_LOW_1 = np.array([0, 70, 50])    # Rangos más amplios estándar
RED_HSV_HIGH_1 = np.array([10, 255, 255])
RED_HSV_LOW_2 = np.array([170, 70, 50])
RED_HSV_HIGH_2 = np.array([180, 255, 255])

BLUE_HSV_LOW = np.array([90, 58, 250])   # Rango azul más estándar y robusto
BLUE_HSV_HIGH = np.array([120, 121, 255])

# Parámetros geométricos para el tablero (NUEVO)
BOARD_MIN_AREA = 20000      # Área mínima en píxeles que debe tener el tablero en imagen
BOARD_ASPECT_RATIO_TOL = 0.2 # Tolerancia de aspecto (cuadrado = 1.0, permitimos 0.8 a 1.2)
CANNY_LOW = 50              # Umbral bajo Canny
CANNY_HIGH = 150            # Umbral alto Canny

BLUE_MIN_AREA = 500
BLUE_MAX_AREA = 10000       # Aumentado un poco por si acaso
BLUE_MORPH_KERNEL = 5

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
primera_accion_robot = True

# ===== POSICIONES FÍSICAS =====
ORIGIN = (0, 0, 0)  # Donde están las piezas sin colocar

# Mapa de coordenadas por casilla: fila, columna
COORDS = {
    (0, 0): (-730, 975, 1000),
    (0, 1): (-820, 1175, 1000),
    (0, 2): (-975, 1350, 1000),
    (1, 0): (-800, 800, 1000),
    (1, 1): (-950, 1000, 1000),
    (1, 2): (-1150, 1175, 1000),
    (2, 0): (-800, 600, 1000),
    (2, 1): (-975, 800, 1000),
    (2, 2): (-1150, 900, 1000),
}
# ===== CONSTANTES DEL ELECTROIMÁN =====

CMD_IMAN_ON = "Y ENCENDER Y"  
CMD_IMAN_OFF = "Z APAGAR Z"

# Alturas Z para la rutina
Z_SEGURO = 8000      # Altura para moverse lateralmente sin chocar
Z_ABAJO = 0    # Altura del tablero/fichas para coger o dejar (reducir para bajar mas)

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

# ===== IA  =====
def best_move(b, player):

    moves = get_possible_moves(b, player, pieces)
    if not moves: return None
    best_val = -float('inf') 
    best_m = moves[0]
    
    for move in moves:
        new_board = copy.deepcopy(b)
        new_pieces = copy.deepcopy(pieces)
        apply_move(new_board, move, player, real=False, local_pieces=new_pieces)
        
        # Llamamos a minimax para evaluar la respuesta del oponente ('O')
        val = minimax(new_board, 'O', 5, False, -float('inf'), float('inf'), new_pieces)
        
        if val > best_val:
            best_val = val
            best_m = move
            
    return best_m

# ===== COMUNICACIÓN SERIAL MODIFICADA =====
def send_to_arduino(data):
    if arduino is None:
        print(f"❌ Arduino no conectado, omitiendo envío: {data}")
        return False

    try:
        if isinstance(data, (tuple, list)):
            linea = f"{data[0]} {data[1]} {data[2]}"
        else:
            linea = str(data)

        print(f"[SERIAL] Enviando: {linea}")
        arduino.write((linea + '\n').encode())
        arduino.flush()

        tiempo_inicio = time.time()

        while True:
            if time.time() - tiempo_inicio > 30:
                print("❌ Timeout esperando DONE de Arduino.")
                return False

            if arduino.in_waiting > 0:
                respuesta = arduino.readline().decode('utf-8', errors='ignore').strip()

                if respuesta == "DONE":
                    print("[SERIAL] ✅ Comando completado por Arduino.")
                    return True

    except (serial.SerialException, OSError) as e:
        print(f"❌ Error en comunicación con Arduino: {e}")
        return False
def go_to_origin():
    print(f"[ROBOT] Volviendo a posición inicial {ORIGIN[0]} {ORIGIN[1]} {Z_ABAJO}...")
    send_to_arduino((ORIGIN[0], ORIGIN[1], Z_ABAJO))

# ===== VISIÓN POR COMPUTADORA =====
def find_pieces_on_board(frame):
    """
    Detecta el tablero de forma robusta usando geometría (bordes y contornos)
    en lugar de solo color HSV, resistiendo cambios de iluminación.
    Fondo negro + Palitos rojos = Alto contraste en escala de grises.
    """
    # 1. Preprocesamiento para detección de bordes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Suavizado para reducir ruido antes de Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Detección de Bordes Canny (independiente del color exacto, busca contraste)
    edged = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    
    # 3. Operaciones morfológicas para cerrar huecos en las líneas de los palitos
    kernel_board = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edged, kernel_board, iterations=1)
    # closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_board, iterations=2) # Alternativa

    # 4. Encontrar contornos de los bordes detectados
    contours_board, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar por área de mayor a menor
    contours_board = sorted(contours_board, key=cv2.contourArea, reverse=True)

    board_contour = None
    board_roi = None

    # 5. Buscar el contorno del tablero basado en geometría
    for cnt in contours_board:
        area = cv2.contourArea(cnt)
        if area < BOARD_MIN_AREA:
            break # Los siguientes serán muy pequeños

        # Aproximar el contorno a un polígono
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True) # 3% de tolerancia

        # El tablero debe ser un cuadrilátero (4 lados aprox)
        if len(approx) >= 4 and len(approx) <= 6:
            # Calcular bounding box y aspect ratio para asegurar que sea cuadrado
            bx, by, bw, bh = cv2.boundingRect(approx)
            aspect_ratio = float(bw) / bh
            
            # Verificar si es aproximadamente cuadrado
            if (1.0 - BOARD_ASPECT_RATIO_TOL) <= aspect_ratio <= (1.0 + BOARD_ASPECT_RATIO_TOL):
                
                # OPCIONAL: Aquí podrías añadir una validación HSV rápida dentro del 
                # bounding box para asegurar que hay algo "rojo", pero normalmente
                # si el fondo es negro y hay un cuadrado grande, será el tablero.
                
                board_contour = approx
                board_roi = (bx, by, bw, bh)
                break # Encontramos el contorno más grande que cumple requisitos

    # Si no detectamos tablero geométricamente, salimos
    if board_roi is None:
        return None, []

    x, y, w, h = board_roi

    # === DETECCIÓN DE FICHAS AZULES (DENTRO DEL TABLERO DETECTADO) ===
    # Usamos HSV aquí porque el usuario dice que funciona bien, 
    # pero limitamos la búsqueda al ROI del tablero.
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear máscara azul mejorada
    mask_azul = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)
    
    # Limpieza morfológica para fichas
    kernel_azul = np.ones((BLUE_MORPH_KERNEL, BLUE_MORPH_KERNEL), np.uint8)
    mask_azul = cv2.morphologyEx(mask_azul, cv2.MORPH_OPEN, kernel_azul)
    mask_azul = cv2.morphologyEx(mask_azul, cv2.MORPH_CLOSE, kernel_azul)

    # Encontrar contornos azules
    contours_azul, _ = cv2.findContours(mask_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_piece_coords = []
    
    # Dividir el ROI del tablero en cuadrícula 3x3 ideal
    cell_w = w // 3
    cell_h = h // 3

    if cell_w <= 0 or cell_h <= 0:
        return board_roi, []

    for cnt in contours_azul:
        area = cv2.contourArea(cnt)

        # Filtrar por área de ficha
        if not (BLUE_MIN_AREA < area < BLUE_MAX_AREA):
            continue

        # Calcular centroide
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Verificar si el centroide de la ficha azul está dentro del tablero detectado
        if x <= cx < x + w and y <= cy < y + h:
            # Calcular a qué casilla pertenece (0,1,2)
            # Usamos lógica relativa al borde exterior del tablero
            j = (cx - x) // cell_w
            i = (cy - y) // cell_h

            # Asegurar límites por redondeos
            i = max(0, min(2, i))
            j = max(0, min(2, j))

            blue_piece_coords.append((i, j))

    # Eliminamos duplicados de casillas si los hubiera por ruido
    blue_piece_coords = list(set(blue_piece_coords))

    # Devolvemos el ROI del tablero (para dibujar la caja) y la lista de coordenadas (fila, col)
    return board_roi, blue_piece_coords



def mouse_callback(event, x, y, flags, param):
    global confirm_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        button_x, button_y, button_w, button_h = param
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            confirm_clicked = True

# ===== JUEGo - Jugador =====
# ===== JUEGo - Jugador =====
def player_turn(player):
    global board, pieces, confirm_clicked
    print(f"\nTurno de {player} (Humano)")
    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return False # MODIFICADO: Retorna False en caso de error

    old_board = copy.deepcopy(board)
    print("Coloca o mueve tu ficha azul (O). Presiona 'Enter' o haz clic en 'Confirmar' cuando hayas terminado.")

    start_time = time.time()
    timeout = 60
    confirm_clicked = False
    window_name = "Coloca tu ficha azul"
    cv2.namedWindow(window_name)

    button_x, button_y, button_w, button_h = 10, 10, 100, 40
    cv2.setMouseCallback(window_name, mouse_callback, (button_x, button_y, button_w, button_h))

    # Limpiar buffer de teclado de OpenCV al iniciar el turno
    while cv2.waitKey(1) != -1:
        pass
        
    turno_exitoso = False  # NUEVO: Bandera para saber si el turno se completó bien

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("⚠️ Error leyendo cámara (Posible desconexión). Cancelando turno actual.")
            break # Sale del bucle, turno_exitoso sigue siendo False
        
        board_roi, detected_pieces = find_pieces_on_board(frame)
        
        if board_roi:
            x, y, w, h = board_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cell_w, cell_h = w // 3, h // 3
            for i, j in detected_pieces:
                cx = x + cell_w // 2 + j * cell_w
                cy = y + cell_h // 2 + i * cell_h
                cv2.circle(frame, (cx, cy), 15, (255, 255, 0), -1)

        # Dibujar botón de confirmar
        cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 255, 0), -1)
        cv2.putText(frame, "Confirmar", (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10)
        
        # Esperar a que el usuario presione Enter (13) o haga clic en el botón
        if key == 13 or confirm_clicked:
            confirm_clicked = False
            print("Evaluando tablero tras confirmación...")
            
            # Tomar un nuevo frame para evaluar la posición final
            ret, frame = cap.read()
            if not ret: break

            _, final_detected_pieces = find_pieces_on_board(frame)
            new_board = [[None, None, None] for _ in range(3)]
            
            # Asignar 'O' a las posiciones donde la cámara ve fichas azules
            for (i, j) in final_detected_pieces:
                if 0 <= i < 3 and 0 <= j < 3: 
                    new_board[i][j] = 'O' 

            placed_at, removed_from = [], []
            for r in range(3):
                for c in range(3):
                    # Comparar el tablero actual (old_board) con lo que ve la cámara (new_board)
                    if old_board[r][c] is None and new_board[r][c] == 'O': 
                        placed_at.append((r, c))
                    elif old_board[r][c] == 'O' and new_board[r][c] is None: 
                        removed_from.append((r, c))

            if not placed_at and not removed_from:
                print("❌ No se detectó ningún cambio en el tablero. Sigue siendo tu turno.")
                # NO salimos del bucle (quitamos el break). El usuario debe intentar de nuevo.
                continue 

            # Lógica de colocación (Fase 1: menos de 3 fichas)
            if pieces['O'] < 3:
                if len(placed_at) == 1 and not removed_from:
                    i, j = placed_at[0]
                    if board[i][j] is None:
                        board[i][j] = 'O'
                        pieces['O'] += 1
                        print(f"✅ Ficha (O) colocada en ({i},{j}). Total de tus fichas: {pieces['O']}")
                        turno_exitoso = True # MARCAMOS ÉXITO
                        break # Movimiento válido, salimos del turno
                else:
                    print("❌ Movimiento inválido. Debes colocar exactamente UNA ficha nueva.")
                    continue
            
            # Lógica de movimiento (Fase 2: 3 fichas, hay que mover)
            else:
                if len(placed_at) == 1 and len(removed_from) == 1:
                    i_removed, j_removed = removed_from[0]
                    i_placed, j_placed = placed_at[0]
                    if board[i_placed][j_placed] is None:
                        board[i_removed][j_removed] = None
                        board[i_placed][j_placed] = 'O'
                        print(f"✅ Ficha (O) movida de ({i_removed},{j_removed}) a ({i_placed},{j_placed}).")
                        turno_exitoso = True # MARCAMOS ÉXITO
                        break # Movimiento válido, salimos del turno
                else:
                     print("❌ Movimiento inválido. Debes mover exactamente UNA de tus fichas a una casilla vacía.")
                     continue

        elif key == ord('q'):
            raise KeyboardInterrupt
        
        if time.time() - start_time > timeout: 
            print("⏱️ Tiempo de turno agotado.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return turno_exitoso # MODIFICADO: Retornamos si el turno se hizo bien o no

def execute_pick_and_place(move):

    # Separamos el ORIGIN para usar sus X e Y, pero poder controlar la altura (Z) libremente

    origen_x, origen_y, _ = ORIGIN



    if move[0] == 'place':

        _, i, j = move

        dest = COORDS[(i, j)]

        print(f"[PICK&PLACE] IA coge nueva ficha y coloca en {i},{j}")


        # --- COGER FICHA ---

        send_to_arduino((origen_x, origen_y, Z_SEGURO)) # 1. Posicionarse sobre el origen

        send_to_arduino((origen_x, origen_y, Z_ABAJO)) # 2. Bajar

        send_to_arduino(CMD_IMAN_ON) # 3. Encender imán

        send_to_arduino("X 1000 X") # *. Pausa en Arduino de 1 seg para asegurar el agarre

        send_to_arduino((origen_x, origen_y, Z_SEGURO)) # 4. Subir con la ficha



        # --- DEJAR FICHA ---

        send_to_arduino((dest[0], dest[1], Z_SEGURO)) # 5. Moverse sobre el destino

        send_to_arduino((dest[0], dest[1], Z_ABAJO)) # 6. Bajar al tablero

        send_to_arduino(CMD_IMAN_OFF) # 7. Apagar imán

        send_to_arduino("X 1000 X") # *. Pausa en Arduino para asegurar que la suelta

        send_to_arduino((dest[0], dest[1], Z_SEGURO)) # 8. Subir



        # --- RETORNO ---

        send_to_arduino((origen_x, origen_y, Z_SEGURO)) # 9. Volver a standby



    elif move[0] == 'move':

        _, i, j, x, y = move

        start = COORDS[(i, j)]

        dest = COORDS[(x, y)]

        print(f"[PICK&PLACE] IA mueve ficha de ({i},{j}) a ({x},{y})")


        # --- COGER FICHA DEL TABLERO ---

        send_to_arduino((start[0], start[1], Z_SEGURO)) # 1. Posicionarse sobre la ficha a mover

        send_to_arduino((start[0], start[1], Z_ABAJO)) # 2. Bajar

        send_to_arduino(CMD_IMAN_ON) # 3. Encender imán

        send_to_arduino("X 1000 X") # *. Pausa en Arduino

        send_to_arduino((start[0], start[1], Z_SEGURO)) # 4. Subir con la ficha



        # --- DEJAR FICHA EN NUEVA CASILLA ---

        send_to_arduino((dest[0], dest[1], Z_SEGURO)) # 5. Moverse sobre el destino

        send_to_arduino((dest[0], dest[1], Z_ABAJO)) # 6. Bajar al tablero

        send_to_arduino(CMD_IMAN_OFF) # 7. Apagar imán

        send_to_arduino("X 1000 X") # *. Pausa en Arduino

        send_to_arduino((dest[0], dest[1], Z_SEGURO)) # 8. Subir



        # --- RETORNO ---

        send_to_arduino((origen_x, origen_y, Z_SEGURO)) # 9. Volver a standby



def play_game():
    current_player = 'O'  # Humano ('O') empieza primero (o cambia a 'X' si quieres que empiece la IA)
    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"¡{'Empate' if winner == 'draw' else winner + ' ha ganado'}!")
            go_to_origin()
            break

        if current_player == 'O': # Turno del Humano ('O')
            exito = player_turn(current_player)
            if not exito:
                print("\n⚠️ El turno fue interrumpido (Cámara o Timeout). Reintentando turno...")
                continue # SALTA el resto del código y vuelve a iniciar el turno del Humano
        else:                     # Turno de la IA ('X')
            move = best_move(board, current_player)
            if move:
                print(f"IA ({current_player}) juega: {move}")
                execute_pick_and_place(move)
                apply_move(board, move, current_player)
            else:
                print("No hay movimientos válidos para la IA")
                go_to_origin()
                break

        # Alternar jugador SOLO si el turno se completó con éxito
        current_player = 'X' if current_player == 'O' else 'O'

if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n⚠️ Programa interrumpido. Volviendo a posición inicial...")
        go_to_origin()
    finally:
        if arduino is not None:
            arduino.close()
        cv2.destroyAllWindows()
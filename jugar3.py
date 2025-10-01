import copy
import serial
import time
import cv2
import numpy as np

# ===== CONFIGURACIÓN GLOBAL =====
CAM_ID = 2  # ID de la cámara
PLAYER_HUMAN = 'X'
PLAYER_AI = 'O'
MAX_DEPTH = 6  # Profundidad para el algoritmo Minimax (ajustable)

# ===== CONFIGURACIÓN SERIAL Y ROBÓTICA (XYZ) =====
# Nota: La configuración física sigue siendo la misma.
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
ORIGIN = (25, 0, 0)  # Posición inicial o de reposo del brazo

# Mapa de coordenadas por casilla: (fila, columna) -> (x, y, z)
COORDS_ROBOT = {
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

# Configuración de detección de colores (HSV)
# Se han ajustado los valores para una detección más precisa y robusta
COLOR_RANGES = {
    # ROJO (Piezas de la IA 'O') - Utilizando dos rangos para capturar el rojo completo
    'O': [
        (np.array([0, 100, 50]), np.array([10, 255, 255])),
        (np.array([170, 100, 50]), np.array([180, 255, 255]))
    ],
    # AZUL (Piezas Humanas 'X')
    'X': [
        (np.array([90, 100, 50]), np.array([130, 255, 255]))
    ]
}

# ===== CLASE ROBOT (Para encapsular la lógica de comunicación serial) =====
class RobotController:
    """Maneja la conexión y comunicación con el brazo robótico (Arduino)."""
    def __init__(self, port, baud_rate):
        self.arduino = None
        try:
            self.arduino = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)  # Espera a que Arduino se inicialice
            print("✅ Conexión con Arduino establecida.")
        except serial.SerialException as e:
            print(f"❌ Error al conectar con Arduino en {port}: {e}")

    def send_coords(self, coords):
        """Envía coordenadas (x, y, z) al Arduino."""
        if self.arduino is None:
            print("❌ Arduino no conectado, omitiendo envío.")
            return

        try:
            line = f"{coords[0]} {coords[1]} {coords[2]}"
            print(f"[SERIAL] Enviando: {line}")
            self.arduino.write((line + '\n').encode('utf-8'))
            
            # **MEJORA:** Espera confirmación o un tiempo más corto.
            # Aquí se asume que el Arduino procesa en menos de 1 segundo.
            # En un sistema real, se debería esperar una respuesta de 'OK' del Arduino.
            time.sleep(1) 
        except serial.SerialException as e:
            print(f"❌ Error al enviar a Arduino: {e}")

    def close(self):
        """Cierra la conexión serial."""
        if self.arduino:
            self.arduino.close()
            print("✅ Conexión con Arduino cerrada.")

# ===== LÓGICA DEL JUEGO (GAME LOGIC) =====

def print_board(b):
    """Imprime el tablero de forma más clara."""
    print("\n  | 0 | 1 | 2 |")
    print("--+---+---+---+")
    for i, row in enumerate(b):
        print(f"{i} | {' | '.join(c if c else ' ' for c in row)} |")
        print("--+---+---+---+")
    print()

def check_winner(b):
    """Verifica si hay un ganador o un empate."""
    # Filas, Columnas
    lines = b + list(map(list, zip(*b)))
    # Diagonales
    lines.append([b[i][i] for i in range(3)])
    lines.append([b[i][2-i] for i in range(3)])
    
    for line in lines:
        if line[0] and all(c == line[0] for c in line):
            return line[0]
    
    # Empate
    if all(None not in row for row in b):
        return 'draw'
        
    return None

def get_possible_moves(b, player, local_pieces):
    """Genera todos los movimientos válidos para el jugador actual."""
    moves = []
    
    # 1. Fase de COLOCACIÓN (hasta 3 piezas)
    if local_pieces[player] < 3:
        for i in range(3):
            for j in range(3):
                if b[i][j] is None:
                    # Formato: ('place', fila, columna)
                    moves.append(('place', i, j))
    
    # 2. Fase de MOVIMIENTO (después de 3 piezas)
    else:
        for i in range(3):
            for j in range(3):
                if b[i][j] == player: # Ficha a mover
                    for x in range(3):
                        for y in range(3):
                            if b[x][y] is None: # Destino vacío
                                # Formato: ('move', from_row, from_col, to_row, to_col)
                                moves.append(('move', i, j, x, y))
    return moves

def apply_move(b, move, player, update_pieces=True, local_pieces=None):
    """Aplica un movimiento al tablero y opcionalmente actualiza el contador de piezas."""
    # Usar el diccionario global si no se pasa uno local
    pieces_to_update = local_pieces if local_pieces is not None else globals()['pieces']
    
    if move[0] == 'place':
        _, i, j = move
        if b[i][j] is None:
            b[i][j] = player
            if update_pieces:
                pieces_to_update[player] += 1
    elif move[0] == 'move':
        _, i, j, x, y = move
        if b[i][j] == player and b[x][y] is None:
            b[i][j] = None
            b[x][y] = player
    
    # El retorno del tablero (modificado in-place) no es estrictamente necesario, pero es un buen patrón.
    return b

# ===== IA (MINIMAX CON ALPHA-BETA PRUNING) - Optimizado y mejor manejado =====

def minimax(b, player, depth, maximizing, alpha, beta, local_pieces):
    """Implementación de Minimax con poda Alpha-Beta."""
    
    winner = check_winner(b)
    if winner is not None:
        if winner == PLAYER_AI: # IA gana
            return 1 + depth # Prefiere ganar antes y más rápido
        elif winner == PLAYER_HUMAN: # Humano gana
            return -1 - depth # Penaliza perder antes y más rápido
        else: # Empate o fin de profundidad
            return 0
    
    if depth == 0:
        return 0

    current_player = PLAYER_AI if maximizing else PLAYER_HUMAN
    
    if maximizing:
        max_eval = -float('inf')
        for move in get_possible_moves(b, current_player, local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, current_player, update_pieces=True, local_pieces=new_pieces)
            
            # El siguiente jugador es el que minimiza
            eval_val = minimax(new_board, PLAYER_HUMAN, depth - 1, False, alpha, beta, new_pieces)
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval
    
    else: # Minimizing (Turno del Humano)
        min_eval = float('inf')
        for move in get_possible_moves(b, current_player, local_pieces):
            new_board = copy.deepcopy(b)
            new_pieces = copy.deepcopy(local_pieces)
            apply_move(new_board, move, current_player, update_pieces=True, local_pieces=new_pieces)
            
            # El siguiente jugador es el que maximiza
            eval_val = minimax(new_board, PLAYER_AI, depth - 1, True, alpha, beta, new_pieces)
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val) # Corregido: 'beta = min(beta, eval_val)'
            if beta <= alpha:
                break
        return min_eval

def get_best_move(b, player, pieces_counter):
    """Función principal para encontrar el mejor movimiento para la IA."""
    moves = get_possible_moves(b, player, pieces_counter)
    if not moves:
        return None
    
    is_maximizing = (player == PLAYER_AI)
    best_val = -float('inf') if is_maximizing else float('inf')
    best_m = moves[0]

    # El siguiente jugador es el opuesto
    next_player = PLAYER_HUMAN if player == PLAYER_AI else PLAYER_AI
    
    # La llamada inicial al minimax debe ser para el *siguiente* estado, 
    # por lo que el parámetro 'maximizing' debe ser para el siguiente jugador.
    next_maximizing = not is_maximizing

    print(f"Calculando mejor movimiento para {player}...")
    
    for move in moves:
        new_board = copy.deepcopy(b)
        new_pieces = copy.deepcopy(pieces_counter)
        apply_move(new_board, move, player, update_pieces=True, local_pieces=new_pieces)
        
        # Evalúa el estado resultante. Si el jugador actual es la IA (maximizador), 
        # la siguiente llamada será para el humano (minimizador).
        val = minimax(new_board, next_player, MAX_DEPTH, next_maximizing, -float('inf'), float('inf'), new_pieces)
        
        # Actualiza el mejor movimiento
        if (is_maximizing and val > best_val) or (not is_maximizing and val < best_val):
            best_val = val
            best_m = move
            print(f"  Nuevo mejor movimiento: {best_m} con valor: {best_val}")

    print(f"Mejor valor final: {best_val}")
    return best_m

# ===== VISIÓN POR COMPUTADORA (CV) - Mejorado para robustez =====

def find_board_and_pieces(frame):
    """
    Detecta la región del tablero (la pieza roja más grande) y las fichas azules ('X').
    
    Retorna:
    - board_roi (tuple): (x, y, w, h) del tablero, o None.
    - piece_coords (list): Lista de (fila, columna) de las piezas detectadas.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Detección del Tablero (usando el color 'O' de la IA como referencia del límite)
    board_mask = cv2.inRange(hsv, *COLOR_RANGES['O'][0]) | cv2.inRange(hsv, *COLOR_RANGES['O'][1])
    
    contours_board, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_board:
        return None, []
        
    # Encuentra el contorno más grande (asumiendo que es el tablero)
    c = max(contours_board, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    # Filtro de área mínima para evitar ruido
    if area < 1000: 
         return None, []

    x, y, w, h = cv2.boundingRect(c)
    board_roi = (x, y, w, h)
    
    # 2. Detección de Piezas Azules (Humanas 'X')
    # Uso de la función mejorada de find_contours para las piezas
    mask_azul = cv2.inRange(hsv, *COLOR_RANGES['X'][0])
    contours_azul, _ = cv2.findContours(mask_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blue_piece_coords = []
    
    # Calcular dimensiones de celda *dentro* del ROI del tablero
    cell_w = w // 3
    cell_h = h // 3
    
    for cnt in contours_azul:
        area_piece = cv2.contourArea(cnt)
        # Filtro de área específico para piezas
        if 200 < area_piece < 10000: 
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Mapear el centro de la pieza a una celda del tablero
                if x <= cx <= x + w and y <= cy <= y + h:
                    i = (cy - y) // cell_h
                    j = (cx - x) // cell_w
                    
                    # Validación de límites de celda
                    if 0 <= i < 3 and 0 <= j < 3:
                        blue_piece_coords.append((i, j))
                        
    # Eliminar duplicados si hay (por ejemplo, por detección muy cercana)
    blue_piece_coords = sorted(list(set(blue_piece_coords)))
    
    return board_roi, blue_piece_coords

def get_board_state_from_vision(pieces_of_interest):
    """
    Captura una imagen y devuelve el estado del tablero basado en las piezas detectadas.
    
    Retorna:
    - new_board (list of lists): Tablero 3x3 con 'X' o None.
    - cap (cv2.VideoCapture): Objeto de la cámara abierto (para cerrar después).
    """
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return None, None
        
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al capturar imagen")
        cap.release()
        return None, None

    board_roi, detected_coords = find_board_and_pieces(frame)
    
    new_board = [[None, None, None] for _ in range(3)]
    for (i, j) in detected_coords:
        # Aquí solo se detectan las piezas humanas ('X')
        new_board[i][j] = pieces_of_interest 
        
    return new_board, cap


# ===== JUEGO Y FLUJO DE CONTROL PRINCIPAL =====

def player_turn(player, current_board, pieces_counter, robot_controller):
    """Maneja el turno del jugador humano (incluye CV para validar el movimiento)."""
    print(f"\n--- Turno de {player} (Humano) ---")
    
    # 1. Copia del estado inicial del tablero
    old_board = copy.deepcopy(current_board) 
    
    print_board(old_board)
    print("Coloca o mueve tu ficha. La cámara está en vivo. Presiona ENTER (o 'q' para salir).")
    
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara para la vista previa.")
        return False # Fallo en el turno

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al capturar imagen en bucle de vista previa.")
                break

            board_roi, detected_pieces = find_board_and_pieces(frame)
            
            # **MEJORA:** Dibujar la cuadrícula para la vista previa
            if board_roi:
                x, y, w, h = board_roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cell_w, cell_h = w // 3, h // 3
                
                for i in range(1, 3): # Dibujar líneas de la cuadrícula
                    cv2.line(frame, (x + i * cell_w, y), (x + i * cell_w, y + h), (0, 255, 0), 1)
                    cv2.line(frame, (x, y + i * cell_h), (x + w, y + i * cell_h), (0, 255, 0), 1)
                
                # Dibujar círculos en el centro de las piezas detectadas
                for i, j in detected_pieces:
                    cx = x + cell_w // 2 + j * cell_w
                    cy = y + cell_h // 2 + i * cell_h
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1) # Amarillo para detectado

            cv2.imshow("Coloca tu ficha", frame)

            key = cv2.waitKey(1)
            if key == 13:  # Enter
                print("Detectando movimiento...")
                # Repetir la captura final para asegurar la imagen estática
                cap.release()
                cv2.destroyAllWindows()
                
                # 2. Captura del estado final
                final_board_state, _ = get_board_state_from_vision(PLAYER_HUMAN)
                if final_board_state is None:
                    print("❌ Fallo en la captura final. Intenta de nuevo.")
                    continue
                
                # 3. Determinar el movimiento humano comparando estados (solo 'X')
                placed_at = []
                removed_from = []
                
                for r in range(3):
                    for c in range(3):
                        is_now_human = final_board_state[r][c] == PLAYER_HUMAN
                        was_human = old_board[r][c] == PLAYER_HUMAN
                        
                        if not was_human and is_now_human:
                            placed_at.append((r, c)) # Nueva ficha en casilla vacía
                        elif was_human and not is_now_human:
                            removed_from.append((r, c)) # Ficha que desapareció de su posición

                # 4. Validar el movimiento
                valid_move = False
                
                if pieces_counter[PLAYER_HUMAN] < 3: # Fase de COLOCACIÓN
                    if len(placed_at) == 1 and not removed_from:
                        i, j = placed_at[0]
                        move_details = ('place', i, j)
                        valid_move = True
                        print(f"✅ Detección: Colocación en ({i},{j}).")
                    else:
                        print(f"❌ Error: Se esperaban 1 pieza nueva (colocación). Detectadas {len(placed_at)} nuevas y {len(removed_from)} eliminadas.")
                        
                else: # Fase de MOVIMIENTO
                    if len(placed_at) == 1 and len(removed_from) == 1:
                        i_removed, j_removed = removed_from[0]
                        i_placed, j_placed = placed_at[0]
                        move_details = ('move', i_removed, j_removed, i_placed, j_placed)
                        valid_move = True
                        print(f"✅ Detección: Movimiento de ({i_removed},{j_removed}) a ({i_placed},{j_placed}).")
                    else:
                        print(f"❌ Error: Se esperaba 1 pieza movida. Detectadas {len(placed_at)} nuevas y {len(removed_from)} eliminadas.")
                
                # 5. Aplicar el movimiento si es válido (y romper el bucle)
                if valid_move:
                    apply_move(current_board, move_details, PLAYER_HUMAN, update_pieces=True, local_pieces=pieces_counter)
                    return True # Turno exitoso
                else:
                    print("Por favor, corrige el movimiento en el tablero y presiona ENTER de nuevo.")
                    cap = cv2.VideoCapture(CAM_ID) # Reabrir la cámara
                    continue
                    
            elif key == ord('q'):
                exit()
                
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    
    return False

def execute_pick_and_place(move, robot_controller):
    """Ejecuta los comandos robóticos para el movimiento de la IA."""
    if move[0] == 'place':
        _, i, j = move
        dest = COORDS_ROBOT[(i, j)]
        print(f"[ROBOT] IA coloca ficha en {i},{j}")
        robot_controller.send_coords(ORIGIN) # Ir a recoger
        robot_controller.send_coords(dest)   # Colocar
        robot_controller.send_coords(ORIGIN) # Volver a reposo
    elif move[0] == 'move':
        _, i, j, x, y = move
        start = COORDS_ROBOT[(i, j)]
        dest = COORDS_ROBOT[(x, y)]
        print(f"[ROBOT] IA mueve ficha de ({i},{j}) a ({x},{y})")
        robot_controller.send_coords(start)  # Ir a recoger la pieza
        robot_controller.send_coords(dest)   # Soltar la pieza
        robot_controller.send_coords(ORIGIN) # Volver a reposo

def play_game():
    """Bucle principal del juego."""
    # Inicialización Global
    global board, pieces # Accede a las variables globales
    
    # Inicialización de la IA y el Robot
    robot = RobotController(SERIAL_PORT, BAUD_RATE)
    current_player = PLAYER_HUMAN
    
    print("--- INICIO DEL JUEGO: Tres en Raya Robótico ---")
    
    while True:
        print_board(board)
        winner = check_winner(board)
        
        if winner:
            print(f"==========================================")
            print(f"¡{'EMPATE' if winner == 'draw' else '¡' + winner + ' ha GANADO'}!")
            print(f"==========================================")
            break

        if current_player == PLAYER_HUMAN:
            # Pasa el objeto del robot por si se necesita en el futuro (ej. para calibración)
            turn_success = player_turn(current_player, board, pieces, robot)
            if not turn_success:
                print("❌ Error crítico durante el turno del jugador. Terminando juego.")
                break
        
        else: # Turno de la IA
            print(f"\n--- Turno de {current_player} (IA) ---")
            # El jugador IA tiene piezas en el tablero o disponibles?
            move = get_best_move(board, current_player, pieces)
            
            if move:
                print(f"IA ({current_player}) eligió el movimiento: {move}")
                execute_pick_and_place(move, robot)
                apply_move(board, move, current_player, update_pieces=True)
            else:
                print("No hay movimientos válidos para la IA. Terminando juego (posiblemente error de lógica).")
                break

        # Cambiar de turno
        current_player = PLAYER_HUMAN if current_player == PLAYER_AI else PLAYER_AI

    robot.close()

if __name__ == "__main__":
    # Inicialización del tablero y contadores
    board = [[None, None, None] for _ in range(3)]
    pieces = {PLAYER_HUMAN: 0, PLAYER_AI: 0}
    
    try:
        play_game()
    finally:
        cv2.destroyAllWindows() # Asegura que todas las ventanas de CV se cierren

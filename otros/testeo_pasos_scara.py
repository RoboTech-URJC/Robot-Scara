import serial
import time
import sys

# Lector de teclado en tiempo real (Multiplataforma). 
# Evita tener que instalar librerías externas o usar 'sudo' en Linux.
try:
    import msvcrt
    def obtener_tecla():
        return msvcrt.getch().decode('utf-8', errors='ignore').lower()
except ImportError:
    import tty, termios
    def obtener_tecla():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()

# ===== CONFIGURACIÓN SERIAL =====
PUERTO = "/dev/ttyACM0"
BAUD_RATE = 115200

def conectar_arduino():
    try:
        print(f"Conectando a {PUERTO}...")
        arduino = serial.Serial(PUERTO, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("✅ Conexión establecida.\n")
        return arduino
    except serial.SerialException as e:
        print(f"❌ Error al conectar: {e}")
        return None

def main():
    arduino = conectar_arduino()
    if not arduino:
        sys.exit(1)

    # Variables que almacenarán la posición absoluta de cada motor
    pos_x = 0
    pos_y = 0
    pos_z = 0

    # Estado del electroimán
    iman_encendido = False
    
    # Cuántos ticks quieres que avance el motor con cada pulsación de tecla
    SALTO = 10 
    SALTO2 = 100 

    print("=== CONTROL MANUAL DEL SCARA ===")
    print(" [A] / [D] -> Mover Eje X (A: -X | D: +X)")
    print(" [W] / [S] -> Mover Eje Y (S: -Y | W: +Y)")
    print(" [J] / [L] -> Movimiento grande X")
    print(" [I] / [K] -> Movimiento grande Y")
    print(" [R] / [F] -> Mover Eje Z (R: +Z | F: -Z)")
    print(" [T] / [G] -> Movimiento grande Z (T: +Z | G: -Z)")
    print(" [O] -> Encender / apagar electroimán")
    print(" [Q] -> Salir del programa")
    print("--------------------------------")

    try:
        while True:
            tecla = obtener_tecla()
            actualizar = False

            # Lógica de movimiento
            if tecla == 'a':
                pos_x -= SALTO
                actualizar = True
            elif tecla == 'd':
                pos_x += SALTO
                actualizar = True
            elif tecla == 'w':
                pos_y += SALTO
                actualizar = True
            elif tecla == 's':
                pos_y -= SALTO
                actualizar = True

            if tecla == 'j':
                pos_x -= SALTO2
                actualizar = True
            elif tecla == 'l':
                pos_x += SALTO2
                actualizar = True
            elif tecla == 'i':
                pos_y += SALTO2
                actualizar = True
            elif tecla == 'k':
                pos_y -= SALTO2
                actualizar = True

            # Movimiento del eje Z
            elif tecla == 'r':
                pos_z += SALTO
                actualizar = True
            elif tecla == 'f':
                pos_z -= SALTO
                actualizar = True
            elif tecla == 't':
                pos_z += SALTO2
                actualizar = True
            elif tecla == 'g':
                pos_z -= SALTO2
                actualizar = True

            # Toggle del electroimán
            elif tecla == 'o':
                iman_encendido = not iman_encendido

                if iman_encendido:
                    arduino.write(b"Y ENCENDER Y\n")
                    print("🧲 Electroimán -> ENCENDIDO")
                else:
                    arduino.write(b"Z APAGAR Z\n")
                    print("🧲 Electroimán -> APAGADO")

                time.sleep(0.05)

            elif tecla == 'q' or tecla == '\x03': # \x03 es la señal de Ctrl+C
                print("\nCerrando conexión...")
                break
                
            # Si se ha pulsado una tecla válida, enviamos los datos absolutos
            if actualizar:
                comando = f"{pos_x} {pos_y} {pos_z}"
                
                # Se imprime en pantalla la coordenada absoluta exacta
                print(f"Posición Absoluta -> X: {pos_x} | Y: {pos_y} | Z: {pos_z}")
                
                arduino.write((comando + '\n').encode())
                
                # Pequeña pausa para dar tiempo al Arduino a leer el buffer Serial
                time.sleep(0.05) 

    finally:
        if arduino:
            # Asegurarse de apagar el electroimán al salir
            if iman_encendido:
                arduino.write(b"Z APAGAR Z\n")
                time.sleep(0.05)

            arduino.close()
            print("Puerto serial cerrado correctamente.")

if __name__ == "__main__":
    main()
import serial
import time
import sys

# ===== CONFIGURACIÓN SERIAL =====
PUERTO = "/dev/ttyACM0"
BAUD_RATE = 115200

def conectar_arduino():
    try:
        print(f"Conectando a {PUERTO}...")
        arduino = serial.Serial(PUERTO, BAUD_RATE, timeout=1)
        time.sleep(2)  # Tiempo de inicialización
        print("✅ Conexión establecida.\n")
        return arduino
    except serial.SerialException as e:
        print(f"❌ Error al conectar con Arduino: {e}")
        return None

def enviar_posicion(arduino, x, y, z):
    comando = f"{x} {y} {z}"
    print(f"[SERIAL] Enviando: {comando}")
    arduino.write((comando + '\n').encode())
    time.sleep(0.5)

def main():
    arduino = conectar_arduino()
    
    if not arduino:
        sys.exit(1) # Salimos del programa con código de error

    print("=== TEST INDIVIDUAL DE MOTORES ===")
    print("Introduce los ticks o coordenadas (X Y Z) separados por un espacio.")
    print("Ejemplo: 1000 -500 200")
    print("Escribe 'q' para salir.\n")

    try:
        while True:
            entrada = input("Posición (X Y Z) > ").strip()
            
            if entrada.lower() in ['q', 'salir', 'exit']:
                print("Cerrando conexión...")
                break
                
            partes = entrada.split()
            
            if len(partes) == 3:
                try:
                    x = float(partes[0])
                    y = float(partes[1])
                    z = float(partes[2])
                    
                    enviar_posicion(arduino, x, y, z)
                    
                except ValueError:
                    print("❌ Error: Introduce solo números válidos.")
            else:
                print("❌ Error: Debes introducir exactamente 3 valores (X, Y y Z).")
                
    # Capturamos Ctrl+C para un cierre limpio si el usuario interrumpe el programa
    except KeyboardInterrupt:
        print("\nInterrupción por teclado detectada. Cerrando...")
        
    finally:
        # Se ejecuta siempre, haya error o el usuario pulse 'q'
        if arduino:
            arduino.close()
            print("Conexión serial cerrada.")

if __name__ == "__main__":
    main()

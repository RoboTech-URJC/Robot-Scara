"""
Developers:
            Adrián Manzanares ->  Github: Amanza17

"""

import serial
import time

#ajustar puerto
arduino = serial.Serial("/dev/ttyACM0",115200, timeout=1)
time.sleep(2)  # Espera a que Arduino se inicialice

*+
with open("posiciones.txt", "r") as f:
    for linea in f:
        linea = linea.strip()  # quitar salto de línea
        if linea:
            print(f"Enviando: {linea}")
            arduino.write((linea + '\n').encode())
            time.sleep(2) 

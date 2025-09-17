"""
Developers:
            Adrián Manzanares ->  Github: Amanza17

"""

import tkinter as tk
from math import *
import serial
import time

#poner el puerto correcto
arduino = serial.Serial("/dev/ttyACM1",115200, timeout=1)
time.sleep(2)  # Espera a que Arduino se inicialice


#para el calculo trigonometrico, b es el brazo externo, c el brazo interno y a la distancia del centro al punto

# Variables globales experimentales
pos_x = None
pos_y = None
ROT_ANGULO_BASE = 1000 / 90 #experimental
ROT_ANGULO_MANO = 1640 / 180 #experimental

LONG_B = 175
LONG_C = 225

def canvas_click(event):
    global pos_x, pos_y
    pos_x = event.x
    pos_y = event.y

    canvas.delete("marcador")
    canvas.create_oval(pos_x-5, pos_y-5, pos_x+5, pos_y+5, fill="red", tags="marcador")


def calcular_valores(pos_x, pos_y):
    ang_base = 90 - degrees(atan(pos_x / (400 - pos_y))) #angulo entre la base y el punto

    r_y = 400 - pos_y

    long_a = sqrt(pos_x **2 + r_y **2) #teorema de pitagoras

    print(f"long_a:{long_a}, long_b{LONG_B}, LONG_C = {LONG_C}")

    ang_a = degrees(acos((LONG_B**2 + LONG_C**2 - long_a**2) / (2 * LONG_B * LONG_C))) #tma. coseno

    ang_b = degrees(acos((long_a**2 + LONG_C**2 - LONG_B**2) / (2 * long_a * LONG_C))) #tma coseno

    val_base = (ang_base + ang_b ) * ROT_ANGULO_BASE

    val_mano =  (ang_a - 180) * ROT_ANGULO_MANO

    return(val_base, val_mano)


def enviar_valores():
    if pos_x is None or pos_y is None:
        print("Selecciona primero un punto en el canvas")
    else:
        valor_slider = slider.get()
        print(f"Punto seleccionado: X={pos_x}, Y={pos_y}, Slider={valor_slider}")

        v_base, v_mano = calcular_valores(pos_x, pos_y)
        print(f"Valores scara: Base={v_base}, Mano={v_base}, Z={valor_slider}")

        comando = f"{v_base} {v_mano} {valor_slider}\n"
        arduino.write(comando.encode('utf-8'))

        time.sleep(2)



#el tk lo hizo la mayoria chatti

# Crear ventana principal
root = tk.Tk()
root.title("Canvas clicable con slider")

# Canvas
canvas = tk.Canvas(root, width=400, height=400, bg="white")
# Dibujar un cuarto de circunferencia (zona de alcance, centrada en (0, 400), radio 400)
canvas.create_arc(
    -400, 0,   # x0, y0 del rectángulo delimitador (centro - radio)
    400, 800,  # x1, y1 del rectángulo delimitador (centro + radio)
    start=0,   # ángulo inicial (0 grados = eje x positivo)
    extent=90, # 90 grados hacia arriba (sentido antihorario)
    style='arc',
    outline='blue',
    width=2
)

# Dibujar un cuadrado cuya esquina superior derecha toque el arco
# Punto en el arco a 45 grados: x = 400*cos(45°), y = 400 + 400*sin(45°)
theta = radians(45)
x_arco = 400 * cos(theta)  # ~282.84
y_arco = 400 + 400 * sin(theta)  # ~682.84
# Cuadrado de lado 100 (ajustable), con esquina superior derecha en (x_arco, y_arco - 400)
lado_cuadrado = 282
canvas.create_rectangle(
    282, 118, 
    0,400,  # esquina inferior derecha
    outline='green',
    width=2
)

canvas.pack(side=tk.LEFT, padx=20, pady=20)
canvas.bind("<Button-1>", canvas_click)  # Detectar clic izquierdo

# Panel derecho con slider y botón
panel = tk.Frame(root)
panel.pack(side=tk.RIGHT, padx=20, pady=20)

slider = tk.Scale(panel, from_=-20000, to=0, orient=tk.VERTICAL, length=300)
slider.pack(pady=20)

boton_enviar = tk.Button(panel, text="Enviar", command=enviar_valores)
boton_enviar.pack(pady=20)

root.mainloop()




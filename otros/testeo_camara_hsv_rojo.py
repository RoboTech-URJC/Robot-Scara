import cv2
import numpy as np

CAM = 2
NUM_PUNTOS = 10

def capturar_puntos(nombre_ventana, mensaje):
    puntos = []
    valores_hsv = []

    cap = cv2.VideoCapture(CAM)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        exit()

    print(mensaje)
    print("Pulsa Q para salir.")

    while len(puntos) < NUM_PUNTOS:
        ret, frame = cap.read()

        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pixel_hsv = hsv[y, x]

                puntos.append((x, y))
                valores_hsv.append(pixel_hsv)

                print(
                    f"Punto {len(puntos)}: "
                    f"H={pixel_hsv[0]}, "
                    f"S={pixel_hsv[1]}, "
                    f"V={pixel_hsv[2]}"
                )

        cv2.namedWindow(nombre_ventana)
        cv2.setMouseCallback(nombre_ventana, mouse_callback)

        for x, y in puntos:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow(nombre_ventana, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return np.array(valores_hsv)


print("========================================")
print("       CALIBRACIÓN HSV DEL TABLERO")
print("========================================")

valores_rojo = capturar_puntos(
    "Calibracion ROJO",
    f"Haz {NUM_PUNTOS} clics sobre zonas ROJAS del tablero."
)

if len(valores_rojo) == 0:
    print("No se seleccionó ningún punto rojo.")
    exit()

h_rojo = valores_rojo[:, 0]
s_rojo = valores_rojo[:, 1]
v_rojo = valores_rojo[:, 2]

rojos_bajos = valores_rojo[h_rojo <= 90]
rojos_altos = valores_rojo[h_rojo > 90]

print("\n========================================")
print("              RESULTADO ROJO")
print("========================================")

if len(rojos_bajos) > 0:
    h1_min = np.min(rojos_bajos[:, 0])
    h1_max = np.max(rojos_bajos[:, 0])
    s1_min = np.min(rojos_bajos[:, 1])
    s1_max = np.max(rojos_bajos[:, 1])
    v1_min = np.min(rojos_bajos[:, 2])
    v1_max = np.max(rojos_bajos[:, 2])

    print("\nRango rojo cercano a H=0:")
    print(f"H: {h1_min} - {h1_max}")
    print(f"S: {s1_min} - {s1_max}")
    print(f"V: {v1_min} - {v1_max}")

else:
    print("No se encontraron muestras rojas cercanas a H=0.")
    h1_min = 0
    h1_max = 15
    s1_min = np.min(s_rojo)
    s1_max = np.max(s_rojo)
    v1_min = np.min(v_rojo)
    v1_max = np.max(v_rojo)

if len(rojos_altos) > 0:
    h2_min = np.min(rojos_altos[:, 0])
    h2_max = np.max(rojos_altos[:, 0])
    s2_min = np.min(rojos_altos[:, 1])
    s2_max = np.max(rojos_altos[:, 1])
    v2_min = np.min(rojos_altos[:, 2])
    v2_max = np.max(rojos_altos[:, 2])

    print("\nRango rojo cercano a H=180:")
    print(f"H: {h2_min} - {h2_max}")
    print(f"S: {s2_min} - {s2_max}")
    print(f"V: {v2_min} - {v2_max}")

else:
    print("No se encontraron muestras rojas cercanas a H=180.")
    h2_min = 165
    h2_max = 180
    s2_min = np.min(s_rojo)
    s2_max = np.max(s_rojo)
    v2_min = np.min(v_rojo)
    v2_max = np.max(v_rojo)

print("\nCódigo para ROJO:")

print(
    f"RED_HSV_LOW_1 = np.array([{h1_min}, {s1_min}, {v1_min}])"
)
print(
    f"RED_HSV_HIGH_1 = np.array([{h1_max}, {s1_max}, {v1_max}])"
)
print(
    f"RED_HSV_LOW_2 = np.array([{h2_min}, {s2_min}, {v2_min}])"
)
print(
    f"RED_HSV_HIGH_2 = np.array([{h2_max}, {s2_max}, {v2_max}])"
)


print("\n========================================")
print("       AHORA CALIBRAREMOS EL AZUL")
print("========================================")

valores_azul = capturar_puntos(
    "Calibracion AZUL",
    f"Haz {NUM_PUNTOS} clics sobre zonas AZULES de las fichas."
)

if len(valores_azul) == 0:
    print("No se seleccionó ningún punto azul.")
    exit()

h_azul = valores_azul[:, 0]
s_azul = valores_azul[:, 1]
v_azul = valores_azul[:, 2]

h_min = np.min(h_azul)
h_max = np.max(h_azul)

s_min = np.min(s_azul)
s_max = np.max(s_azul)

v_min = np.min(v_azul)
v_max = np.max(v_azul)

print("\n========================================")
print("              RESULTADO AZUL")
print("========================================")

print(f"H: {h_min} - {h_max}")
print(f"S: {s_min} - {s_max}")
print(f"V: {v_min} - {v_max}")

print("\nCódigo para AZUL:")

print(
    f"BLUE_HSV_LOW = np.array([{h_min}, {s_min}, {v_min}])"
)

print(
    f"BLUE_HSV_HIGH = np.array([{h_max}, {s_max}, {v_max}])"
)


print("\n========================================")
print("          CONFIGURACIÓN COMPLETA")
print("========================================")

print(
    f"RED_HSV_LOW_1 = np.array([{h1_min}, {s1_min}, {v1_min}])"
)
print(
    f"RED_HSV_HIGH_1 = np.array([{h1_max}, {s1_max}, {v1_max}])"
)
print(
    f"RED_HSV_LOW_2 = np.array([{h2_min}, {s2_min}, {v2_min}])"
)
print(
    f"RED_HSV_HIGH_2 = np.array([{h2_max}, {s2_max}, {v2_max}])"
)

print()

print(
    f"BLUE_HSV_LOW = np.array([{h_min}, {s_min}, {v_min}])"
)
print(
    f"BLUE_HSV_HIGH = np.array([{h_max}, {s_max}, {v_max}])"
)
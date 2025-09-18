import cv2
import numpy as np

cap = cv2.VideoCapture(2)  # seleccion de camara, a futuro se puede hacer automática

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo acceder a la cámara")
        break # un error, para el program cambiar en el futuro

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango rojo para cuadrícula
    rojo_bajo1 = np.array([0, 120, 70])
    rojo_alto1 = np.array([10, 255, 255])
    rojo_bajo2 = np.array([170, 120, 70])
    rojo_alto2 = np.array([180, 255, 255])
    mask_rojo = cv2.inRange(hsv, rojo_bajo1, rojo_alto1) | cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mask_rojo = cv2.morphologyEx(mask_rojo, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)

    # Contornos de la cuadrícula
    contours, _ = cv2.findContours(mask_rojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        cell_w = w // 3
        cell_h = h // 3

        # Rango verde
        verde_bajo = np.array([40, 50, 50])
        verde_alto = np.array([90, 255, 255])
        mask_verde = cv2.inRange(hsv, verde_bajo, verde_alto)

        # Rango azul
        azul_bajo = np.array([100, 150, 0])
        azul_alto = np.array([140, 255, 255])
        mask_azul = cv2.inRange(hsv, azul_bajo, azul_alto)

        # Dibujar y escribir color en cada celda
        for i in range(3):
            for j in range(3):
                cx = x + cell_w//2 + j*cell_w
                cy = y + cell_h//2 + i*cell_h

                # Tomar una pequeña ventana alrededor del centro
                s = 5  # tamaño del vecindario
                x1, x2 = max(cx-s,0), min(cx+s, frame.shape[1]-1)
                y1, y2 = max(cy-s,0), min(cy+s, frame.shape[0]-1)

                roi_verde = mask_verde[y1:y2, x1:x2]
                roi_azul = mask_azul[y1:y2, x1:x2]

                # Decidir color
                if np.any(roi_verde):
                    color = "Verde"
                    color_bgr = (0,255,0)
                elif np.any(roi_azul):
                    color = "Azul"
                    color_bgr = (255,0,0)
                else:
                    color = "Nada"
                    color_bgr = (0,140,255)

                # Dibujar punto y texto
                cv2.circle(frame, (cx, cy), 5, color_bgr, -1)
                cv2.putText(frame, color, (cx-30, cy-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color_bgr, 2, cv2.LINE_AA)

    cv2.imshow("Original con centros de celdas", frame)
    cv2.imshow("solo rojo", cv2.bitwise_and(frame, frame, mask=mask_rojo))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#include <AccelStepper.h>

// Motores
///AccelStepper motorBase(1, 2, 5);     // (modo, STEP, DIR) - Motor base
//AccelStepper motorMano(1, 3, 6);     // Motor mano
//AccelStepper motorZ(1, 4, 7);    // Motor brazo (nuevo)

AccelStepper motor(AccelStepper::DRIVER, 4, 7); // (modo, STEP, DIR)

void setup() {
  Serial.begin(115200);

  // Configurar velocidad y aceleración
  motor.setMaxSpeed(1000);       // Velocidad máxima (pasos por segundo)
  motor.setAcceleration(500);    // Aceleración (pasos por segundo^2)

  // Posición inicial
  motor.setCurrentPosition(0);

  // Primera meta: 1000 pasos hacia adelante
  motor.moveTo(1000);
}

void loop() {
  // Ejecuta el movimiento poco a poco
  if (motor.distanceToGo() == 0) {
    // Cuando llegue al destino, invertir dirección
    motor.moveTo(-motor.currentPosition());
  }

  motor.run(); // Importante: esto mueve el motor poco a poco
}

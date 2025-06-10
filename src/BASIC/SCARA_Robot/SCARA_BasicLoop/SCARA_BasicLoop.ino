#include <AccelStepper.h>

// Motores
AccelStepper motorBase(1, 2, 5);     // (modo, STEP, DIR) - Motor base
AccelStepper motorMano(1, 3, 6);     // Motor mano
AccelStepper motorZ(1, 4, 7);    // Motor brazo (nuevo)

int posicion = 0;
bool primerPaso = true;
bool segundoPaso = false;
bool tercerPaso = false;

void setup() {
  Serial.begin(115200);
  
  // Enable
  pinMode(8, OUTPUT);
  digitalWrite(8, LOW); // Activar drivers

  // Velocidad y aceleración
  motorBase.setMaxSpeed(4000);
  motorBase.setAcceleration(2000);

  motorMano.setMaxSpeed(4000);
  motorMano.setAcceleration(2000);

  motorZ.setMaxSpeed(4000);
  motorZ.setAcceleration(2000);

  // Posiciones iniciales
  motorBase.setCurrentPosition(0);
  motorMano.setCurrentPosition(0);
  motorZ.setCurrentPosition(0);

  // Movimiento inicial
  motorBase.moveTo(800);
  motorMano.moveTo(800);
  motorZ.moveTo(800);

  delay(500); // pequeña pausa
}

void loop() {
  bool goalBase = motorBase.run();
  bool goalMano = motorMano.run();
  bool goalZ = motorZ.run();

  // Movimiento del motor base
  if (!goalBase && primerPaso && !segundoPaso) {
    primerPaso = false;
    motorBase.setCurrentPosition(800);
    motorBase.moveTo(10);

  } else if (!goalBase && !primerPaso && !segundoPaso) {
    motorBase.stop();
    motorBase.setCurrentPosition(10);
    motorBase.moveTo(800);

    segundoPaso = true;
  }

  // Movimiento del motor mano
  else if (!goalMano && !primerPaso && segundoPaso && !tercerPaso) {
    motorMano.setCurrentPosition(800);
    motorMano.moveTo(10);
    segundoPaso = false;
    

  } else if (!goalMano && !primerPaso && segundoPaso && tercerPaso) {
    motorMano.stop();
    motorMano.setCurrentPosition(10);
    motorMano.moveTo(800);
    tercerPaso = true;
  }
//De aqui a abajo mal!
  // Movimiento del motor brazo (nuevo)
  else if (!goalZ && !primerPaso && segundoPaso && tercerPaso) {
    motorZ.setCurrentPosition(800);
    motorZ.moveTo(10);
    tercerPaso = false;

  } else if (!goalZ && !primerPaso && segundoPaso && !goalMano) {
    motorZ.stop();
    motorZ.setCurrentPosition(10);
    motorZ.moveTo(2000);
    primerPaso =
  }
}

#include <AccelStepper.h>

// Solo definimos el motor de la base
AccelStepper motorBase(1, 2, 5);      // (modo, STEP, DIR) - X
int posicion = 0;
bool primerPaso = true;
void setup() {
  // Configurar el pin de enable
  pinMode(8, OUTPUT);
  digitalWrite(8, LOW); // Habilitar los motores
  
  // Configurar velocidad y aceleración muy bajas para movimiento suave
  motorBase.setMaxSpeed(1000);       // Velocidad muy baja
  motorBase.setAcceleration(150);    // Aceleración muy suave
  
  // Establecer posición actual como 0
  motorBase.setCurrentPosition(0);
  
  // Definir el movimiento que queremos hacer
  motorBase.moveTo(800);
  delay(500);
  
           // Solo un cuarto de vuelta para empezar
}

void loop() {
  // Simplemente ejecutar el movimiento del motor base
  bool goalnotReached = motorBase.run();
  if(!goalnotReached && primerPaso) { //Ya ha llegado a 800 por primera vez
    primerPaso = false;
    motorBase.setCurrentPosition(800);
    motorBase.moveTo(10);
  }else if(!goalnotReached && !primerPaso) { // Ya ha llegado a 0
    motorBase.stop();
    motorBase.setCurrentPosition(10);
    motorBase.moveTo(800);
    primerPaso = true;
  }

} 
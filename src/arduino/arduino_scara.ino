#include <AccelStepper.h>

// Definición de motores
AccelStepper motorBase(AccelStepper::DRIVER, 2, 5);
AccelStepper motorMano(AccelStepper::DRIVER, 3, 6);
AccelStepper motorZ(AccelStepper::DRIVER, 4, 7);

const int ENABLE_PIN = 8;

long numeroBase = 0;
long numeroMano = 0;
long numeroZ = 0;

void electroimanON() {
  // Ponemos DIR en HIGH y damos 1 paso a la bobina
  digitalWrite(dirPinA, HIGH);
  digitalWrite(stepPinA, HIGH);
  delayMicroseconds(50);
  digitalWrite(stepPinA, LOW);
  Serial.println("[ESTADO] -> ENCENDIDO");
}

void electroimanOFF() {
  // Damos 2 micropasos para desplazar el chopper a la fase contraria / corte
  digitalWrite(dirPinA, LOW);
  for(int i = 0; i < 2; i++) {
    digitalWrite(stepPinA, HIGH);
    delayMicroseconds(50);
    digitalWrite(stepPinA, LOW);
    delayMicroseconds(50);
  }
  Serial.println("[ESTADO] -> APAGADO");
}

void setup() {
  Serial.begin(115200);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW); // Habilitar drivers

  // Configuración del electroimán / pines adicionales
  pinMode(stepPinA, OUTPUT);
  pinMode(dirPinA, OUTPUT);
  electroimanOFF(); // Asegura que el electroimán empiece apagado

  motorBase.setMaxSpeed(1000); motorBase.setAcceleration(500);
  motorMano.setMaxSpeed(1000); motorMano.setAcceleration(500);
  motorZ.setMaxSpeed(1000); motorZ.setAcceleration(500);

  Serial.println("Arduino listo para recibir posiciones por Serial.");
  Serial.println("Formato: Base Mano Z separados por espacio, o pausa: X tiempo X, ejemplo: X 1000 X");
}

void loop() {
    motorBase.run();
  motorMano.run();
  motorZ.run();
  if (motorBase.distanceToGo() == 0 && motorMano.distanceToGo() == 0 && motorZ.distanceToGo() == 0){
  if (Serial.available() > 0) {
    String linea = Serial.readStringUntil('\n');
    linea.trim();

    // Revisar si es comando de pausa: empieza y termina con X
    if (linea.startsWith("X") && linea.endsWith("X")) {
      // Extraer el número en medio
      int primerEspacio = linea.indexOf(' ');
      int segundoEspacio = linea.lastIndexOf(' ');
      if (primerEspacio >= 0 && segundoEspacio > primerEspacio) {
        long tiempo = linea.substring(primerEspacio + 1, segundoEspacio).toInt();
        Serial.print("Pausando ");
        Serial.print(tiempo);
        Serial.println(" ms");
        delay(tiempo); // Pausa
      }
    } else if (linea.startsWith("Y") && linea.endsWith("Y")) {
      // Encendido del electroimán
      electroimanON();

    } else if (linea.startsWith("Z") && linea.endsWith("Z")) {
      // Apagado del electroimán
      electroimanOFF();
      
    } else {
      // Comando de movimiento
      int primerEspacio = linea.indexOf(' ');
      int segundoEspacio = linea.lastIndexOf(' ');

      if (primerEspacio > 0 && segundoEspacio > primerEspacio) {
        numeroBase = linea.substring(0, primerEspacio).toInt();
        numeroMano = linea.substring(primerEspacio + 1, segundoEspacio).toInt();
        numeroZ = linea.substring(segundoEspacio + 1).toInt();

        Serial.print("Moviendo motores a: ");
        Serial.print(numeroBase); Serial.print(", ");
        Serial.print(numeroMano); Serial.print(", ");
        Serial.println(numeroZ);

        motorBase.moveTo(numeroBase);
        motorMano.moveTo(numeroMano);
        motorZ.moveTo(numeroZ);
      }
    }
  }


  }
}

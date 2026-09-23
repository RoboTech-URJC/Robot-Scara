#include <AccelStepper.h>

// Definición de motores
AccelStepper motorBase(AccelStepper::DRIVER, 2, 5);
AccelStepper motorMano(AccelStepper::DRIVER, 3, 6);
AccelStepper motorZ(AccelStepper::DRIVER, 4, 7);

const int ENABLE_PIN = 8;
const int MAGNET_PIN = A0;
long numeroBase = 0;
long numeroMano = 0;
long numeroZ = 0;

void electroimanON() {
  digitalWrite(MAGNET_PIN, HIGH);
  Serial.println("[ESTADO] -> ENCENDIDO");
}

void electroimanOFF() {
  digitalWrite(MAGNET_PIN, LOW);
  Serial.println("[ESTADO] -> APAGADO");
}

void setup() {
  Serial.begin(115200);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW); // Habilitar drivers

  // Configuración del electroimán
  pinMode(MAGNET_PIN, OUTPUT);
  electroimanOFF(); // Asegura que el electroimán empiece apagado

  motorBase.setMaxSpeed(2000); motorBase.setAcceleration(1000);
  motorMano.setMaxSpeed(2000); motorMano.setAcceleration(1000);
  motorZ.setMaxSpeed(1200); motorZ.setAcceleration(1000);

  Serial.println("Arduino listo para recibir posiciones por Serial.");
  Serial.println("Formato: Base Mano Z separados por espacio, o pausa: X tiempo X, ejemplo: X 1000 X");
}

bool isMoving = false; //Flag para saber si el robot se está moviendo

void loop() {
  motorBase.run();
  motorMano.run();
  motorZ.run();

  //Comprobar si el robot estaba moviéndose y acaba de llegar a su destino
  if (isMoving && motorBase.distanceToGo() == 0 && motorMano.distanceToGo() == 0 && motorZ.distanceToGo() == 0) {
    isMoving = false;       // Ya ha llegado
    Serial.println("DONE"); // Enviar confirmación al ordenador
  }

  // Solo leemos el puerto Serial si NO nos estamos moviendo
  if (!isMoving && Serial.available() > 0) {
    String linea = Serial.readStringUntil('\n');
    linea.trim();
    if (linea.length() == 0) return; // Ignorar líneas vacías

    // Revisar si es comando de pausa
    if (linea.startsWith("X") && linea.endsWith("X")) {
      int primerEspacio = linea.indexOf(' ');
      int segundoEspacio = linea.lastIndexOf(' ');
      if (primerEspacio >= 0 && segundoEspacio > primerEspacio) {
        long tiempo = linea.substring(primerEspacio + 1, segundoEspacio).toInt();
        Serial.print("Pausando "); Serial.print(tiempo); Serial.println(" ms");
        delay(tiempo); 
        Serial.println("DONE"); // Confirmar que la pausa terminó
      }
    } 
    // Comando encender imán
    else if (linea.startsWith("Y") && linea.endsWith("Y")) {
      electroimanON();
      Serial.println("DONE"); // Confirmar encendido
    } 
    // Comando apagar imán
    else if (linea.startsWith("Z") && linea.endsWith("Z")) {
      electroimanOFF();
      Serial.println("DONE"); // Confirmar apagado
    } 
    // Comando de movimiento de motores
    else {
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

        isMoving = true; // Activar flag: Empezamos a movernos
      }
    }
  }
}
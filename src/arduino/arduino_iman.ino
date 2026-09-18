const int stepPinA = 12; // D12
const int dirPinA  = 13; // D13

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
  pinMode(stepPinA, OUTPUT);
  pinMode(dirPinA, OUTPUT);
  
  electroimanOFF();
}

void loop() {
  electroimanON();
  delay(3000);

  electroimanOFF();
  delay(3000);
}
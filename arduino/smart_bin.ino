#include <Servo.h>

Servo myServo;

void setup() {
  myServo.attach(9);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char input = Serial.read();
    if (input == '1') {
      myServo.write(90);  // Abre
      delay(1000);
      myServo.write(0);   // Cierra
    }
  }
}

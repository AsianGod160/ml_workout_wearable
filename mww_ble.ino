#include <ArduinoBLE.h>

BLEService fitnessService("181C");

// Characteristics
BLECharacteristic heartRateChar("2AB4", BLERead | BLENotify, 50);    // Heart rate
BLECharacteristic workoutChar("2AC8", BLERead | BLENotify, 50);      // Workout + reps
BLECharacteristic commandChar("2A3A", BLEWrite, 50);  // Command

bool workoutEnded = false;
unsigned long lastHeartRateUpdate = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!BLE.begin()) {
    Serial.println("BLE failed to initialize");
    while (1);
  }

  BLE.setLocalName("MyBLEWearable");

  // Add characteristics to service
  fitnessService.addCharacteristic(heartRateChar);
  fitnessService.addCharacteristic(workoutChar);
  fitnessService.addCharacteristic(commandChar);
  BLE.addService(fitnessService);

  // Handle command input
  commandChar.setEventHandler(BLEWritten, onCommandReceived);

  BLE.advertise();
  Serial.println("BLE device is now advertising...");
}

void loop() {
  BLE.poll();

  if (!workoutEnded && millis() - lastHeartRateUpdate > 1000) {
    sendHeartRate();
    lastHeartRateUpdate = millis();
  }
}

void sendHeartRate() {
  int heartRate = random(70, 120);  // Simulated heart rate
  String data = "{\"heartRate\":" + String(heartRate) + "}";
  heartRateChar.setValue(data.c_str());
//  heartRateChar.notify();
  Serial.println("Sent heart rate: " + data);
}

void onCommandReceived(BLEDevice central, BLECharacteristic characteristic) {
  char cmd_buf[256];
  const char* cmd = reinterpret_cast<const char*>(characteristic.value()); //not null terminated for some reason
  size_t copy_len = min(characteristic.valueLength(), sizeof(cmd_buf) - 1); //length will usually be val len
  memcpy(cmd_buf, cmd, copy_len);
  cmd_buf[copy_len] = '\0';
  
  const char *json_cmp_to = "{\\\"command\\\": \\\"stop\\\"}";
  size_t json_len = strlen(json_cmp_to);
  
  if (copy_len == json_len && !memcmp(cmd_buf, json_cmp_to, copy_len)) {
    Serial.println(cmd_buf);
    workoutEnded = true;
    sendWorkoutSummary();
  }
}

void sendWorkoutSummary() {
  String summary = "{\"exercise\":\"squat\",\"reps\":14}";
  workoutChar.setValue(summary.c_str());
//  workoutChar.notify();
  Serial.println("Sent workout summary: " + summary);
}
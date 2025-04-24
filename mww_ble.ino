//#include <Arduino.h>
//#include <Wire.h>
//#include "Arduino_BMI270_BMM150.h"
//
//void setup() {
//  Serial.begin(9600);
//  while (!Serial);
//  Serial.println("Started");
//
//  if (!IMU.begin()) {
//    Serial.println("Failed to initialize IMU!");
//    while (1);
//  }
//
//  Serial.println("Accel and Gyro Data (CSV Format)");
//  //For reference, but not needed because script captures serial and brings these rows in
////  Serial.println("Time,Accel_X,Accel_Y,Accel_Z,Gyro_X,Gyro_Y,Gyro_Z,dt");
//}
// 
//unsigned long currentTime = 0;
//
//void loop() {
//  float ax, ay, az;  // Accelerometer values
//  float gx, gy, gz;  // Gyroscope values
//
//  // Read accelerometer data
//  if (IMU.accelerationAvailable()) {
//    IMU.readAcceleration(ax, ay, az);
//  }
//
//  // Read gyroscope data
//  if (IMU.gyroscopeAvailable()) {
//    IMU.readGyroscope(gx, gy, gz);
//  }
//
//  uint32_t dt = millis()-currentTime;
//  currentTime = millis(); // Get current time in milliseconds
//  
//  // Print CSV format data to serial monitor
//  Serial.print(currentTime); // Print timestamp
//  Serial.print(",");
//  Serial.print(ax, 6); // Printing with 6 decimal places for precision
//  Serial.print(",");
//  Serial.print(ay, 6);
//  Serial.print(",");
//  Serial.print(az, 6);
//  Serial.print(",");
//  Serial.print(gx, 6);
//  Serial.print(",");
//  Serial.print(gy, 6);
//  Serial.print(",");
//  Serial.print(gz, 6);
//  Serial.print(",");
//  Serial.println(dt);
//
//  delay(80); // Adjust delay as needed (in milliseconds)
//}
#include <Arduino.h>
#include <Wire.h>
#include "Arduino_BMI270_BMM150.h"
#include <mbed.h>              // Needed for Ticker on Nano 33 BLE
#include <ArduinoBLE.h>

using namespace mbed;

BLEService fitnessService("181C");

// Characteristics
BLECharacteristic heartRateChar("2AB4", BLERead | BLENotify, 50);    // Heart rate
BLECharacteristic workoutChar("2AC8", BLERead | BLENotify, 50);      // Workout + reps
BLECharacteristic commandChar("2A3A", BLEWrite, 50);  // Command



Ticker imuTicker;

#define SAMPLE_RATE_HZ 10
#define BUFFER_SIZE 10  // Number of samples to buffer

struct IMUData {
  float ax, ay, az;
  float gx, gy, gz;
  uint32_t dt;
};

//State Vars
volatile IMUData buffer[BUFFER_SIZE]; //data buf for imu
volatile uint8_t writeIndex = 0;
volatile uint8_t readIndex = 0;
unsigned long lastHeartRateUpdate = 0;
IMUData *current_data;


//Flags
volatile bool dataAvailable = false;
volatile bool imuSampleReady = false;
bool workoutEnded = false;


void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!BLE.begin()) {
    Serial.println("BLE failed to initialize");
    while (1);
  }
  
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }


  BLE.setLocalName("SeniorProject");
  
  fitnessService.addCharacteristic(heartRateChar);
  fitnessService.addCharacteristic(workoutChar);
  fitnessService.addCharacteristic(commandChar);
  BLE.addService(fitnessService);
  
  commandChar.setEventHandler(BLEWritten, onCommandReceived);

  BLE.advertise();

  //Change this to allow more frequent switching
  imuTicker.attach(HZ_10_callback, (float) 1/SAMPLE_RATE_HZ ); //100 ms = 10 times a second, immediately starts timer
}

void loop() {
  IMUData data;
  bool hasData = false;
  
  if (imuSampleReady){
    imuSampleReady = false;
    float ax, ay, az;
    float gx, gy, gz;

    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(ax, ay, az);
    }
    if (IMU.gyroscopeAvailable()) {
      IMU.readGyroscope(gx, gy, gz);
    }
    noInterrupts();

    //store retrieved data into buffer
    current_data = (IMUData*) &buffer[writeIndex];
    current_data->ax = ax;
    current_data->ay = ay;
    current_data->az = az;
    current_data->gx = gx;
    current_data->gy = gy;
    current_data->gz = gz;

    writeIndex = (writeIndex + 1) % BUFFER_SIZE;

    // Handle buffer overrun, not always necessary
    if (writeIndex == readIndex) {
      // Buffer overrun; drop the oldest sample
      readIndex = (readIndex + 1) % BUFFER_SIZE;
    }
    interrupts();
  }

  
  if (readIndex != writeIndex) {
    // deep copy
    data.ax = buffer[readIndex].ax;
    data.ay = buffer[readIndex].ay;
    data.az = buffer[readIndex].az;
    data.gx = buffer[readIndex].gx;
    data.gy = buffer[readIndex].gy;
    data.gz = buffer[readIndex].gz;
    data.dt = buffer[readIndex].dt;

    readIndex = (readIndex + 1) % BUFFER_SIZE;
    hasData = true;  // Flag that we grabbed a sample
  }
  
  if (hasData) {
    //print safely
    Serial.print(data.ax, 6);
    Serial.print(",");
    Serial.print(data.ay, 6);
    Serial.print(",");
    Serial.print(data.az, 6);
    Serial.print(",");
    Serial.print(data.gx, 6);
    Serial.print(",");
    Serial.print(data.gy, 6);
    Serial.print(",");
    Serial.println(data.gz, 6);
  }

  BLE.poll();
  if (!workoutEnded && millis() - lastHeartRateUpdate > 1000) {
    sendHeartRate();
    lastHeartRateUpdate = millis();
  }
  else if (workoutEnded && imuSampleReady){//reuse 10hz flag
    sendWorkoutSummary();
  }
}

// Timer ISR, triggered every 100 ms
void HZ_10_callback() {
  imuSampleReady = true;
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

  Serial.print("Commande Received: ");
  Serial.println(cmd_buf);
  const char *json_cmp_to = "{\"command\":\"stop\"}";
//  Commande Received: {"command":"stop"}
//  const char *json_cmp_to = "{\\\"command\\\": \\\"stop\\\"}";
  size_t json_len = strlen(json_cmp_to);
  if (copy_len == json_len && !memcmp(cmd_buf, json_cmp_to, copy_len)) {
    workoutEnded = true;
  }
}

void sendWorkoutSummary() {
  String summary = "{\"exercise\":\"squat\",\"reps\":14}";
  workoutChar.setValue(summary.c_str());
//  workoutChar.notify();
  Serial.println("Sent workout summary: " + summary);
}
#include <Arduino.h>
#include <Wire.h>
#include "Arduino_BMI270_BMM150.h"
#include <mbed.h>              // Needed for Ticker on Nano 33 BLE
#include <ArduinoBLE.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


// const tflite::Model* rep_count_model;
// tflite::MicroInterpreter* rep_counter_interpreter;
// TfLiteTensor* rc_input;
// TfLiteTensor* rc_output;

const tflite::Model *classification_model;
tflite::MicroInterpreter *classification_interpreter;
TfLiteTensor *cl_input;
TfLiteTensor *cl_output;

constexpr int kTensorArenaSize_rc = 2 * 1024;
uint8_t tensor_arena_rc[kTensorArenaSize_rc];

using namespace mbed;

BLEService fitnessService("181C");

// Characteristics
BLECharacteristic heartRateChar("2AB4", BLERead | BLENotify, 50);    // Heart rate
BLECharacteristic workoutChar("2AC8", BLERead | BLENotify, 50);      // Workout + reps
BLECharacteristic commandChar("2A3A", BLEWrite, 50);  // Command

Ticker imuTicker;
Ticker heartRateTicker;
Ticker bluetoothTicker;

MAX30105 heart_rate_sensor;

#define SAMPLE_RATE_HZ 10 //10 IMU Samples Per Second
#define HEART_RATE_HZ 30 //30 heart rate samples per second
#define RC_BUFFER_SIZE 128  // Number of samples to buffer
#define CL_BUFFER_SIZE 96 //flattened buffer size for interpreter input
#define CL_COLUMN_SIZE 32
#define CL_ROW_SIZE 3

#define RATE_SIZE 4 //heart rate buffer size, 4 is good

#define USE_MODEL
// #define USE_BLE

byte rates[RATE_SIZE]; //Array of heart rates
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred

float beatsPerMinute;
int beatAvg;

struct IMUData {
  float ax, ay, az;
  float gx, gy, gz;
  uint32_t dt;
};

//State Vars
volatile IMUData buffer[RC_BUFFER_SIZE]; //data buf for imu
volatile uint8_t writeIndex = 0;
volatile uint8_t readIndex = 0;


//Flags
volatile bool imuSampleReady = false;
volatile bool heartRateReady = false;
volatile bool bluetoothReady = false;
volatile bool workoutEnded = false;
volatile bool modelReady = false;

// float rc_model_data[RC_BUFFER_SIZE] = {
//   -1.        ,   2.32630728,   5.55996102,   8.61088854,
//   11.39410695,  13.83209034,  15.85692922,  17.41222218,
//   18.45464688,  18.95516683,  18.89984014,  18.29020794,
//   17.14325139,  15.4909187 ,  13.37923523,  10.86702143,
//    8.02425446,   4.93011893,   1.67080128,  -1.66291092,
//   -4.97815789,  -8.18259418, -11.18696094, -13.90757223,
//  -16.26864603, -18.20441517, -19.66095924, -20.59770655,
//  -20.9885642 , -20.82264494, -20.10457043, -18.85434245,
//  -17.10678583, -14.91057836, -12.3268949 ,  -9.42770335,
//   -6.29376005,  -3.01236027,   0.32509335,   3.62563679,
//    6.79733418,   9.75183859,  12.406853  ,  14.68842256,
//   16.53299469,  17.88918924,  18.71922972,  18.99999555,
//   18.72366606,  17.89793834,  16.54581286,  14.70495275,
//   12.42663475,   9.77432091,   6.8218908 ,   3.65158371,
//    0.35170782,  -2.98581959,  -6.26803245,  -9.40350546,
//  -12.30490074, -14.89140058, -17.09095863, -18.84230668,
//  -20.09666135, -20.81908286, -20.98944833, -20.60301227,
//  -19.67053877, -18.21800166, -16.28586103, -13.92793623,
//  -11.2099067 ,  -8.20748255,  -5.00429561,  -1.68956993,
//    1.64436356,   4.90463891,   8.00044188,  10.8455396 ,
//   13.3606825 ,  15.47581187,  17.13201125,  18.28314758,
//   18.89715623,  18.95693412,  18.46081616,  17.42262159,
//   15.87126909,  13.84997124,  11.41503081,   8.63427253,
//    5.58515378,   2.35260708,  -0.97332574,  -4.30000156,
//   -7.53475659, -10.58748746, -13.37316105, -15.81418305,
//  -17.84255937, -19.40179002, -20.448443  , -20.95336404,
//  -20.90248866, -20.29723399, -19.15445926, -17.5059962 ,
//  -15.39776238, -12.88848216, -10.04805098,  -6.9555884 ,
//   -3.69723426,  -0.36374928,   2.95201309,   6.15769303,
//    9.16399706,  11.88718527,  14.25140387,  16.19079808,
//   17.65134653,  18.59236597,  18.98764451,  18.82617177,
//   18.11244552,  16.86634646,  15.12258439,  12.9297314
// };

uint8_t cl_model_data[3][RC_BUFFER_SIZE] = {
      {144, 141, 134, 127, 123, 121, 117, 113, 110,  97, 101, 109, 119,
        131, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140,
        140, 140, 140, 140, 140, 140},
       {139, 137, 134, 131, 130, 129, 128, 127, 124, 114, 123, 134, 147,
        161, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173,
        173, 173, 173, 173, 173, 173},
       {210, 201, 189, 178, 171, 164, 164, 156, 151, 141, 137, 143, 145,
        147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
        147, 147, 147, 147, 147, 147}
};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  //Dynamically define model, save .bss space
  constexpr int kTensorArenaSize_cl = 64 * 1024; //64 KB
  uint8_t *tensor_arena_cl = (uint8_t*) malloc(kTensorArenaSize_cl);
  if(!tensor_arena_cl){
    Serial.println("Arena Alloc Failed");
  }

  while (!BLE.begin()) {
    Serial.println("BLE failed to initialize");
    while (1);
  }

  while (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  while (!heart_rate_sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Failed to initialize Heart Rate Sensor!");
    while (1);
  }

  //BLE Setup:==============================================
  BLE.setLocalName("MyBLEWearable");
  
  fitnessService.addCharacteristic(heartRateChar);
  fitnessService.addCharacteristic(workoutChar);
  fitnessService.addCharacteristic(commandChar);
  BLE.addService(fitnessService);
  
  commandChar.setEventHandler(BLEWritten, onCommandReceived);
  
  BLE.advertise();
  //=========================================================


  //Heart Rate Setup:==============================================
  heart_rate_sensor.setup(); //Configure sensor with default settings
  heart_rate_sensor.setPulseAmplitudeRed(0x0A); //Turn Red LED to low to indicate sensor is running
  heart_rate_sensor.setPulseAmplitudeGreen(0); //Turn off Green LED
  //===============================================================

  // rep_count_model = tflite::GetModel(rep_count_model_ptr);
  classification_model = tflite::GetModel(classification_model_ptr);
  // if (rep_count_model->version() != TFLITE_SCHEMA_VERSION) {
  //   Serial.println("Rep Count Model Schema Mismatch!");
  //   while (1);
  // }
  /*else*/ if (classification_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Classification Model Mismatch!");
    while(1);
  }

  static tflite::AllOpsResolver resolver;

  // static tflite::MicroInterpreter static_interpreter1(rep_count_model, resolver, tensor_arena_rc, kTensorArenaSize_rc);
  // rep_counter_interpreter = &static_interpreter1;
  static tflite::MicroInterpreter static_interpreter2(classification_model, resolver, tensor_arena_cl, kTensorArenaSize_cl);
  classification_interpreter = &static_interpreter2;

  // rep_counter_interpreter->AllocateTensors();
  // rc_input = rep_counter_interpreter->input(0);
  // rc_output = rep_counter_interpreter->output(0);

  classification_interpreter->AllocateTensors();
  cl_input = classification_interpreter->input(0);
  cl_output = classification_interpreter->output(0);

  Serial.println("Models ready.");

  //Change these to allow more frequent switching, timers for IMU and Heart Rate
  #ifdef USE_BLE //imuticker will then attach in the ble connected event instead
  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);
  #else 
  imuTicker.attach(HZ_10_callback, (float) 1/SAMPLE_RATE_HZ ); //100 ms = 10 times a second, immediately starts timer
  #endif
  bluetoothTicker.attach(HZ_1_callback, 1);
  heartRateTicker.attach(HZ_4_callback, (float) 1/HEART_RATE_HZ ); //250 ms = 4 times a second, immediately starts timer
}

void loop() {
  IMUData data;
  bool hasData = false;
  
  if (imuSampleReady){ //100 ms passed, collect data
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
    IMUData *current_data = (IMUData*) &buffer[writeIndex];
    current_data->ax = ax;
    current_data->ay = ay;
    current_data->az = az;
    current_data->gx = gx;
    current_data->gy = gy;
    current_data->gz = gz;

    writeIndex = (writeIndex + 1) % RC_BUFFER_SIZE;

    // Handle buffer overrun, not always necessary
    if (writeIndex == readIndex) {
      // Buffer overrun; drop the oldest sample
      readIndex = (readIndex + 1) % RC_BUFFER_SIZE;
    }
    interrupts();
  }

  #ifndef USE_MODEL //Using rep_count_model, do not read from buffer/print to save cycles and mem
  if (readIndex != writeIndex) {//store data into object
    // deep copy
    data.ax = buffer[readIndex].ax;
    data.ay = buffer[readIndex].ay;
    data.az = buffer[readIndex].az;
    data.gx = buffer[readIndex].gx;
    data.gy = buffer[readIndex].gy;
    data.gz = buffer[readIndex].gz;
    data.dt = buffer[readIndex].dt;

    readIndex = (readIndex + 1) % RC_BUFFER_SIZE;
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
  #endif
  

  BLE.poll();
  if (!workoutEnded && bluetoothReady) {
    bluetoothReady = false;
    sendHeartRate();
  }
  else if (workoutEnded && imuSampleReady && BLE.connected()){//reuse 10hz flag
    sendWorkoutSummary();
  }

  if (heartRateReady){
    heartRateReady = false;
    long irValue = heart_rate_sensor.getIR();
    if (checkForBeat(irValue) == true) {
      //We sensed a beat!
      long delta = millis() - lastBeat;
      lastBeat = millis();

      beatsPerMinute = 60 / (delta / 1000.0);

      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute; //Store this reading in the array
        rateSpot %= RATE_SIZE; //Wrap variable

        //Take average of readings
        beatAvg = 0;
        for (byte x = 0 ; x < RATE_SIZE ; x++)
          beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
      }
      // Serial.print("BPM: "); Serial.println(beatsPerMinute);
      // Serial.print("AVG BPM: "); Serial.println(beatAvg);
    }
  }

  #ifdef USE_MODEL
  if (modelReady){
    modelReady = false;
    TfLiteStatus status;
    
    // //Rep Count invocation============================================
    // for (int i = 0; i < RC_BUFFER_SIZE; i++){
    //   rc_input->data.f[i] = rc_model_data[i];
    // }
    // TfLiteStatus status = rep_counter_interpreter->Invoke();
    // if (status != kTfLiteOk) {
    //   Serial.println("Rep Count Invoke failed!");
    // }
    // float result = rc_output->data.f[0];
    // Serial.print("Rep Count Model output: ");
    // Serial.println(result);

    //Classification Model Invocation=================================
    for (int i = 0; i < CL_ROW_SIZE; ++i) {
      for (int j = 0; j < CL_COLUMN_SIZE; ++j) {
        int idx = i * 32 + j;
        cl_input->data.uint8[idx] = cl_model_data[i][j];
      }
    }
    status = classification_interpreter->Invoke();
    if (status != kTfLiteOk) {
      Serial.println("Classification Invoke Failed!");
    }
    float cl_result[3];
    Serial.print("Classification Model Output: ");


    //retrieve model output size:
    int num_elements = 1;
    for (int i = 0; i < cl_output->dims->size; ++i) {
      num_elements *= cl_output->dims->data[i];
    }

    float scale = cl_output->params.scale;
    int zero_point = cl_output->params.zero_point;
    // Read output
    for (int i = 0; i < num_elements; i++) {
      cl_result[i] = (cl_output->data.uint8[i] - zero_point) * scale;
      Serial.print(cl_result[i]);
      Serial.print("  ");
    }

    Serial.println();
  }
  #endif
}

// Timer ISR, triggered every 100 ms
void HZ_10_callback() {
  imuSampleReady = true;
}
void HZ_4_callback() {
  heartRateReady = true;
}
void HZ_1_callback() {
  bluetoothReady = true;
  modelReady = true;
}

void sendHeartRate() {
  // int heartRate = random(70, 120);  // Simulated heart rate
  String data = "{\"heartRate\":" + String(beatAvg) + "}";
  heartRateChar.setValue(data.c_str());
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
  size_t json_len = strlen(json_cmp_to);
  if (copy_len == json_len && !memcmp(cmd_buf, json_cmp_to, copy_len)) {
    workoutEnded = true;
  }
}

void onBLEConnected(BLEDevice central){
  Serial.println("BLE Device Connected... Starting IMU Collection");
  imuTicker.attach(HZ_10_callback, float(1) / SAMPLE_RATE_HZ);
}

void onBLEDisconnected(BLEDevice central){
  imuTicker.detach();
}

void sendWorkoutSummary() {
  String summary = "{\"exercise\":\"squat\",\"reps\":14}";
  workoutChar.setValue(summary.c_str());
  Serial.println("Sent workout summary: " + summary);
}


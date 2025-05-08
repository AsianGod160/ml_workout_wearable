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

const tflite::Model* rep_count_model;
tflite::MicroInterpreter* rep_counter_interpreter;
TfLiteTensor* rc_input;
TfLiteTensor* rc_output;

const tflite::Model *classification_model;
tflite::MicroInterpreter *classification_interpreter;
TfLiteTensor *cl_input;
TfLiteTensor *cl_output;

constexpr int kTensorArenaSize_rc = 10 * 1024;
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
#define RC_TENSOR_SIZE 128  // Number of samples to buffer
#define CL_BUFFER_SIZE 96 //flattened buffer size for interpreter input
#define CL_BURST_LEN 20
#define CL_COLUMN_SIZE (32 * CL_BURST_LEN)
#define CL_ROW_SIZE 3

#define RC_BURST_LEN 5
#define DATA_LEN 640

#define RATE_SIZE 4 //heart rate buffer size, 4 is good
#define CONVOLUTION_SIZE 10

//Axis
#define X 0
#define Y 1
#define Z 2
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
#ifdef USE_MODEL
volatile float a_x[RC_TENSOR_SIZE]; //x rc
volatile float a_y[RC_TENSOR_SIZE]; //y rc
volatile float a_z[RC_TENSOR_SIZE]; //z rc
volatile float *accelerometer_data[] = {
  a_x, a_y, a_z
};
volatile float cl_model_data[CL_ROW_SIZE][CL_COLUMN_SIZE]; //{{x_cl}, {y_cl}, {z_cl}
volatile float accel_buf_x[CONVOLUTION_SIZE]; //sliding window for models
volatile float accel_buf_y[CONVOLUTION_SIZE]; //sliding window for models
volatile float accel_buf_z[CONVOLUTION_SIZE]; //sliding window for models
volatile float *accel_bufs[] = {
  accel_buf_x, accel_buf_y, accel_buf_z
};
volatile uint8_t accel_write_index = 0;
volatile uint8_t RC_write_index = 0;
volatile uint8_t RC_read_index = 0;
volatile uint8_t CL_write_index = 0;
volatile uint8_t CL_read_index = 0;
volatile uint8_t RC_buf_count = 0;
volatile uint16_t CL_buf_count = 0;

//Model Preprocessing: Running sums (Used for Averages)==========================
volatile float rc_running_sum_x = 0;
volatile float rc_running_sum_y = 0;
volatile float rc_running_sum_z = 0;
volatile float *rc_running_sum[] = {&rc_running_sum_x, &rc_running_sum_y, &rc_running_sum_z};

volatile float cl_running_sum_x = 0;
volatile float cl_running_sum_y = 0;
volatile float cl_running_sum_z = 0;
volatile float *cl_running_sum[] = {&cl_running_sum_x, &cl_running_sum_y, &cl_running_sum_z};

//Model Preprocessing: Rolling Max (Used for Normalizations)======================
volatile float rc_rolling_max_x = 0;
volatile float rc_rolling_max_y = 0;
volatile float rc_rolling_max_z = 0;
volatile float *rc_rolling_max[] = {&rc_rolling_max_x, &rc_rolling_max_y, &rc_rolling_max_z};

volatile float cl_rolling_max_x = 0;
volatile float cl_rolling_max_y = 0;
volatile float cl_rolling_max_z = 0;
volatile float *cl_rolling_max[] = {&cl_rolling_max_x, &cl_rolling_max_y, &cl_rolling_max_z};

volatile uint32_t rc_count = 0;
volatile uint8_t cl_count = 0;

#else
volatile IMUData buffer[RC_TENSOR_SIZE]; //data buf for imu
volatile uint8_t readIndex = 0;
volatile uint8_t writeIndex = 0;
#endif


//Flags
volatile bool imuSampleReady = false;
volatile bool heartRateReady = false;
volatile bool bluetoothReady = false;
volatile bool workoutEnded = false;
volatile bool rc_modelReady = false;
volatile bool cl_modelReady = false;
volatile bool deviceConnected = false;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  //Dynamically define model, save .bss space
  constexpr int kTensorArenaSize_cl = 64 * 1024; //64 KB
  uint8_t *tensor_arena_cl = (uint8_t*) malloc(kTensorArenaSize_cl);
  if(!tensor_arena_cl){
    Serial.println("Arena Alloc Failed");
  }

  //BLE init
  while (!BLE.begin()) {
    Serial.println("BLE failed to initialize");
    while (1);
  }

  //imu init
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

  //TFlite Model Setup:============================================
  rep_count_model = tflite::GetModel(rep_count_model_ptr);
  classification_model = tflite::GetModel(classification_model_ptr);
  if (rep_count_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Rep Count Model Schema Mismatch!");
    while (1);
  }
  if (classification_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Classification Model Mismatch!");
    while(1);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter1(rep_count_model, resolver, tensor_arena_rc, kTensorArenaSize_rc);
  rep_counter_interpreter = &static_interpreter1;
  static tflite::MicroInterpreter static_interpreter2(classification_model, resolver, tensor_arena_cl, kTensorArenaSize_cl);
  classification_interpreter = &static_interpreter2;

  TfLiteStatus status = rep_counter_interpreter->AllocateTensors();
  rc_input = rep_counter_interpreter->input(0);
  rc_output = rep_counter_interpreter->output(0);
  if (status != kTfLiteOk) {
    Serial.println("❌ Rep Counter model failed to allocate tensors!");
  }

  status = classification_interpreter->AllocateTensors();
  cl_input = classification_interpreter->input(0);
  cl_output = classification_interpreter->output(0);
  if (status != kTfLiteOk) {
    Serial.println("❌ Classification model failed to allocate tensors!");
  }

  Serial.println("Models ready.");
  //===============================================================

  //Change these to allow more frequent switching, timers for IMU and Heart Rate
  #ifdef USE_BLE //imuticker will then attach in the ble connected event instead
  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);
  #else 
  imuTicker.attach(HZ_10_callback, (float) 1/SAMPLE_RATE_HZ ); //100 ms = 10 times a second, immediately starts timer
  #endif
  bluetoothTicker.attach(HZ_1_callback, 1);
  heartRateTicker.attach(HZ_30_callback, (float) 1/HEART_RATE_HZ ); //250 ms = 4 times a second, immediately starts timer
}

void loop() {
  IMUData data;
  bool hasData = false;
  
  if (imuSampleReady){ //100 ms passed, collect data
    imuSampleReady = false;
    float ax, ay, az;
    float gx, gy, gz;

    #ifdef USE_MODEL
    //copy data into rep count and classifier model buffers respectively (accel x, y, and z)
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(ax, ay, az);
      // Serial.print("Ra: ");
      // Serial.print(ax);
      // Serial.print(ay);
      // Serial.print(az);
      noInterrupts();

      //preprocessing, saturate convolution buffers to average later
      accel_bufs[0][accel_write_index] = ax;
      accel_bufs[1][accel_write_index] = ay;
      accel_bufs[2][accel_write_index] = az;
      rc_count++;

      //if convolution is saturated
      if (rc_count >= 10){ //accel buff full
        rc_count--; //keep it hovering
        float accel_smoothed[] = {0, 0, 0}; //used to store x, y, and z smoothed

        //find average / perform convolution
        for (int i = 0; i < CONVOLUTION_SIZE; i++){
          for (int j = 0; j < 3; j++){
            accel_smoothed[j] += accel_bufs[j][i]; //row based sum
          }
        }
        for(int axis = 0; axis < 3; axis++){
          accel_smoothed[axis] /= CONVOLUTION_SIZE; //div final sum by 10
        }


        float ax_smoothed = accel_smoothed[X];
        float ay_smoothed = accel_smoothed[Y];
        float az_smoothed = accel_smoothed[Z];
        //used for rc
        a_x[RC_write_index] = ax_smoothed;
        a_y[RC_write_index] = ay_smoothed;
        a_z[RC_write_index] = az_smoothed;
        
        for(int i = 0; i < 3; i++){ //Compute the maximum as values are added. Used to normalize later on
          if (abs(accel_smoothed[i]) > *rc_rolling_max[i]){
            *rc_rolling_max[i] = abs(accel_smoothed[i]);
          }
        }

        //increment input tensor buff idx
        RC_write_index = (RC_write_index + 1) % RC_TENSOR_SIZE;
        RC_buf_count++;

        //rc running sum, used in final average preprocessing
        rc_running_sum_x += ax_smoothed;
        rc_running_sum_y += ay_smoothed;
        rc_running_sum_z += az_smoothed;

        //used for cl
        cl_model_data[X][CL_write_index] = ax_smoothed; //x accel store in classification buffer
        cl_model_data[Y][CL_write_index] = ay_smoothed; //y accel store in classification buffer
        cl_model_data[Z][CL_write_index] = az_smoothed; //z accel store in classification buffer

        for(int i = 0; i < 3; i++){ //Compute the maximum as values are added. Used to normalize later on
          if (abs(accel_smoothed[i]) > *cl_rolling_max[i]){
            *cl_rolling_max[i] = abs(accel_smoothed[i]);
          }
        }
        CL_write_index = (CL_write_index + 1) % CL_COLUMN_SIZE;
        CL_buf_count++;

        cl_running_sum_x += ax_smoothed;
        cl_running_sum_y += ay_smoothed;
        cl_running_sum_z += az_smoothed;
      }

      interrupts();
    }
    //increment
    accel_write_index = (accel_write_index + 1) % CONVOLUTION_SIZE;

    //=========================================================
    #else //ifndef USE_MODEL, just store into IMUData buffer
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(ax, ay, az);
    }
    if (IMU.gyroscopeAvailable()) { //don't really need gyroscope
      IMU.readGyroscope(gx, gy, gz);
    }
    // store retrieved data into buffer
    noInterrupts();
    IMUData *current_data = (IMUData*) &buffer[writeIndex];
    current_data->ax = ax;
    current_data->ay = ay;
    current_data->az = az;
    current_data->gx = gx;
    current_data->gy = gy;
    current_data->gz = gz;

    writeIndex = (writeIndex + 1) % RC_TENSOR_SIZE;
    interrupts();
    #endif
  }

  //PRINT TO SERIAL FOR TESTING/DATA COLLECTION===========================================================
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
    readIndex = (readIndex + 1) % RC_TENSOR_SIZE;
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
  
  //BLE===============================================================================
  BLE.poll();
  if (!workoutEnded && bluetoothReady) {
    bluetoothReady = false;
    sendHeartRate();
  }
  else if (workoutEnded && imuSampleReady && BLE.connected()){//reuse 10hz flag
    sendWorkoutSummary();
  }

  //HEART RATE SLIDING WINDOW==========================================================
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
    }
  }
  // Serial.println(CL_buf_count);
  //Inference======================================================
  #ifdef USE_MODEL
  //Rep Count invocation============================================
  if(rc_modelReady){
    RC_buf_count = 0;
    rc_modelReady = false;
    TfLiteStatus status;
    for (int i = 0; i < 3; i++){ //3 axes
      for (int j = 0; j < RC_TENSOR_SIZE; j++){
        rc_input->data.f[j] = (accelerometer_data[i][j] - (*rc_running_sum[i]/RC_TENSOR_SIZE)) / abs(*rc_rolling_max[i]);
        // Serial.print("RC:");
        // Serial.println((accelerometer_data[i][j] - (*rc_running_sum[i]/RC_TENSOR_SIZE)) / abs(*rc_rolling_max[i]));
      }
      *rc_running_sum[i] = 0; //reset running sum
      *rc_rolling_max[i] = 0; //reset rolling max
      TfLiteStatus status = rep_counter_interpreter->Invoke();
      if (status != kTfLiteOk) {
        Serial.print("Rep Count Invoke failed!: ");
        Serial.println(i);
      }
      float result = rc_output->data.f[0];

      //Print in form "Rep Count Model Output(i): {out}"
      Serial.print("Rep Count Model output(");
      Serial.print(i);
      Serial.print("): ");
      Serial.println(result);
    }
  }
  //Classification Model Invocation=================================
  if (cl_modelReady){
    //reset states for next use
    CL_buf_count = 0;
    CL_write_index = 0; 
    cl_modelReady = false;

    //input stuff
    TfLiteStatus status;
    float scale = cl_input->params.scale;
    int zero_point = cl_input->params.zero_point;

    //retrieve model output size
    int num_elements = 1;
    for (int i = 0; i < cl_output->dims->size; ++i) {
      num_elements *= cl_output->dims->data[i];
    }

    float out_scale = cl_output->params.scale;
    int out_zero_point = cl_output->params.zero_point;

    //experimental:===============================
    for (int k = 0; k < CL_BURST_LEN; k++){
      for (int i = 0; i < CL_ROW_SIZE; ++i) { //x, y, or z
        for (int j = 0; j < CL_COLUMN_SIZE/CL_BURST_LEN; ++j) {
          uint8_t idx = i * (CL_COLUMN_SIZE/CL_BURST_LEN) + j;
          float val = (cl_model_data[i][j + k*(CL_COLUMN_SIZE/CL_BURST_LEN)] - (*cl_running_sum[i] / CL_COLUMN_SIZE)) / abs(*cl_rolling_max[i]); //subtract inputs by average
          // Serial.print("CL: ")
          // Serial.println((cl_model_data[i][j] - (*cl_running_sum[i] / CL_COLUMN_SIZE)) / abs(*cl_rolling_max[i])); //subtract inputs by average
          int quantized = (int)(val / scale + zero_point);

          //clamp
          quantized = constrain(quantized, 0, 255);

          cl_input->data.uint8[idx] = quantized;
        }
      }
      status = classification_interpreter->Invoke();
      if (status != kTfLiteOk) {
        Serial.println("Classification Invoke Failed!");
      }
      float cl_result[3];
      Serial.print("Classification Model Output: ");

      // Read output
      for (int i = 0; i < num_elements; i++) {
        cl_result[i] = (cl_output->data.uint8[i] - out_zero_point) * out_scale;
        Serial.print(cl_result[i]);
        Serial.print("  ");
      }
      Serial.println();
    }
    for (int i = 0; i < CL_ROW_SIZE; i++){
      *cl_running_sum[i] = 0; //reset running sum for this axis
      *cl_rolling_max[i] = 0; //reset rolling max
    }
    //experimental:===============================
  }
  #endif
}

// Timer ISR, triggered every 100 ms
void HZ_10_callback() {
  imuSampleReady = true;
}
void HZ_30_callback() {
  heartRateReady = true;
}
void HZ_1_callback() {
  #ifdef USE_BLE
  if (deviceConnected)
    bluetoothReady = true;
  #endif
  #ifdef USE_MODEL 
  //check if CL buffer is full
  if (CL_buf_count >= CL_COLUMN_SIZE){
    cl_modelReady = true;
  }
  if (RC_buf_count >= RC_TENSOR_SIZE){
    rc_modelReady = true;
  }
  #endif
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
  deviceConnected = true;
}

void onBLEDisconnected(BLEDevice central){
  imuTicker.detach();
  deviceConnected = false;
}

void sendWorkoutSummary() {
  String summary = "{\"exercise\":\"squat\",\"reps\":14}";
  workoutChar.setValue(summary.c_str());
  Serial.println("Sent workout summary: " + summary);
}

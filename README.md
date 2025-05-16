# ML Workout Wearable Firmware

This repository contains the Arduino firmware for a TinyML-powered workout wearable based on the Arduino Nano 33 BLE.  
It reads 9-axis IMU and heart-rate data, classifies the exercise (bench, squat, deadlift) with a TFLite model, counts repetitions with a second TFLite model, and streams live heart rate and workout summaries over BLE to a companion app.

---

## 📂 Repository Structure

/  
├──serial_logger.py # UART Capture Script  
├──README.md # (this file)  
├──imu_printer/imu_printer.ino # Main Sketch  
├──imu_printer/model.h # Header for embedded TFLite models  
├──imu_printer/model.c # Embedded TFLite model data  

---

## 🔧 Hardware

- **Board:** Arduino Nano 33 BLE  
- **IMU:** Bosch BMI270 + BMM150 (9-axis: accel, gyro, magnetometer)  
- **Heart-Rate Sensor:** MAX30102 (via SparkFun MAX30105 library)  
- **Connections:**  
  - IMU & MAX30102 → I²C (SDA, SCL), IMU is implicitly connected
  - 3.3 V and GND → Sensors
- **Power & Programming:** USB

---

## 📦 Software Dependencies

Install the following libraries via the Arduino Library Manager:

1. **Arduino mbed-enabled Boards** (to support `Ticker`)  
2. **ArduinoBLE**  
3. **Arduino_BMI270_BMM150**  
4. **SparkFun MAX30105** (and its **heartRate** helper)  
5. **TensorFlowLite Micro** by Chirale (or `TensorFlowLite_ESP32` fork that includes all_ops_resolver)  
6. **Wire** (core)

---

## ⚙️ BLE Services & Characteristics

- **Service UUID:** `181C` (Fitness Machine)  
- **Characteristics:**  
  - **Heart Rate** `2AB4` (Read + Notify) – live heart‐rate buffer  
  - **Workout Summary** `2AC8` (Read + Notify) – JSON string:  
    ```json
    {"exercise":"<bench|squat|deadlift>","reps":<float>}
    ```  
  - **Command** `2A3A` (Write) – (reserved for future control commands)

---

## 🚀 Installation & Usage

1. **Clone** this repo into your Arduino sketchbook folder.  
2. **Open** `imu_printer.ino` in Arduino IDE.  
3. **Install** all required libraries via **Sketch → Include Library → Manage Libraries…**  
4. **Select** board **Arduino Nano 33 BLE** and correct COM port.  
5. **Upload** the sketch to your board.  
6. **Pair** with your smartphone app (implements BLE TFT Fitness Machine central).  
   - Live heart rate → updates on the app.  
   - When the phone disconnects, the device sends a final workout summary.

---

## 🔍 How It Works

1. **Sensor Setup:**  
   - BMI270_BMM150 IMU sampled in bursts for motion data.  
   - MAX30102 streamed into a small circular buffer for heart rate.

2. **TinyML Inference:**  
   - **`classification_model`**  
     - Inputs sliding-window IMU data for exercise classification.  
   - **`rep_count_model`**  
     - Inputs processed sensor features to increment a reps counter.

3. **BLE Communication:**  
   - Periodically **notify** live heart rate.  
   - On workout completion or disconnect, **notify** workout summary JSON.

---

## 🔄 Extending & Troubleshooting

- **Model Updates:** Replace `model.h`/`model.c` with your own `.tflite` arrays.  
- **Buffer Sizes:** Tweak `CL_BURST_LEN`, `RC_BURST_LEN` in `imu_printer.ino` to adjust window lengths.  
- **Battery & Power:** Ensure 3.3 V supply to sensors; the Nano 33 BLE’s regulator handles USB power or LiPo input.
---
import serial
import time
import csv
import argparse
from pathlib import Path

# Adjust the port and baudrate accordingly
ser = serial.Serial('COM6', 9600, timeout=1)
time.sleep(0.2)  # Give some time for Arduino to reset

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-n','--name', help='File Name', default='sensor_data', required=False)
parser.add_argument('-t','--time', help='Enable Time Stamping on stdout Display', action='store_true', required=False)

args = parser.parse_args()


curr_path = Path(f"{args.name}.csv")

i = 1
while curr_path.exists():
    curr_path = Path(f"{args.name}({str(i)}).csv")
    i += 1

with open(curr_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Accel_X","Accel_Y","Accel_Z","Gyro_X","Gyro_Y","Gyro_Z"])  # Example headers

    while True:
        line = ser.readline().decode('utf-8').strip()

        if line:
            print(f"[{time.strftime('%H:%M:%S')}] " + line if args.time else line)
            if not line[0].is_digit: # omit bluetooth messages
                continue
            writer.writerow(line.strip('"').split(','))

import serial

portname = "COM7"
baud = 9600

ser = serial.Serial("COM7", 9600)

while 1:
    if ser.in_waiting > 0:
        line = ser.readline()
        print(line)


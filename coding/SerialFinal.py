import serial

portname = "COM7"
baud = 9600

arduino = serial.Serial(portname, baud)


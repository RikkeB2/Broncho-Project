
import pygame
import time
import numpy as np
import cv2
import serial

# values
m_rot = 50
m_bend = 50
m_move = 50

# serial
ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(0.5)
# initialization
ser.write(b'i\n')
time.sleep(0.5)


# joystick 
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count()==0:
    print("No joystick!")
    quit()
#else:
#    print("joystick detect!")

js = pygame.joystick.Joystick(0)
js.init()

print("System Ready!")

''' start control loop '''
try:
    while True:
        pygame.event.pump()        
        num_axes = js.get_numaxes()

        rot = js.get_axis(0)
        bend = js.get_axis(1)
        
        if (rot > 0.5):
            
            if (m_rot >= 2):
                ser.write(b'r\n') # -2 on arduino side
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_rot = int(data)
                        break
                print("rotate clockwise")
            
        if (rot < -0.5):            
            if (m_rot < 180):
                ser.write(b'l\n') # 2 on arduino side
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_rot = int(data)
                        break
                print("rotate counter-clockwise")
            
        if (bend < -0.5):
            if (m_bend >= 42):
                ser.write(b'd\n')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_bend = int(data)
                        break
                print("bend down")

        if (bend > 0.5):
            if (m_bend < 96):
                ser.write(b'u\n')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_bend = int(data)
                        break
                print("bend up")
        
        # check joystick button -> forward or backward
        num_buttons = js.get_numbuttons()
        movef = js.get_button(0)
        moveb = js.get_button(2)

        if (movef > 0.5):
            # servo
            if (m_move < 3600):
                ser.write(b'f\n')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_move = int(data)
                        time.sleep(0.01)
                        break
                print("Move forward")

        if (moveb > 0.5):
            if (m_move >= 50):
                ser.write(b'b\n')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_move = int(data)
                        time.sleep(0.01)
                        break
                print("Move backward")
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("Closing...")
    ser.close()
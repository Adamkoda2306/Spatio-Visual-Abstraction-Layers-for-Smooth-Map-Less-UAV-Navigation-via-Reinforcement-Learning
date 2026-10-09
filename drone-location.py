# manual_keyboard_control_airsim.py
# Works with AirSim 1.5.0
# Keyboard control using Python + keyboard library

import airsim
import keyboard
import time
import os

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# Speed settings
speed = 3      # movement speed
yaw_rate = 30  # turning speed
z_speed = 2    # altitude speed

print("""
==============================
DRONE KEYBOARD CONTROL STARTED
==============================

W  = Forward
S  = Backward
A  = Left
D  = Right
UP ARROW    = Move Up
DOWN ARROW  = Move Down
Q  = Rotate Left
E  = Rotate Right
L  = Land
ESC = Exit

""")

try:
    while True:

        vx = 0
        vy = 0
        vz = 0
        yaw = 0

        # Forward / Backward
        if keyboard.is_pressed('w'):
            vx = speed
        elif keyboard.is_pressed('s'):
            vx = -speed

        # Left / Right
        if keyboard.is_pressed('a'):
            vy = -speed
        elif keyboard.is_pressed('d'):
            vy = speed

        # Up / Down
        if keyboard.is_pressed('up'):
            vz = -z_speed   # negative z = up in AirSim
        elif keyboard.is_pressed('down'):
            vz = z_speed

        # Rotate
        if keyboard.is_pressed('q'):
            yaw = -yaw_rate
        elif keyboard.is_pressed('e'):
            yaw = yaw_rate

        # Send movement command
        client.moveByVelocityBodyFrameAsync(
            vx, vy, vz,
            duration=0.1,
            yaw_mode=airsim.YawMode(True, yaw)
        )

         # Get current position
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position

        # Clear terminal and print position
        os.system('cls' if os.name == 'nt' else 'clear')

        print("==== AIRSIM DRONE CONTROL ====")
        print("W/S = Forward/Backward")
        print("A/D = Left/Right")
        print("UP/DOWN = Up/Down")
        print("Q/E = Rotate")
        print("L = Land")
        print("ESC = Exit")
        print("----------------------------")
        print(f"X Position : {pos.x_val:.2f} m")
        print(f"Y Position : {pos.y_val:.2f} m")
        print(f"Z Position : {pos.z_val:.2f} m")
        print("----------------------------")

        # Land
        if keyboard.is_pressed('l'):
            print("Landing...")
            client.landAsync().join()

        # Exit
        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

# Cleanup
client.hoverAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("Program ended.")
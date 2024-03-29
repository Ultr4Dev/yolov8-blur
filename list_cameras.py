"""
This script lists all the cameras connected to the computer. It has to be used as a standalone script and not imported as a module in another script.
This is because it breaks a bunch of stuff and i dont feel like fixing it.
"""

from device import getDeviceList
device_list = getDeviceList()
for deviceX in device_list:
    current_index = device_list.index(deviceX)
    print(f"Device {current_index}: {deviceX[0]}")

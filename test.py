from device import getDeviceList
device_list = getDeviceList()
for deviceX in device_list:
    current_index = device_list.index(deviceX)
    print(f"Device {current_index}: {deviceX[0]}")

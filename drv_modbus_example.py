from drv_modbus import send
from drv_modbus import request
from pymodbus.client import ModbusTcpClient
import time

c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
c.connect()

home = [408.285, 0.0, 680.0120000000001, 178.969, -0.241, -103.145]

HOME_POSITION = {'x': 120, 'y': 345, 'z': 600, 'rx': -180, 'ry': 0, 'rz': -17 ,'speed':150}

send.Go_Position(c, HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'],
                 HOME_POSITION['rx'], HOME_POSITION['ry'], HOME_POSITION['rz'], HOME_POSITION['speed'])
time.sleep(1) 


# send.Grasp_OFF(c)
# time.sleep(1)
# send.Grasp_ON(c)
# time.sleep(1)
# send.Grasp_OFF(c)
c.close()


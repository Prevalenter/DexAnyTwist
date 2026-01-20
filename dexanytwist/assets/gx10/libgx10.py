import time
import yaml
import sys
import os
import numpy as np

from ..dynamixel_sdk import PortHandler, PacketHandler
from ..motor import Motor
from ..utils import search_ports, get_port_by_serial_number


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


abs_path = os.path.abspath(__file__)

gx10_config_file = os.path.join(os.path.dirname(abs_path), "config.yaml")
gx10_configs = load_config(gx10_config_file)

PROTOCOL_VERSION = gx10_configs["BASIC"]["PROTOCOL_VERSION"]
BAUDRATE = gx10_configs["BASIC"]["BAUDRATE"]
NAME = gx10_configs["HAND"]["NAME"]
NUM_MOTORS = gx10_configs["HAND"]["NUM"]  # 电机数量
THUMB_IDS = gx10_configs["HAND"]["THUMB_IDS"]  # 大拇指ID
INDEX_IDS = gx10_configs["HAND"]["INDEX_IDS"]  # 食指ID
MID_IDS = gx10_configs["HAND"]["MID_IDS"]  # 中指ID

POSKP = gx10_configs["ExtendedPos"]["Motor1"]["Pos_kp"]
POSKI = gx10_configs["ExtendedPos"]["Motor1"]["Pos_ki"]
POSKD = gx10_configs["ExtendedPos"]["Motor1"]["Pos_kd"]
PROVEL = gx10_configs["ExtendedPos"]["Motor1"]["Profile_vel"]
PROACC = gx10_configs["ExtendedPos"]["Motor1"]["Profile_acc"]


class Hand:
    def __init__(self, port=None, serial_number=None) -> None:

        if port == None and serial_number == None:
            print("Please using port or serial_number!")
            sys.exit(0)

        self.is_connected = False
        if port is not None:
            self.port = port
        else:
            if serial_number is not None:
                ports_info = search_ports()
                if serial_number in ports_info:
                    self.port = ports_info[serial_number]
                else:
                    print(f"Serial number: {serial_number} not available!")
                    sys.exit(0)

        self.name = NAME

    def connect(self, curr_limit=1000, goal_current=600, goal_pwm=200):
        """
        连接Hand，并且使能每个电机为默认的力控位置模式,
        curr_limit为电机最大限流(mA，最大不超过1750)，goal_current为电机目标电流(mA， 不能超过curr_limit)，goal_pwm为电机目标PWM(0-885)。
        goal_current限制最大力，goal_pwm限制运动速度。
        """

        portHandler = PortHandler(self.port)
        packetHandler = PacketHandler(PROTOCOL_VERSION)

        if portHandler.openPort() and portHandler.setBaudRate(BAUDRATE):
            print(f"Open {self.port} Success...")
            self.is_connected = True
        else:
            print(f"Failed...")
            self.is_connected = False
            sys.exit(0)

        self.portHandler = portHandler
        self.packetHandler = packetHandler

        self.motors = [
            Motor(i + 1, portHandler, packetHandler, curr_limit)
            for i in range(NUM_MOTORS)
        ]

        for m in self.motors:
            m.init_config(
                curr_limit=curr_limit, goal_current=goal_current, goal_pwm=goal_pwm
            )
            # m.set_profile(PROACC, PROVEL)
            # m.set_pos_pid(POSKP, POSKD, POSKI)
        print(f"{self.name} init done...")

    def off(self):
        """
        失能所有电机
        """
        for m in self.motors:
            m.torq_off()

    def on(self):
        """
        使能所有电机
        """
        for m in self.motors:
            m.torq_on()

    def home(self):
        """
        GX10会到原点
        """
        motors = self.motors
        for m in motors:
            m.set_pos(90)
        time.sleep(1)

    def getjs(self):
        """
        获取GX10关节角度，单位度
        """
        # 固定电机舵盘安装位置，初始角度90，因此减去90
        js = [m.get_pos() - 90 for m in self.motors]
        return js

    def setjs(self, js):
        """
        设置GX10关节角度，单位度
        """
        # 固定电机舵盘安装位置，初始角度90，因此加上90
        for m, j in zip(self.motors, js):
            m.set_pos(j + 90)

    def setj(self, i, j):
        """
        设置第i个关节角度(从1到10)，单位度
        """
        if i < 1 or i > 10:
            print(f"Joint index {i} is out of range!")
            return
        # 固定电机舵盘安装位置，初始角度90，因此加上90
        self.motors[i - 1].set_pos(j + 90)

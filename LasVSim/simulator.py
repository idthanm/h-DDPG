# coding=utf-8

"""
Simulation Base Modal of Autonomous Car Simulation System
Author: Li Bing
Date: 2017-8-23
"""

import _struct as struct
import untangle
import os
from LasVSim.traffic_module import *
from LasVSim.agent_module import *
from xml.dom.minidom import Document
# import StringIO
import time
from LasVSim import data_structures
from LasVSim.traffic_module import TrafficData
from math import cos, sin, pi, fabs

class Simulation(object):
    """Simulation Class.

    Simulation class for one simulation.

    Attributes:
        tick_count: Simulation run time. Counted by simulation steps.
        sim_time: Simulation run time. Counted by simulation steps multiply step
            length.
        stopped: A bool variable as a flag indicating whether simulation is
            ended.
        traffic: A traffic module instance.
        agent: A Agent module instance.
        data: A data module instance.
        other_vehicles: A list containing all other vehicle's info at current
            simulation step from traffic module.
        light_status: A dic variable containing current intersection's traffic
            light state for each direction.



    """

    def __init__(self, default_setting_path=None):

        self.tick_count = 0  # Simulation run time. Counted by simulation steps.
        self.sim_time = 0.0  # Simulation run time. Counted by steps multiply stpe length.
        self.other_vehicles = None  # 仿真中所有他车状态信息
        self.light_status = None  # 当前十字路口信号灯状态
        self.stopped = False  # 仿真结束标志位
        self.simulation_loaded = False  # 仿真载入标志位
        self.traffic_data = TrafficData()  # 初始交通流数据对象
        self.settings = Settings(file_path=default_setting_path)  # 仿真设置对象
        self.step_length = self.settings.step_length
        self.external_control_flag = False  # 外部控制输入标识，若外部输入会覆盖内部控制器
        self.traffic = None
        self.agent = None
        self.ego_history = None
        self.data = None
        self.seed = None

        # self.reset(settings=self.settings, overwrite_settings=overwrite_settings, init_traffic_path=init_traffic_path)
        # self.sim_step()

    def set_seed(self, seed=None):  # call this just before training (usually only once)
        if seed is not None:
            self.seed = seed

    def reset(self, settings=None, overwrite_settings=None, init_traffic_path=None):
        """Clear previous loaded module.

        Args:
            settings: LasVSim's setting class instance. Containing current
                simulation's configuring information.
        """
        if hasattr(self, 'traffic'):
            del self.traffic
        if hasattr(self, 'agent'):
            del self.agent
        if hasattr(self, 'data'):
            del self.data

        self.tick_count = 0
        self.settings = settings
        if overwrite_settings is not None:
            self.settings.start_point = overwrite_settings['init_state']
        self.stopped = False
        # self.data = Data()
        self.ego_history = {}

        """Load traffic module."""
        step_length = self.settings.step_length * self.settings.traffic_frequency
        self.traffic = Traffic(path=settings.map,
                               traffic_type=settings.traffic_type,
                               traffic_density=settings.traffic_lib,
                               step_length=step_length,
                               init_traffic=self.traffic_data.load_traffic(init_traffic_path),
                               seed=self.seed)
        self.traffic.init(settings.start_point, settings.car_length)
        self.other_vehicles = self.traffic.get_vehicles()

        """Load agent module."""
        self.agent = Agent(settings)

    def load_scenario(self, path, overwrite_settings=None):
        """Load an existing LasVSim simulation configuration file.

        Args:
            path:
        """
        if os.path.exists(path):
            settings = Settings()
            settings.load(path+'/simulation_setting_file.xml')
            # self.reset(settings)
            self.reset(settings, overwrite_settings=overwrite_settings, init_traffic_path=path)
            self.simulation_loaded = True
            return
        print('\033[31;0mSimulation loading failed: 找不到对应的根目录\033[0m')
        self.simulation_loaded = False

    # def save_scenario(self, path):
    #     """Save current simulation configuration
    #
    #     Args:
    #         path: Absolute path.
    #     """
    #     self.settings.save(path)

    def export_data(self, path):
        self.data.export_csv(path)

    def sim_step(self, steps=None):
        if steps is None:
            steps = 1
        for step in range(steps):
            if self.stopped:
                print("Simulation Finished")
                return False

            # traffic
            if self.tick_count % self.settings.traffic_frequency == 0:
                self.traffic.set_own_car(self.agent.x,
                                         self.agent.y,
                                         self.agent.v,
                                         self.agent.heading)
                self.traffic.sim_step()
                self.other_vehicles = self.traffic.get_vehicles()

            if not self.__collision_check():
                self.stopped = True


            # 保存当前步仿真数据
            # self.data.append(
            #     self_status=[self.tick_count * float(self.settings.step_length),
            #                  self.agent.dynamic.x,
            #                  self.agent.dynamic.y,
            #                  self.agent.dynamic.v,
            #                  self.agent.dynamic.heading],
            #     self_info=self.agent.dynamic.get_info(),
            #     vehicles=self.other_vehicles
            #     # light_values=self.traffic.get_light_values(),
            #     # trajectory=self.agent.route
            #     # dis=self.traffic.get_current_distance_to_stopline(),
            #     # speed_limit=self.traffic.get_current_lane_speed_limit()
            #     )
            self.tick_count += 1
        return True

    def get_all_objects(self):
        return self.other_vehicles

    def get_ego_info(self):
        return self.agent.get_info()

    def get_ego_road_related_info(self):
        return self.traffic.get_road_related_info_of_ego()

    def get_time(self):
        return self.traffic.sim_time

    def __collision_check(self):
        for vehs in self.other_vehicles:
            if (fabs(vehs['x']-self.agent.x) < 10 and
               fabs(vehs['y']-self.agent.y) < 2):
                self.ego_x0 = (self.agent.x +
                               cos(self.agent.heading/180*pi)*self.agent.lw)
                self.ego_y0 = (self.agent.y +
                               sin(self.agent.heading/180*pi)*self.agent.lw)
                self.ego_x1 = (self.agent.x -
                               cos(self.agent.heading/180*pi)*self.agent.lw)
                self.ego_y1 = (self.agent.y -
                               sin(self.agent.heading/180*pi)*self.agent.lw)
                self.surrounding_lw = (vehs['length']-vehs['width'])/2
                self.surrounding_x0 = (
                    vehs['x'] + cos(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.surrounding_y0 = (
                    vehs['y'] + sin(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.surrounding_x1 = (
                    vehs['x'] - cos(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.surrounding_y1 = (
                    vehs['y'] - sin(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.collision_check_dis = ((vehs['width']+self.agent.width)/2+0.5)**2
                if ((self.ego_x0-self.surrounding_x0)**2 +
                    (self.ego_y0-self.surrounding_y0)**2
                        < self.collision_check_dis):
                    return False
                if ((self.ego_x0-self.surrounding_x1)**2 +
                    (self.ego_y0-self.surrounding_y1)**2
                        < self.collision_check_dis):
                    return False
                if ((self.ego_x1-self.surrounding_x1)**2 +
                    (self.ego_y1-self.surrounding_y1)**2
                        < self.collision_check_dis):
                    return False
                if ((self.ego_x1-self.surrounding_x0)**2 +
                    (self.ego_y1-self.surrounding_y0)**2
                        < self.collision_check_dis):
                    return False
        return True


class Settings:  # 可以直接和package版本的Settings类替换,需要转换路径点的yaw坐标
    """
    Simulation Settings Class
    """

    def __init__(self, file_path=None):
        self.load(file_path)

    def __del__(self):
        pass

    def load(self, filePath=None):
        if filePath is None:
            filePath = DEFAULT_SETTING_FILE
        self.__parse_xml(filePath)
        self.__load_step_length()
        self.__load_map()
        self.__load_self_car()
        self.__load_traffic()
        self.__load_start_point()

    def __parse_xml(self, path):
        f = open(path)
        self.root = untangle.parse(f.read()).Simulation

    def __load_step_length(self):
        self.step_length = int(self.root.StepLength.cdata)

    def __load_map(self):
        self.map = str(self.root.Map.Type.cdata)

    def __load_start_point(self):
        self.start_point = [float(self.root.Start_point.X.cdata),
                            float(self.root.Start_point.Y.cdata),
                            float(self.root.Start_point.Speed.cdata),
                            float(self.root.Start_point.Yaw.cdata)]


    def __load_traffic(self):
        self.traffic_type = str(self.root.Traffic.Type.cdata)
        self.traffic_lib = str(self.root.Traffic.Lib.cdata)
        self.traffic_frequency = int(self.root.Traffic.Frequency.cdata)

    def __load_self_car(self):
        self.car_length = float(self.root.SelfCar.Length.cdata)
        self.car_width = float(self.root.SelfCar.Width.cdata)
        self.car_weight = float(self.root.SelfCar.Weight.cdata)
        self.car_center2head = float(self.root.SelfCar.CenterToHead.cdata)
        self.car_faxle2center = float(self.root.SelfCar.FAxleToCenter.cdata)
        self.car_raxle2center = float(self.root.SelfCar.RAxleToCenter.cdata)



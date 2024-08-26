"""
 Copyright (c) 2024, Your Name
 All rights reserved.
 CopyrightText: SEO HYEON HO
 License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.spatial import KDTree
from typing import List, Dict, Tuple, Any
import copy
import logging
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Settings:
    max_speed: float
    radius: float
    time_step: float
    neighbor_dist: float
    max_neighbors: int
    time_horizon: float
    velocity: np.ndarray

# 사용자 정의 파라미터
settings = Settings(
    max_speed=2.0,
    radius=7.0,
    time_step=0.125,
    neighbor_dist=19.0,
    max_neighbors=5,
    time_horizon=3.0,
    velocity=np.array([1.0, 1.0, 1.0])
)

class Plane:
    def __init__(self, point: np.ndarray = np.zeros(3), normal: np.ndarray = np.zeros(3)):
        self.point: np.ndarray = point
        self.normal: np.ndarray = normal

def find_neighbors(agents: np.ndarray, settings: Settings) -> List[Dict[str, Any]]:
    tree = KDTree(agents)
    distances, indices = tree.query(agents, k=settings.max_neighbors + 1)
    
    neighbors_list = []
    for i, (agent, agent_distances, agent_indices) in enumerate(zip(agents, distances, indices)):
        agent_info = {
            'agent': {'id': i, 'me_coords': tuple(agent), 'velocity': settings.velocity},
            'neighbors': []
        }
        for dist, idx in zip(agent_distances[1:], agent_indices[1:]):
            if dist <= settings.neighbor_dist:
                neighbor_agent = agents[idx]
                agent_info['neighbors'].append({
                    'agent_id': idx,
                    'agent_coords': tuple(neighbor_agent),
                    'distance': dist,
                    'collision': dist <= settings.radius * 2
                })
        neighbors_list.append(agent_info)
    
    return neighbors_list

def compute_new_velocity(pref_velocity: np.ndarray, agent_info: Dict[str, Any], settings: Settings) -> np.ndarray:
    orca_planes = []
    for neighbor in agent_info['neighbors']:
        relative_position = np.array(neighbor['agent_coords']) - np.array(agent_info['agent']['me_coords'])
        relative_velocity = settings.velocity - np.array([0, 0, 0])
        dist_sq = neighbor['distance'] ** 2
        combined_radius = settings.radius * 2
        combined_radius_sq = combined_radius ** 2

        plane = Plane()
        u = np.zeros(3)

        if dist_sq > combined_radius_sq:
            w = relative_velocity - (1 / settings.time_horizon) * relative_position
            w_length_sq = np.dot(w, w)
            dot_product1 = np.dot(w, relative_position)

            if dot_product1 < 0.0 and dot_product1 * dot_product1 > combined_radius_sq * w_length_sq:
                w_length = np.sqrt(w_length_sq)
                plane.normal = w / w_length if w_length > 0 else np.zeros(3)
                u = (combined_radius / settings.time_horizon - w_length) * plane.normal
            else:
                leg = np.sqrt(dist_sq - combined_radius_sq)
                if relative_position[0] * w[1] - relative_position[1] * w[0] > 0.0:
                    plane.normal = np.array([relative_position[0] * leg - relative_position[1] * combined_radius,
                                             relative_position[1] * leg + relative_position[0] * combined_radius,
                                             0.0]) / dist_sq
                else:
                    plane.normal = -np.array([relative_position[0] * leg + relative_position[1] * combined_radius,
                                              relative_position[1] * leg - relative_position[0] * combined_radius,
                                              0.0]) / dist_sq

                dot_product2 = np.dot(relative_velocity, plane.normal)
                u = (dot_product2 - (1 / settings.time_horizon)) * plane.normal
        else:
            w = relative_velocity - relative_position / settings.time_step
            w_length = np.linalg.norm(w)
            plane.normal = w / w_length if w_length > 0 else np.zeros(3)
            u = (combined_radius / settings.time_step - w_length) * plane.normal

        plane.point = settings.velocity + 0.5 * u
        orca_planes.append(plane)

    new_velocity, plane_leng = linear_program3(orca_planes, settings.max_speed, pref_velocity, False, np.array([1, 1, 1]))
    
    if plane_leng < len(orca_planes):
        new_velocity = linear_program4(orca_planes, plane_leng, settings.max_speed, new_velocity)

    if np.linalg.norm(new_velocity) == 0:
        random_factors = np.random.uniform(0, 0.2, size=3)
        new_velocity = pref_velocity * random_factors
    
    # xy 평면에서만 시뮬레이션 테스트 할 때
    # new_velocity[-1] = 0

    return new_velocity

def linear_program1(planes: List[Plane], plane_no: int, line: Tuple[np.ndarray, np.ndarray], radius: float, opt_velocity: np.ndarray, direction_opt: bool, result: np.ndarray) -> bool:
    dot_product = line[0].dot(line[1]) # 실수값
    discriminant = dot_product * dot_product + radius * radius - line[0] @ line[0] 
    print(f'<Agent> linear_program1: discriminant={discriminant} = {dot_product * dot_product} + {radius * radius} + {line[0] * line[0]}') # 3차원 nd어레이값
    if discriminant < 0.0:
        return False

    sqrt_discriminant = np.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant
    print(f'<Agent> linear_program1: discriminant={discriminant} t_left={t_left} t_right={t_right} plane_no={plane_no}')
    for i in range(plane_no):
        numerator = (planes[i].point - line[0]).dot(planes[i].normal)
        denominator = line[1].dot(planes[i].normal)
        if abs(denominator) <= 1e-6:   
            if numerator > 0.0:
                return False
            else:
                continue

        t = numerator / denominator
        print(f'<Agent> linear_program1: i={i} t={t}')
        if denominator >= 0.0:
            t_left = max(t_left, t)
        else:
            t_right = min(t_right, t)

        if t_left > t_right:
            return False
    if direction_opt:
        if opt_velocity.dot(line[1]) > 0.0:
            result[:] = line[0] + t_right * line[1]
        else:
            result[:] = line[0] + t_left * line[1]
    else:
        t = line[1].dot(opt_velocity - line[0])
        if t < t_left:
            result[:] = line[0] + t_left * line[1]
        elif t > t_right:
            result[:] = line[0] + t_right * line[1]
        else:
            result[:] = line[0] + t * line[1]

    return True

def linear_program2(planes: List[Plane], plane_no: int, radius: float, opt_velocity: np.ndarray, direction_opt: bool, result: np.ndarray) -> bool:
    plane_dist = planes[plane_no].point.dot(planes[plane_no].normal)
    plane_dist_sq = plane_dist * plane_dist
    radius_sq = radius * radius
    if plane_dist_sq > radius_sq:
        return False

    plane_radius_sq = radius_sq - plane_dist_sq
    plane_center = plane_dist * planes[plane_no].normal

    if direction_opt:
        plane_opt_velocity = opt_velocity - (opt_velocity.dot(planes[plane_no].normal)) * planes[plane_no].normal
        plane_opt_velocity_length_sq = (plane_opt_velocity[0] * plane_opt_velocity[0] + plane_opt_velocity[1] * plane_opt_velocity[1] + plane_opt_velocity[2] * plane_opt_velocity[2])
        if plane_opt_velocity_length_sq <= 1e-6:
            result[:] = plane_center
        else:
            result[:] = plane_center + np.sqrt(plane_radius_sq / plane_opt_velocity_length_sq) * plane_opt_velocity
    else:
        result[:] = opt_velocity + ((planes[plane_no].point - opt_velocity).dot(planes[plane_no].normal)) * planes[plane_no].normal
        if (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]) > radius_sq:
            plane_result = result - plane_center
            plane_result_length_sq = (plane_result[0] * plane_result[0] + plane_result[1] * plane_result[1] + plane_result[2] * plane_result[2])
            result[:] = plane_center + np.sqrt(plane_radius_sq / plane_result_length_sq) * plane_result
    print(f'<Agent> linear_program2: plane_center={plane_center} result={result} plane_no={plane_no}')

    for i in range(plane_no):
        if planes[i].normal.dot(planes[i].point - result) > 0.0:
            cross_product = np.cross(planes[i].normal, planes[plane_no].normal)

            if (cross_product[0] * cross_product[0] + cross_product[1] * cross_product[1] + cross_product[2] * cross_product[2])<= 1e-6:
                return False

            line_normal = np.cross(cross_product, planes[plane_no].normal)
            line_point = planes[plane_no].point + (((planes[i].point - planes[plane_no].point).dot(planes[i].normal)) / (line_normal.dot(planes[i].normal))) * line_normal

            if not linear_program1(planes, i, (line_point, line_normal), radius, opt_velocity, direction_opt, result):
                return False

    return True

def linear_program3(planes: List[Plane], radius: float, opt_velocity: np.ndarray, direction_opt: bool, new_velocity: np.ndarray) -> int:
    if direction_opt:
        new_velocity[:] = opt_velocity * radius
    elif (opt_velocity[0] * opt_velocity[0] + opt_velocity[1] * opt_velocity[1] + opt_velocity[2] * opt_velocity[2]) > radius * radius:
        new_velocity[:] = opt_velocity / np.linalg.norm(opt_velocity) * radius
    else:
        new_velocity[:] = opt_velocity

    for i, pl in enumerate(planes):
        if pl.normal.dot(pl.point - new_velocity) > 0.0:
            temp_result = copy.deepcopy(new_velocity)

            if not linear_program2(planes, i, radius, opt_velocity, direction_opt, new_velocity):
                new_velocity[:] = temp_result
                return new_velocity, i
    print(f'<Agent> linear_program3: direction_opt={direction_opt} result={new_velocity}')
    return new_velocity, len(planes)

def linear_program4(planes: List[Plane], begin_plane: int, radius: float, new_velocity: np.ndarray):
    distance = 0.0
    
    for i in range(begin_plane, len(planes)):
        if planes[i].normal.dot(planes[i].point - new_velocity) > distance:
            proj_planes = []

            for j in range(i):
                cross_product = np.cross(planes[j].normal, planes[i].normal)
                if (cross_product[0] * cross_product[0] + cross_product[1] * cross_product[1] + cross_product[2] * cross_product[2]) <= 1e-6:
                    if planes[i].normal.dot(planes[j].normal) > 0.0:
                        continue
                    else:
                        plane_point = 0.5 * (planes[i].point + planes[j].point)

                else:
                    line_normal = np.cross(cross_product, planes[i].normal)
                    numerator = (planes[j].point - planes[i].point).dot(planes[j].normal)
                    denominator = line_normal.dot(planes[j].normal)
                    plane_point = planes[i].point + (numerator / denominator) * line_normal
                
                plane_nor = planes[j].normal - planes[i].normal
                plane_normal = plane_nor / np.linalg.norm(plane_nor)
                add_plane = Plane()
                add_plane.point = plane_point
                add_plane.normal = plane_normal
                proj_planes.append(add_plane)
            
            # temp_result = result.copy() # 얕은 복사인가 깊은 복사인가? 깊은 복사겠지? 
            temp_result = copy.deepcopy(new_velocity) 

            if linear_program3(proj_planes, radius, planes[i].normal, True, new_velocity)[1] < len(proj_planes):
                new_velocity[:] = temp_result

            distance = planes[i].normal.dot(planes[i].point - new_velocity)
            print(f'<Agent> linear_program4: proj_planes={proj_planes} begin_plane={begin_plane} distance={distance} result={new_velocity}')
    
    return new_velocity

def reached_goals(agents: np.ndarray, goals: np.ndarray, threshold: float = 1.0) -> bool:
    return np.all(np.linalg.norm(agents - goals, axis=1) < threshold)

def update(frame: int, agents: np.ndarray, goals: np.ndarray, scatter: plt.scatter, settings: Settings) -> Tuple[plt.scatter]:
    if not reached_goals(agents, goals):
        neighbors_list = find_neighbors(agents, settings)
        for i, info in enumerate(neighbors_list):
            pref_velocity = goals[i] - agents[i]
            pref_velocity = pref_velocity / np.linalg.norm(pref_velocity) * settings.max_speed
            
            if info['neighbors']:
                new_velocity = compute_new_velocity(pref_velocity, info, settings)
                speed = np.linalg.norm(new_velocity)
                if speed > settings.max_speed:
                    new_velocity = new_velocity / speed * settings.max_speed
            else:
                new_velocity = pref_velocity

            agents[i] += new_velocity
        
        scatter._offsets3d = (agents[:, 0], agents[:, 1], agents[:, 2])
        logger.info(f"Frame {frame}: Agents updated")

    return (scatter,)

# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     agents = np.array([
#         [-101, -100, -100], [-100, -103,  100], [-104,  100, -104], [-103,  100,  100],
#         [ 100, -100, -100], [ 100, -104,  100], [ 101,  100, -101], [ 103,  104,  100]
#     ], dtype=float)
#     goals = -agents
#     return agents, goals

# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     num_agents = 16
#     radius = 100

#     # 구면 좌표계에서 임의의 점 생성
#     theta = np.random.uniform(0, 2*np.pi, num_agents)  # 방위각
#     phi = np.random.uniform(0, np.pi, num_agents)      # 극각

#     # 구면 좌표계를 직교 좌표계로 변환
#     x = radius * np.sin(phi) * np.cos(theta)
#     y = radius * np.sin(phi) * np.sin(theta)
#     z = radius * np.cos(phi)

#     # agents 배열 생성
#     agents = np.column_stack((x, y, z))

#     # goals는 agents의 반대 지점
#     goals = -agents

#     return agents, goals

# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     num_agents = 30

#     # 지정된 범위 내에서 랜덤하게 agent 위치 생성
#     agents = np.random.uniform(low=-100, high=100, size=(num_agents, 3))

#     # goals는 agents의 반대 지점
#     goals = np.array([
#         [-60, -60, 45],[-60, -60, 30],[-60, -60, -45],
#         [-50, -50, 60],[-50, -50, 15],[-50, -50, -60],
#         [-40, -40, 67],[-40, -40, 0],[-40, -40, -67],
#         [-30, -30, 60],[-30, -30, -15],[-30, -30, -60],
#         [-20, -20, 45],[-20, -20, -30],[-20, -20, -45],
#         [20, 20, 45],[20, 20, 30],[20, 20, -45],
#         [30, 30, 60],[30, 30, 15],[30, 30, -60],
#         [40, 40, 67],[40, 40, 0],[40, 40, -67],
#         [50, 50, 60],[50, 50, -15],[50, 50, -60],
#         [60, 60, 45],[60, 60, -30],[60, 60, -45],
#     ], dtype=float)

#     return agents, goals

# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     num_agents = 16

#     # Agents: x=-100 평면의 반지름 100인 원 위에 균등 배치
#     theta = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
#     agents = np.zeros((num_agents, 3))
#     agents[:, 0] = -100  # x 좌표
#     agents[:, 1] = 100 * np.cos(theta)  # y 좌표
#     agents[:, 2] = 100 * np.sin(theta)  # z 좌표

#     # Goals: x=100 평면의 X 모양으로 균등 배치
#     goals = np.zeros((num_agents, 3))
#     goals[:, 0] = 100  # x 좌표

#     # X 모양의 첫 번째 대각선
#     diagonal1 = np.linspace(-100, 100, num_agents // 2)
#     goals[:num_agents//2, 1] = diagonal1
#     goals[:num_agents//2, 2] = diagonal1

#     # X 모양의 두 번째 대각선
#     diagonal2 = np.linspace(100, -100, num_agents // 2)
#     goals[num_agents//2:, 1] = diagonal2
#     goals[num_agents//2:, 2] = -diagonal2

#     return agents, goals

def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
    num_agents = 8
    radius = 100
    # 방위각을 랜덤하게 생성
    theta = np.random.uniform(0, 2 * np.pi, num_agents)  # 방위각
    # xy 평면에서의 좌표 계산
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(num_agents)  # z 좌표는 0으로 설정
    # agents 배열 생성
    agents = np.column_stack((x, y, z))
    # goals는 agents의 반대 지점 (xy 평면에서 반대 방향)
    goals = -agents
    return agents, goals

def main():
    agents, goals = setup_scenario()
    logger.info("Scenario set up completed")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_zlim(-110, 110)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    scatter = ax.scatter(agents[:, 0], agents[:, 1], agents[:, 2], c='b', s=100, label='Agents')
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c='r', s=100, label='Goals')

    for agent, goal in zip(agents, goals):
        ax.plot([agent[0], goal[0]], [agent[1], goal[1]], [agent[2], goal[2]], 'k--', alpha=0.3)

    ax.legend()

    anim = animation.FuncAnimation(fig, update, frames=200, fargs=(agents, goals, scatter, settings),
                                   interval=50, blit=False)

    logger.info("Animation started")
    plt.show()

if __name__ == "__main__":
    main()

# 단위 테스트 예시
import unittest

class TestORCA(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            max_speed=2.0,
            radius=10.0,
            time_step=0.125,
            neighbor_dist=100.0,
            max_neighbors=5,
            time_horizon=3.0,
            velocity=np.array([1.0, 1.0, 1.0])
        )

    def test_find_neighbors(self):
        agents = np.array([[0, 0, 0], [50, 50, 50], [100, 100, 100]])
        neighbors = find_neighbors(agents, self.settings)
        self.assertEqual(len(neighbors), 3)
        self.assertEqual(len(neighbors[0]['neighbors']), 1)
        self.assertEqual(neighbors[0]['neighbors'][0]['agent_id'], 1)

    def test_reached_goals(self):
        agents = np.array([[0, 0, 0], [1, 1, 1]])
        goals = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        self.assertFalse(reached_goals(agents, goals))
        self.assertTrue(reached_goals(agents, goals, threshold=1.0))

if __name__ == '__main__':
    unittest.main()
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
import matplotlib.animation as animation
from scipy.spatial import KDTree
from typing import List, Dict, Tuple, Any
import copy
import logging
from dataclasses import dataclass

# logging setting
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
    obstacle_radius: float

settings = Settings(
    max_speed=30.0,
    radius=7.0,                 
    time_step=0.05,            
    neighbor_dist= 2 * 30,     
    max_neighbors=5,            
    time_horizon=3.0,           
    velocity=np.array([0.0, 0.0, 0.0]),
    obstacle_radius = 10.0
)

class Plane:
    def __init__(self, point: np.ndarray = np.zeros(3), normal: np.ndarray = np.zeros(3)):
        self.point: np.ndarray = point
        self.normal: np.ndarray = normal

def find_neighbors(agents: np.ndarray, velocities: np.ndarray, settings: Settings) -> List[Dict[str, Any]]:
    tree = KDTree(agents)
    distances, indices = tree.query(agents, k=settings.max_neighbors + 1)
    
    neighbors_list = []
    for i, (agent, agent_velocity, agent_distances, agent_indices) in enumerate(zip(agents, velocities, distances, indices)):
        agent_info = {
            'agent': {
                'id': i, 
                'me_coords': tuple(agent), 
                'velocity': agent_velocity
            },
            'neighbors': []
        }
        for dist, idx in zip(agent_distances[1:], agent_indices[1:]):
            if dist <= settings.neighbor_dist:
                neighbor_agent = agents[idx]
                agent_info['neighbors'].append({
                    'agent_id': idx,
                    'agent_coords': tuple(neighbor_agent),
                    'velocity': velocities[idx],
                    'distance': dist,
                    'collision': dist <= settings.radius * 2
                })
        neighbors_list.append(agent_info)
    
    return neighbors_list

def compute_new_velocity(pref_velocity: np.ndarray, agent_info: Dict[str, Any], settings: Settings) -> np.ndarray:
    orca_planes = []
    current_velocity = np.array(agent_info['agent']['velocity'])
    for neighbor in agent_info['neighbors']:
        relative_position = np.array(neighbor['agent_coords']) - np.array(agent_info['agent']['me_coords'])
        relative_velocity = np.array(neighbor['velocity']) - current_velocity
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

        plane.point = current_velocity + 0.5 * u
        orca_planes.append(plane)

    new_velocity, plane_leng = linear_program3(orca_planes, settings.max_speed, pref_velocity, False, current_velocity)
    if plane_leng < len(orca_planes):
        new_velocity = linear_program4(orca_planes, plane_leng, settings.max_speed, new_velocity)
    if np.linalg.norm(new_velocity) == 0:
        random_factors = np.random.uniform(0, 0.2, size=3)
        new_velocity = pref_velocity * random_factors
    # only test for x-y plane
    new_velocity[-1] = 0
    return new_velocity

def linear_program1(planes: List[Plane], plane_no: int, line: Tuple[np.ndarray, np.ndarray], radius: float, opt_velocity: np.ndarray, direction_opt: bool, result: np.ndarray) -> bool:
    dot_product = line[0].dot(line[1]) 
    discriminant = dot_product * dot_product + radius * radius - line[0] @ line[0] 
    if discriminant < 0.0:
        return False

    sqrt_discriminant = np.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant
    for i in range(plane_no):
        numerator = (planes[i].point - line[0]).dot(planes[i].normal)
        denominator = line[1].dot(planes[i].normal)
        if abs(denominator) <= 1e-6:   
            if numerator > 0.0:
                return False
            else:
                continue

        t = numerator / denominator
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
            
            temp_result = copy.deepcopy(new_velocity) 

            if linear_program3(proj_planes, radius, planes[i].normal, True, new_velocity)[1] < len(proj_planes):
                new_velocity[:] = temp_result

            distance = planes[i].normal.dot(planes[i].point - new_velocity)

    return new_velocity

def reached_goals(agents: np.ndarray, goals: np.ndarray, threshold: float = 2.0) -> bool:
    return np.all(np.linalg.norm(agents - goals, axis=1) < threshold)

def update(frame: int, agents: np.ndarray, velocities: np.ndarray, goals: np.ndarray, scatter: plt.scatter, settings: Settings, time_text: plt.Text, elapsed_time: float) -> Tuple[plt.scatter, plt.Text, float]:
    if not reached_goals(agents[:-1], goals[:-1]): 
        neighbors_list = find_neighbors(agents, velocities, settings)
        for i, info in enumerate(neighbors_list):
            if i == len(agents) - 1:  
                continue  
            
            pref_velocity = goals[i] - agents[i]
            if np.linalg.norm(pref_velocity) > 1e-6:
                pref_velocity = pref_velocity / np.linalg.norm(pref_velocity) * settings.max_speed
            else:
                pref_velocity = np.zeros(3)
            
            if info['neighbors']:
                new_velocity = compute_new_velocity(pref_velocity, info, settings)
                speed = np.linalg.norm(new_velocity)
                if speed > settings.max_speed:
                    new_velocity = new_velocity / speed * settings.max_speed
            else:
                new_velocity = pref_velocity

            agents[i] += new_velocity * settings.time_step
            velocities[i] = new_velocity
        
        # Keep the obstacle stationary
        velocities[-1] = np.zeros(3)
        
        scatter._offsets3d = (agents[:, 0], agents[:, 1], agents[:, 2])
        logger.info(f"Frame {frame}: Agents updated")

        elapsed_time += settings.time_step
    
    else:
        logger.info("All agents have reached their goals. Stopping time measurement.")
    
    # update time
    time_text.set_text(f'Time: {elapsed_time:.2f} s')

    return scatter, time_text, elapsed_time


# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     num_agents = 40
#     radius = 100
#     theta = np.random.uniform(0, 2*np.pi, num_agents)  
#     phi = np.random.uniform(0, np.pi, num_agents)     
#     x = radius * np.sin(phi) * np.cos(theta)
#     y = radius * np.sin(phi) * np.sin(theta)
#     z = radius * np.cos(phi)
#     agents = np.column_stack((x, y, z))
#     goals = -agents
#     return agents, goals


# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     num_agents = 10
#     radius = 100
#     theta = np.random.uniform(0, 2 * np.pi, num_agents)  
#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)
#     z = np.zeros(num_agents) 
#     agents = np.column_stack((x, y, z))
#     goals = -agents
#     return agents, goals

# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     num_agents = 100
#     radius = 100
#     indices = np.arange(num_agents)
#     theta = np.pi * (3 - np.sqrt(5)) * indices  
#     phi = np.arccos(1 - 2 * (indices + 0.5) / num_agents)  
#     x = radius * np.sin(phi) * np.cos(theta)
#     y = radius * np.sin(phi) * np.sin(theta)
#     z = radius * np.cos(phi)
#     agents = np.column_stack((x, y, z))
#     goals = -agents
#     return agents, goals

# def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
#     num_agents =11 
#     radius = 100
#     angles = np.linspace(0, 2 * np.e, num_agents, endpoint=False)  
#     x = radius * np.cos(angles)   
#     y = radius * np.sin(angles)  
#     z = np.zeros(num_agents)      
#     agents = np.column_stack((x, y, z))
#     goals = -agents
#     # print(f'start:{agents}, goal_points:{goals}')
#     return agents, goals

def setup_scenario() -> Tuple[np.ndarray, np.ndarray]:
    num_agents = 19  
    radius = 100

    angles = np.linspace(0, 2 * np.pi, num_agents - 1, endpoint=False) 
    x = radius * np.cos(angles)  
    y = radius * np.sin(angles)  
    z = np.zeros(num_agents-1)      

    agents = np.column_stack((x, y, z))
    goals = -agents

    obstacle_agent = np.array([[0, 0, 0]])
    obstacle_goal = np.array([[0, 0, 0]])

    agents = np.vstack((agents, obstacle_agent))
    goals = np.vstack((goals, obstacle_goal))

    return agents, goals

def main():
    agents, goals = setup_scenario()
    velocities = np.zeros_like(agents) 
    logger.info("Scenario set up completed")

    obstacle_radius = 10.0

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_zlim(-110, 110)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    scatter = ax.scatter(agents[:-1, 0], agents[:-1, 1], agents[:-1, 2], c='b', s=100, label='Agents')
    ax.scatter(goals[:-1, 0], goals[:-1, 1], goals[:-1, 2], c='r', s=100, label='Goals')
    
    ax.scatter(agents[-1, 0], agents[-1, 1], agents[-1, 2], c='k', s=obstacle_radius**2, label='Obstacle')

    for agent, goal in zip(agents[:-1], goals[:-1]):
        ax.plot([agent[0], goal[0]], [agent[1], goal[1]], [agent[2], goal[2]], 'k--', alpha=0.3)
    
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    ax.legend()
    
    elapsed_time = 0.0
    
    def update_wrapper(frame):
        nonlocal elapsed_time
        result = update(frame, agents, velocities, goals, scatter, settings, time_text, elapsed_time)
        elapsed_time = result[2]  
        return result[:2] 
    
    anim = animation.FuncAnimation(fig, update_wrapper, frames=2000, interval=50, blit=False)

    logger.info("Animation started")
    plt.show()

if __name__ == "__main__":
    main()

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
            velocity=np.array([1.0, 1.0, 1.0]),
            obstacle_radius = 10.0
        )

    def test_find_neighbors(self):
        agents = np.array([[0, 0, 0], [50, 50, 50], [100, 100, 100]])
        velocities = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        neighbors = find_neighbors(agents, velocities, self.settings)
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
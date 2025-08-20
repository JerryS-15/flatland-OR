import argparse
import hashlib
from tqdm import tqdm
import numpy as np
import pickle
import os
from time import sleep

import flatland
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator,rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator,schedule_from_file
from flatland.envs.malfunction_generators  import malfunction_from_params, MalfunctionParameters
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

def create_env(env_params, seed):
    return RailEnv(
        number_of_agents=env_params['number_of_agents'],
        width=env_params['width'],
        height=env_params['height'],
        rail_generator=sparse_rail_generator(
            max_num_cities=env_params['max_num_cities'],
            grid_mode=False,
            max_rails_between_cities=env_params['max_rails_between_cities'],
            max_rails_in_city=env_params['max_rail_pairs_in_city'],
        ),
        schedule_generator=sparse_schedule_generator(
            speed_ratio_map=env_params['speed_ratio_map'],
        ),
        obs_builder_object=DummyObservationBuilder(),
        malfunction_generator_and_process_data=malfunction_from_params(
            MalfunctionParameters(
                malfunction_rate=env_params['malfunction_rate'],
                min_duration=env_params['min_duration'],
                max_duration=env_params['max_duration']
            )
        ),
        # malfunction_generator_and_process_data=None,
        remove_agents_at_target=True,
        random_seed=seed
    )

def load_env_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_or_actions(step):
    action_data = load_env_data(action_path)
    return action_data[step]

if __name__ == "__main__":

    seed = 1
    env_renderer_enable = True
    fps = 30

    env_path = f"or_solution_data/env_data_v2_{seed}.pkl"
    action_path = f"or_solution_data/action_data_v2_{seed}.pkl"

    flatland_parameters = {
        # Flatland Env
        "number_of_agents": 5,
        "width": 30,
        "height": 35,
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "speed_ratio_map": {1.0: 1 / 3, 0.5: 1 / 3, 0.25: 1 / 3},
        # Flatland - malfunction
        "malfunction_rate": 1 / 4500,
        "min_duration": 20,
        "max_duration": 50
    }

    env = create_env(flatland_parameters, seed)
    env.reset()

    # print("### agent type in env ###")
    # print(type(env.agents[0]))

    for i, agent in enumerate(env.agents):
        print(f"Agent {i} speed: {agent.speed_data}")
        # print(f"Agent {i} earliest_departure: {env.schedule.agents_schedule[i].earliest_departure}")
        # print(f"Agent {i}: earliest_departure = {agent.earliest_departure}")

    if env_renderer_enable:
        env_renderer = RenderTool(env, screen_height=env.height * 50,
                              screen_width=env.width*50,show_debug=False)
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

    steps = 0
    while True:
        action = get_or_actions(steps)

        print(f"{steps}: {action}")

        steps += 1

        observation, all_rewards, done, info = env.step(action)
        print(f"dones: {done}")
        for idx, agent in enumerate(env.agents):
            print(f"Agent {idx} position: {agent.position}, direction: {agent.direction}, target={agent.target}, init={agent.initial_position}")
            # print(f"Agent {idx} info: {agent.status.name}")

        if env_renderer_enable:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
            sleep(1 / fps)

        if done['__all__']:
            print("Finish episode.")
            break
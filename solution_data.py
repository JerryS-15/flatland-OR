#!/usr/bin/env python

# Compile codes in PythonCBS in folder CBS-corridor with cmake and import PythonCBS class
from libPythonCBS import PythonCBS

# Import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator,rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator,schedule_from_file
from flatland.envs.malfunction_generators  import malfunction_from_params, MalfunctionParameters
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
import flatland

import argparse
import pickle
import os
from tqdm import tqdm

from env_v2_generator_test import extract_agent_info, extract_station_info


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
        remove_agents_at_target=True,
        random_seed=seed
    )

def save_env_data(env_params, save_dir, env_seed):
    seed = env_seed
    save_path = os.path.join(save_dir, f"env_data_v2_{seed}.pkl")

    env = create_env(env_params, seed)
    env.reset()

    data = {
        "seed": seed,
        "rail": env.rail,
        "stations": extract_station_info(env),
        "agent_info": extract_agent_info(env),
        "env_params": env_params
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return env

# def get_or_solution():

if __name__ == "__main__":

    flatland_parameters = {
        # Flatland Env
        "number_of_agents": 2,
        "width": 30,
        "height": 35,
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "speed_ratio_map": {1.0: 1 / 4, 0.5: 1 / 4, 0.33: 1 / 4, 0.25: 1 / 4},
        # Flatland - malfunction
        "malfunction_rate": 1 / 4500,
        "min_duration": 20,
        "max_duration": 50
    }

    save_dir = "or_solution_data"
    os.makedirs(save_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default=False, type=bool, help="If render image for debug.")
    parser.add_argument("--seed", default=0, type=int, help="Initial seed for data collection.")
    parser.add_argument("--eps", default=100, type=int, help="Number of episodes to collect for dataset.")
    args = parser.parse_args()

    seed_init = args.seed
    n_eps = args.eps

    print("---------------------------------------")
    print(f"OR Solution Data Collection Started.")
    print(f"Initial seed: {seed_init}")
    print(f"Number of episodes: {args.eps}")
    print(f"Number of agents: {flatland_parameters['number_of_agents']}")
    print("---------------------------------------")

    for i in tqdm(range(0, n_eps), desc="Generate OR solutions"):
        seed = seed_init + i
        action_save_path = os.path.join(save_dir, f"action_data_v2_{seed}.pkl")
        env = save_env_data(flatland_parameters, save_dir, seed)

        framework = "LNS"  # "LNS" for large neighborhood search
        default_group_size = flatland_parameters['number_of_agents'] # max number of agents in a group.
        max_iterations = 1000
        stop_threshold = 10
        agent_priority_strategy = 3
        neighbor_generation_strategy = 3
        debug = False
        time_limit =200
        replan = True

        solver = PythonCBS(env, framework, time_limit, default_group_size, debug, replan,stop_threshold,agent_priority_strategy,neighbor_generation_strategy)
        solver.search(1.1, max_iterations)
        solver.buildMCP()

        steps=0
        actions = []
        while True:
        
            action = solver.getActions(env, steps, 3.0)
        
            # Debug
            print(f"{steps}: {action}")
            actions.append(action)

            observation, all_rewards, done, info = env.step(action)

            steps += 1
            if done['__all__']:
                solver.clearMCP()
                break
        
        with open(action_save_path, "wb") as f:
            pickle.dump(actions, f)
        
    print(f"✅ Flatland v{flatland.__version__} envs with action data saved in '{save_dir}/' folder.")
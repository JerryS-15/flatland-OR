import argparse
import hashlib
from tqdm import tqdm
import numpy as np
import pickle
import os
from time import sleep
from tqdm import tqdm

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
        # malfunction_generator_and_process_data=malfunction_from_params(
        #     MalfunctionParameters(
        #         malfunction_rate=env_params['malfunction_rate'],
        #         min_duration=env_params['min_duration'],
        #         max_duration=env_params['max_duration']
        #     )
        # ),
        malfunction_generator_and_process_data=None,
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

    # seed = 5
    # env_renderer_enable = True
    # fps = 30

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default=False, type=bool, help="If render image for debug.")
    parser.add_argument("--seed", default=1, type=int, help="Initial seed for data collection.") # seed=0 generate random env in v2.2.1
    parser.add_argument("--eps", default=100, type=int, help="Number of episodes to collect for dataset.")
    parser.add_argument("--n_agents", default=5, type=int, help="Number of agents for data collection.")
    args = parser.parse_args()

    seed_init = args.seed
    n_eps = args.eps
    n_agents = args.n_agents

    flatland_parameters = {
        # Flatland Env
        "number_of_agents": n_agents,
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

    print("---------------------------------------")
    print(f"OR Solution Step-wise Data Collection Started.")
    print(f"Number of episodes: {args.eps}")
    print(f"Number of agents: {flatland_parameters['number_of_agents']}")
    print("---------------------------------------")

    save_dir = f"or_solution_data_agent_{flatland_parameters['number_of_agents']}"

    for i in tqdm(range(0, n_eps), desc="Generate step-wise data"):
        seed = seed_init + i

        env_path = f"{save_dir}/env_data_v2_{seed}.pkl"
        action_path = f"{save_dir}/action_data_v2_{seed}.pkl"
        step_path = f"{save_dir}/step_data_v2_{seed}.pkl"

        env = create_env(flatland_parameters, seed)
        env.reset()

        steps = 0
        step_data = []
        while True:
            action = get_or_actions(steps)
            steps += 1
            observation, all_rewards, done, info = env.step(action)
            step_data_agents = []
            for idx, agent in enumerate(env.agents):
                agent_data = {
                    "handle": agent.handle,
                    "position": agent.position,
                    "direction": agent.direction,
                    "target": agent.target,
                    "initial_position": agent.initial_position,
                    "status": agent.status.name,  # 存为字符串
                    "speed": agent.speed_data["speed"],
                    "position_fraction": agent.speed_data["position_fraction"],
                    "malfunction": agent.malfunction_data["malfunction"],
                    "moving": agent.moving,
                }
                # print(f"Agent {idx} position: {agent.position}, direction: {agent.direction}, target={agent.target}, init={agent.initial_position}")
                # print(f"Agent {idx} with data {agent}")
                step_data_agents.append(agent_data)
        
            step_data.append(step_data_agents)

            if done['__all__']:
                # print("Finish episode.")
                break
    
        with open(step_path, "wb") as f:
            pickle.dump(step_data, f)
    print(f"✅ Flatland v{flatland.__version__} steps data with {n_eps} episodes saved in '{save_dir}/'.")
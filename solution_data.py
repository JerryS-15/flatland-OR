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
# import imageio

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

def save_env_data(env_params, save_dir, env_seed):
    seed = env_seed
    save_path = os.path.join(save_dir, f"env_data_v2_{seed}.pkl")

    env = create_env(env_params, seed)
    env.reset()

    # for i, agent in enumerate(env.agents):
    #         print(f"Agent {i}: earliest_departure = {agent.earliest_departure}")

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default=False, type=bool, help="If render image for debug.")
    parser.add_argument("--seed", default=1, type=int, help="Initial seed for data collection.") # seed=0 generate random env in v2.2.1
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

    save_dir = f"or_solution_data_agent_{flatland_parameters['number_of_agents']}"
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(0, n_eps), desc="Generate OR solutions"):
        seed = seed_init + i
        action_save_path = os.path.join(save_dir, f"action_data_v2_{seed}.pkl")

        env = save_env_data(flatland_parameters, save_dir, seed)

        # for i in range(env.get_num_agents()):
        #     earliest_departure = env.schedule.agents_schedule[i].earliest_departure
        #     print(f"Agent {i} earliest_departure: {earliest_departure}")

        # for i, agent in enumerate(env.agents):
        #     print(f"Agent {i}:")
        #     print(f"  Earliest Departure: {agent.earliest_departure}")
        #     print(f"  Latest Arrival: {agent.latest_arrival}")
        # print(f"Max episode steps: {env._max_episode_steps}")

        # render_tool = RenderTool(env, agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND, show_debug=False)
        # render_tool.reset()
        # frames = []
        # env.reset()
        # render_tool.render_env(show=False, frames=True, show_observations=False)
        # frames.append(render_tool.get_image())

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
            # print(f"{steps} actions: {action}")
            actions.append(action)

            observation, all_rewards, done, info = env.step(action)
            # Debug
            # print(f"{steps} obs: {observation}")
            # print(f"{steps} all_rewards: {all_rewards}")
            # print(f"{steps} done: {done}")

            # render_tool.render_env(show=False, frames=True, show_observations=False)
            # frames.append(render_tool.get_image())

            steps += 1
            if done['__all__']:
                print(f"dones: {done}")
                print("Current episode finished.")
                solver.clearMCP()
                break
        
        with open(action_save_path, "wb") as f:
            pickle.dump(actions, f)

        # video_path = os.path.join(save_dir, f"solution_render_{seed}.mp4")
        # imageio.mimsave(video_path, frames, fps=5)
        
    print(f"✅ Flatland v{flatland.__version__} envs with action data saved in '{save_dir}/' folder.")
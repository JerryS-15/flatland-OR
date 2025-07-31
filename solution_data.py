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
import time, glob
import argparse


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

if __name__ == "__main__":

    flatland_parameters = {
        # Flatland Env
        "number_of_agents": 5,
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default=False, type=bool, help="If render image for debug.")
    
    args = parser.parse_args()

    seed = 0

    env = create_env(flatland_parameters, seed)
    env.reset()

    #####################################################################
    # Initialize Mapf-solver
    #####################################################################
    framework = "LNS"  # "LNS" for large neighborhood search
    default_group_size = 5 # max number of agents in a group.
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

    #####################################################################
    # Show the flatland visualization, for debugging
    #####################################################################
    if args.render:
        env_renderer = RenderTool(env, screen_height=env.height * 50,
                                screen_width=env.width*50,show_debug=False)
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    
    steps=0
    while True:
        #####################################################################
        # Simulation main loop
        #####################################################################
        
        action = solver.getActions(env, steps, 3.0)
        
        # Debug
        print(f"{steps}: {action}")

        observation, all_rewards, done, info = env.step(action)
        
        if args.render:
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
            time.sleep(0.5)

        steps += 1
        if done['__all__']:
            solver.clearMCP()
            break
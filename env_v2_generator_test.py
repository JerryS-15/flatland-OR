import argparse
import hashlib
from tqdm import tqdm
import numpy as np
import pickle

import flatland
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator,rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator,schedule_from_file
from flatland.envs.malfunction_generators  import malfunction_from_params, MalfunctionParameters
from flatland.core.env_observation_builder import DummyObservationBuilder

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

def hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()


if __name__ == "__main__":

    init_seed = 42
    save_path = "env_v2.pkl"

    flatland_parameters = {
        # Flatland Env
        "number_of_agents": 10,
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

    all_episodes = []

    for i in tqdm(range(0, 10), desc="Generate v2.2.1 Env"):
        seed = init_seed + i

        env = create_env(flatland_parameters, seed)
        env.reset()

        rail_grid = env.rail.grid.astype(np.uint8)
        rail_hash = hash_array(rail_grid)

        agent_info = [(a.initial_position, a.target) for a in env.agents]
        agent_hash = hashlib.md5(pickle.dumps(agent_info)).hexdigest()

        all_episodes.append({
            "episode_id": i + 1,
            "seed": seed,
            "rail_hash": rail_hash,
            "agent_hash": agent_hash,
            "grid_shape": rail_grid.shape,
            # Optional storage of original data（for debug）
            # "rail_grid": rail_grid
            # "agent_info": agent_info,
        })

        tqdm.write(f"Episode {i+1}, Seed {seed} Stored.")
    
    with open(save_path, "wb") as f:
        pickle.dump(all_episodes, f)
    print(f"✅ Flatland v{flatland.__version__} test env data is saved at {save_path}.")
from metadrive import MultiAgentIntersectionEnv


class MultiAgentsEnv(MultiAgentIntersectionEnv):
    # State length: 19
    # Action length: 2
    # Image shape: (84, 84, 3)
    def __init__(self, config=None, num_agents=8):
        if config is None:
            config = {
                "allow_respawn": False,
                "delay_done": 100,
                "out_of_road_done": True,
                "crash_done": True,
                "num_agents": num_agents,
                "use_render": False,
                "manual_control": False,
                "map_config": {
                    "lane_num": 3
                },
                "vehicle_config": {
                    "show_lidar": False,
                    "image_source": "depth_camera",
                    "rgb_to_grayscale": False,
                    # "rgb_camera": (200, 88),
                    "depth_camera": (200, 88, False)
                },
                "show_fps": True,
                "image_observation": True,
            }
        super(MultiAgentsEnv, self).__init__(config)

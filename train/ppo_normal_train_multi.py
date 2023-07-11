import copy
from datetime import datetime
import os

from algos.multi.multi_ppo_clip_normal import MultiPPOClipNormalAgents
from envs.multi_agents_lidar import MultiAgentsLidarEnv

import json


def train():
    """
    initialize environment hyperparameters
    """
    env_name = "Multi-Intersection-PPO-Clip_Normal-v1"
    num_agents = 4
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(1e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 3  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)
    numbers_of_models_to_save = max_training_timesteps // save_model_freq
    update_timestep = max_ep_len * 2  # update policy every n timesteps

    """
    PPO Normal Hyperparameters
    """
    # update_timestep = 100  # update policy every n timesteps
    K_epochs = 5  # update policy for K epochs in one PPO update
    batch_size = 128  # minibatch size

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    """
    Policy Parameters
    """
    hidden_size = 256
    action_size = 2

    """
    Make Enviorment
    """
    env = MultiAgentsLidarEnv(num_agents=num_agents)

    state_dim = env.observation_space['agent0'].shape[0]

    action_dim = env.action_space['agent0'].shape[0]

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    """
    Dumping the config file into the log directory
    """
    train_config = {
        "env_name": env_name,
        "num_agents": num_agents,
        "max_ep_len": max_ep_len,
        "max_training_timesteps": max_training_timesteps,
        "print_freq": print_freq,
        "log_freq": log_freq,
        "save_model_freq": save_model_freq,
        "update_timestep": update_timestep,
        "numbers_of_models_to_save": numbers_of_models_to_save,
    }
    ppo_config = {
        "K_epochs": K_epochs,
        "batch_size": batch_size,
        "eps_clip": eps_clip,
        "gamma": gamma,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "action_std": action_std,
        "action_std_decay_rate": action_std_decay_rate,
        "min_action_std": min_action_std,
        "action_std_decay_freq": action_std_decay_freq,
    }
    policy_config = {
        "hidden_size": hidden_size,
        "action_size": action_size,
    }
    config = dict({
        "train_config": train_config,
        "ppo_config": ppo_config,
        "policy_config": policy_config,
    })
    # print("config : ", config)
    log_json_file = log_dir + '/config.json'
    with open(log_json_file, "w") as outfile:
        json.dump(config, outfile)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files) - 1
    print("current number of files : ", run_num)

    #### create new log file for each run
    log_f_name = log_dir + 'PPO_' + env_name + "_log_reward_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 1  #### change this to prevent overwriting weights in same env_name folder

    directory = "preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_iteration = 0
    checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, save_iteration)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("Number of models to save : " + str(numbers_of_models_to_save))
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    ####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = MultiPPOClipNormalAgents(num_agents=num_agents,
                                         batch_size=batch_size,
                                         state_dim=state_dim,
                                         lr_actor=lr_actor,
                                         lr_critic=lr_critic,
                                         gamma=gamma,
                                         K_epochs=K_epochs,
                                         eps_clip=eps_clip,
                                         hidden_dim=hidden_size,
                                         action_dim=action_dim,
                                         action_std_init=action_std,
                                         )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    success_rates = []

    # training loop
    while time_step <= max_training_timesteps:

        obs = env.reset()
        current_ep_reward = 0
        total_successes = 0
        d = {}
        d["__all__"] = False

        for t in range(1, max_ep_len + 1):
            actions = ppo_agent.select_actions(obs)

            obs, r, d, i = env.step(actions)
            # env.render(mode="top_down", film_size=(1000, 1000), screen_size=(1000, 1000))
            # Remove dead agent
            for key in d.keys():
                if d[key]:
                    obs.pop(key, None)
            # print("Actions : ", actions)
            # print("Reward : ", r)
            # print("Terminated : ", d)
            # print("State size", len(ppo_agent.buffer.states))
            # ppo_agent.buffer.add_reward(r, num_agents)
            # ppo_agent.buffer.add_terminated(d, num_agents)

            ppo_agent.rollout_buffer.rewards.append(r)
            ppo_agent.rollout_buffer.is_terminated.append(copy.deepcopy(d))
            # print("Reward Size : ", len(ppo_agent.buffer.rewards))
            # print("Terminated Size : ", len(ppo_agent.buffer.is_terminated))

            time_step += 1
            current_ep_reward += sum(r.values())
            # print("time_step : ", time_step)

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                # obs = env.reset()

            # if continuous action space; then decay action std of ouput action distribution
            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                save_iteration += 1
                checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, save_iteration)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if d["__all__"]:
                # print("Episodic Reward :", current_ep_reward)
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, "final")
    print("saving model at : " + checkpoint_path)
    ppo_agent.save(checkpoint_path)
    # print("Ziping the model")
    # zip_directory(directory, directory + "PPO_{}_{}.zip".format(env_name, end_time))
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()

import os
import pandas as pd
import matplotlib.pyplot as plt


def save_graph():
    print("============================================================================================")
    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'
    env_1 = 'Multi-Intersection-PPO-Clip_Beta-v1'
    env_2 = 'Multi-Intersection-PPO-Clip_Normal-v1'

    envs = [env_1, env_2]

    solving_threshold = 80

    fig_num = 1  #### change this to prevent overwriting figures in same env_name folder
    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson', 'gray',
              'black']

    # make directory for saving figures
    figures_dir = "figs"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + envs[0] + '_' + envs[1] + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_save_path = figures_dir + '/PPO_' + envs[0] + '_' + envs[1] + '_fig_' + str(fig_num) + '.png'

    ax = plt.gca()

    for i, env_name in enumerate(envs):
        short_env_name = 'PPO Beta (Ours)' if i == 0 else 'PPO Gaussian'
        # get number of log files in directory
        log_dir = "logs" + '/' + env_name + '/'
        current_num_files = next(os.walk(log_dir))[2]
        num_runs = len(current_num_files) - 1
        all_runs = []
        for run_num in range(num_runs):
            log_f_name = log_dir + '/PPO_' + env_name + "_log_reward_" + str(run_num) + ".csv"
            print("loading data from : " + log_f_name)
            data = pd.read_csv(log_f_name)
            data = pd.DataFrame(data)

            print("data shape : ", data.shape)

            all_runs.append(data)
            print("--------------------------------------------------------------------------------------------")

        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()
        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg[short_env_name] = data_avg['reward'].rolling(window=window_len_smooth,
                                                                                 win_type='triang',
                                                                                 min_periods=min_window_len_smooth).mean()
        data_avg['reward_var_' + short_env_name] = data_avg['reward'].rolling(window=window_len_var, win_type='triang',
                                                                              min_periods=min_window_len_var).mean()

        data_avg.plot(kind='line', x='timestep', y=short_env_name, ax=ax,
                      color=colors[i % len(colors)],
                      linewidth=linewidth_smooth,
                      alpha=alpha_smooth)
        data_avg.plot(kind='line', x='timestep', y='reward_var_' + short_env_name, ax=ax, color=colors[i % len(colors)],
                      linewidth=linewidth_var,
                      alpha=alpha_var)

        # # keep only reward_smooth in the legend and rename it
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend([handles[0]], ["reward_avg_" + str(env_name) + '_' + str(len(all_runs)) + "_runs"], loc=2)

    # keep alternate elements (reward_smooth_i) in the legend

    # ax.axhline(y=solving_threshold, color='black', linewidth=1, linestyle='--', label="Solving Threshold")
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Average Rewards", fontsize=14)

    # ax.legend(new_handles, new_labels, loc=4)
    handles, labels = ax.get_legend_handles_labels()
    print("handles : ", handles)
    print("labels : ", labels)
    new_handles = []
    new_labels = []
    for i in range(len(handles)):
        if (i % 2 == 0):
            new_handles.append(handles[i])
            new_labels.append(labels[i])
    ax.legend(new_handles, new_labels, loc=4)

    plt.title("Multi Agent Intersection-MLP-V1", fontsize=16)

    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)

    print("============================================================================================")
    plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)
    print("============================================================================================")

    plt.show()


if __name__ == '__main__':
    save_graph()

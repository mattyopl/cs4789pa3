import gymnasium as gym
import numpy as np
import npg_utils
import matplotlib.pyplot as plt
import os

def sample(theta, env, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []
    for n in range(N):
        trajectory_grads = []
        trajectory_rewards = []
        
        # TODO Get initial state
        observation = env.reset(seed=np.random.randint(2**31))[0]
        for t in range(200):
            # TODO Extract features, get trajectory_grads and get trajectory_rewards
            phis = npg_utils.extract_features(observation, 2)
            action_probs = npg_utils.compute_action_distribution(theta, phis).flatten()
            
            action = np.random.choice(2, p=action_probs)
            grad = npg_utils.compute_log_softmax_grad(theta, phis, action)


            observation, reward, terminated, truncated, info = env.step(action)

            trajectory_rewards.append(reward)
            trajectory_grads.append(grad)
            if(terminated):
                break

        total_rewards.append(trajectory_rewards)
        total_grads.append(trajectory_grads)


    return total_grads, total_rewards


def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100,1)
    env = gym.make('CartPole-v0')

    episode_rewards = []

    for t in range(T):
        print(f"Iteration {t}")
        # TODO Update theta according to handout, and record rewards
        grads, rewards = sample(theta, env, N) # sample trajs
        fisher = npg_utils.compute_fisher_matrix(grads, lamb) # get the fisher
        v_grad = npg_utils.compute_value_gradient(grads, rewards) # get the v
        eta = npg_utils.compute_eta(delta, fisher, v_grad) #compute eta

        # Natural Gradient Step: theta = theta + eta * F^(-1) * deltaV
        try:
            nat_grad = np.linalg.solve(fisher, v_grad)  # more stable than inv
        except np.linalg.LinAlgError:
            nat_grad = np.zeros_like(theta)

        theta += eta * nat_grad

        # Record average reward for this iteration
        avg_reward = np.mean([sum(r) for r in rewards])
        episode_rewards.append(avg_reward)
        print(f"  Avg reward: {avg_reward:.2f} | Step size: {eta:.5f}")

    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    theta_dir = 'learned_policies/NPG'
    os.makedirs(theta_dir, exist_ok=True)
    np.save(os.path.join(theta_dir, 'expert_theta.npy'), theta)

    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plot_dir = './plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "rewards"))
    plt.show()

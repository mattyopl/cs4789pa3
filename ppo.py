from argparse import ArgumentParser
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import ActorCritic, ContinuousActorCritic
from tqdm import tqdm
import pathlib


def compute_gae_returns(rewards, values, dones, gamma, gae_lambda):
    """
    Returns the advantages computed via GAE and the discounted returns. 

    Instead of using the Monte Carlo estimates for the returns,
    use the computed advantages and the value function
    to compute an estimate for the returns. 
    
    Hint: How can you easily do this if lambda = 1?

    :param rewards: The reward at each state-action pair
    :param values: The value estimate at the state
    :param dones: Whether the state is terminal/truncated
    :param gamma: Discount factor
    :param gae_lambda: lambda coef for GAE
    """       
    T = len(rewards)
    advantages = torch.zeros(T, dtype=values.dtype, device=values.device)
    gae = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns




def ppo_loss(agent: ActorCritic, states, actions, advantages, logprobs, returns, clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5) -> torch.Tensor:
    """
    Compute the PPO loss. You can combine the policy, value and entropy losses into a single value. 

    :param policy: The policy network
    :param states: States batch
    :param actions: Actions batch
    :param advantages: Advantages batch
    :param logprobs: Log probability of actions
    :param returns: Returns at each state-action pair
    :param clip_ratio: Clipping term for PG loss
    :param ent_coef: Entropy coef for entropy loss
    :param vf_coef: Value coef for value loss
    """  
    _, new_logprob, entropy, values = agent.action_value(states, actions)


    prob_ratio = torch.exp(new_logprob - logprobs)

    # Clipped surrogate objective
    clipped_ratio = torch.clamp(prob_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    # 1. Policy Loss
    policy_loss = -torch.min(prob_ratio * advantages, clipped_ratio * advantages).mean()

    # 2. Value function loss (squared error)
    value_loss = nn.MSELoss()(values, returns)

    # 3. Entropy Bonus (mean to encourage exploration)
    entropy_loss = entropy.mean()

    # Final combined loss
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

    return loss

def make_env(env_id, **kwargs):
    def env_fn():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        return env
    return env_fn


def train(
    env_id="CartPole-v0",
    epochs=500,
    num_envs=4,
    gamma=0.99,
    gae_lambda=0.9,
    lr=3e-4,
    num_steps=128,
    minibatch_size=32,
    clip_ratio=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    update_epochs=3,
    seed=42,
    checkpoint=False,
    max_grad_norm=0.5,
):
    """
    Returns trained policy. 
    """

    # Try not to modify this
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.vector.SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
    eval_env = make_env(env_id)()
    
    if isinstance(env.single_action_space, gym.spaces.Discrete):
        policy = ActorCritic(env.single_observation_space.shape[0], env.single_action_space.n).to(device)
    else:
        policy = ContinuousActorCritic(env.single_observation_space.shape[0], env.single_action_space.shape[0])
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    states = torch.zeros((num_steps, num_envs) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + env.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps + 1, num_envs)).to(device)

    obs, _ = env.reset(seed = np.random.randint(2**30))
    obs = torch.from_numpy(obs).float().to(device)

    pathlib.Path(f"learned_policies/{env_id}/").mkdir(parents=True, exist_ok=True)

    obs_shape = env.single_observation_space.shape
    act_shape = env.single_action_space.shape if not isinstance(env.single_action_space, gym.spaces.Discrete) else ()


    for iteration in tqdm(range(1, epochs + 1)):
        # TODO: Collect num_steps transitions from env and fill in the tensors for states, actions, ....
        for t in range(num_steps): #iterating over the num_steps
            with torch.no_grad():
                action, logprob, _, value = policy.action_value(obs)
            # Adding the transition info
            states[t] = obs
            actions[t] = action
            logprobs[t] = logprob
            values[t] = value

            obs_np = action.cpu().numpy() # moving to cpu to take the step
            obs_next, reward, done, _, _ = env.step(obs_np)

            obs = torch.from_numpy(obs_next).float().to(device) #moving back to gpu for calcs
            rewards[t] = torch.tensor(reward, device=device)
            dones[t] = torch.tensor(done, device=device)


        # Hint: The last state collected may not always be terminal. How can you get an estimate of the return from that state?

        # TODO: Compute Advantages and Returns

        with torch.no_grad():
            values[-1] = policy.value(obs)

        advantages, returns = compute_gae_returns(rewards, values, dones, gamma, gae_lambda)

        # Flatten everything
        batch_states = states.reshape(-1, *obs_shape)
        batch_actions = actions.reshape(-1, *act_shape)
        batch_logprobs = logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)


        # TODO: Perform num_steps / minibatch_size gradient updates per update_epoch
        inds = np.arange(num_steps * num_envs)

        for _ in range(update_epochs): # Perform multiple epochs of policy optimization on minibatches of the collected data
            np.random.shuffle(inds) # Shuffle the collected data to break correlations
            for start in range(0, len(inds), minibatch_size): # Divide the data into minibatches of size minibatch size
                end = start + minibatch_size
                mb_inds = inds[start:end]

                # compute the PPO loss and update the policy using gradient descent
                loss = ppo_loss(
                    policy,
                    batch_states[mb_inds],
                    batch_actions[mb_inds],
                    batch_advantages[mb_inds],
                    batch_logprobs[mb_inds],
                    batch_returns[mb_inds],
                    clip_ratio,
                    ent_coef,
                    vf_coef,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Clip the gradient
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        # Uncomment for eval/checkpoint
        # if iteration % 10 == 0: 
        #     print(f"Eval Reward {iteration}:", (val(policy, eval_env)))
        #     if checkpoint:
        #         torch.save(policy, f"learned_policies/{env_id}/model_{iteration}.pt")
        
    
    return policy


def val(model, env, num_ep=100):
    rew = 0
    for i in range(num_ep):
        done = False
        obs, _ = env.reset(seed=np.random.randint(2**30))
        obs = torch.from_numpy(obs).float()
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = model.action_value(obs)
            obs, reward, done, trunc, _ = env.step(action.cpu().numpy())
            obs = torch.from_numpy(obs).float()
            done |= trunc
            rew += reward

    return rew / num_ep

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=3, help="Number of epochs over data every iteration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    # Feel free to add or remove parameters

    args = parser.parse_args()  
    policy = (train(**vars(args)))
    torch.save(policy, f"learned_policies/{args.env_id}/model.pt")
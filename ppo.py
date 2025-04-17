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
    raise NotImplementedError()

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
    
    raise NotImplementedError()

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
    
    for iteration in tqdm(range(1, epochs + 1)):
        raise NotImplementedError()
        # TODO: Collect num_steps transitions from env and fill in the tensors for states, actions, ....
        
        # Hint: The last state collected may not always be terminal. How can you get an estimate of the return from that state?

        # TODO: Compute Advantages and Returns

        # TODO: Perform num_steps / minibatch_size gradient updates per update_epoch

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
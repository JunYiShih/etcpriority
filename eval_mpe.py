import torch
import gym  # or gymnasium
import imageio
from custom_envs.mpe import simple_spread_pri
from custom_envs.mpe import simple_spread_rp
from custom_envs.mpe import simple_circle_rp
from custom_envs.mpe import simple_circle_pri
from algorithms.mappo.algorithms.r_mappo.algorithm.r_actor_critic_dpo_pri import R_Actor
from algorithms.mappo.config import get_config
import sys
import numpy as np
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='Multiwalker', help="Which scenario to run on")
    # parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=8, help="number of players")


    all_args = parser.parse_known_args(args)[0]

    return all_args
def dict_to_tensor(d):
  d = list(d.values())
  d = np.array(d)
  d = torch.tensor(d)
  return d
def preprocess_obs(obs):
  obs = dict_to_tensor(obs)
  return obs
def get_logits(obs, policies, rnn_state):
  logits = []
  new_rnn_state = []
  for i, p in enumerate(policies):
    o = obs[i].unsqueeze(0)
    l, _, r = p(o, rnn_state[i], torch.tensor(0))
    logits.append(l.squeeze(0))
    new_rnn_state.append(r.squeeze(0))
  
  logits = torch.stack(logits)
  new_rnn_state = torch.stack(new_rnn_state)
  return logits, new_rnn_state
def get_actions(obs, env, rnn_state, policies, training=False):
    actions = {}
    logits, rnn_state = get_logits(obs, policies, rnn_state)
    logits = logits.cpu().numpy()
    logits = np.clip(logits, -1, 1)
    for i, agent in enumerate(env.possible_agents):
        actions[agent] = logits[i]

    return actions, logits, rnn_state
args = sys.argv[1:]
parser = get_config()
all_args = parse_args(args, parser)
all_args.use_recurrent_policy = True
all_args.use_naive_recurrent_policy = False
all_args.use_centralized_V = False
if all_args.cuda and torch.cuda.is_available():
    print("choose to use gpu...")
    device = torch.device("cuda:0")
    torch.set_num_threads(all_args.n_training_threads)
    if all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
else:
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)
n_agents = 8
env = simple_circle_rp.parallel_env(N=n_agents, penalty_ratio=all_args.com_ratio,max_cycles=40,
                    full_comm=False, local_ratio=all_args.local_ratio, continuous_actions=True,
                    delay = 1,packet_drop_prob=0.2,
                    bandwidth_limit = 10,landmarks = 1,render_mode = 'rgb_array')
all_args.actor_hidden_size = 256
all_args.critic_hidden_size = 256
actor1 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)
actor2 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)
actor3 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)
actor4 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)
actor5 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)
actor6 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)
actor7 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)
actor8 = R_Actor(all_args, env.observation_space('agent_0'), env.action_space('agent_0'), device)


frames = []
model_dir = os.path.join(os.path.dirname(__file__), 'model_formation_r_8')

actor1.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent0.pt')))
actor1.eval()

actor2.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent1.pt')))
actor2.eval()

actor3.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent2.pt')))
actor3.eval()
actor4.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent3.pt')))
actor4.eval()

actor5.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent4.pt')))
actor5.eval()

actor6.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent5.pt')))
actor6.eval()
actor7.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent6.pt')))
actor7.eval()

actor8.load_state_dict(torch.load(os.path.join(model_dir, 'actor_agent7.pt')))
actor8.eval()



policies = []
policies.append(actor1)
policies.append(actor2)
policies.append(actor3)
policies.append(actor4)
policies.append(actor5)
policies.append(actor6)
policies.append(actor7)
policies.append(actor8)


obs = env.reset(50000)
seed = 2
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)
obs = preprocess_obs(obs)
recurrent_N = 1

rnn_state = np.zeros((n_agents, recurrent_N, all_args.actor_hidden_size), dtype=np.float32)

for _ in range(500):  # Adjust the number of steps if needed
    # Render the environment and store the frame
    while env.agents:
      actions, logits, rnn_state = get_actions(obs, env, rnn_state, policies, False)

      next_obs, rewards, dones, truncations, infos = env.step(actions)
      next_obs = preprocess_obs(next_obs)
      obs = next_obs
      rewards = dict_to_tensor(rewards)
      rewards_numpy = rewards.cpu().numpy() if rewards.is_cuda else rewards.numpy()
      print(rewards_numpy)
      seed_reward = rewards.squeeze().numpy()

      image = env.render()
      frames.append(image)
      
writer = imageio.get_writer('render.mp4', fps=10)

for frame in frames:
    writer.append_data(frame)
writer.close()
env.close()

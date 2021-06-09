import jax as jx
import jax.numpy as jnp
from jax import grad, jit, vmap

import haiku as hk

import json

import argparse

from tqdm import tqdm

import optimizers

from minatar import Environment

class AC_network(hk.Module):
    def __init__(self, num_actions, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions

    def __call__(self, s):
        phi = hk.Sequential([
                        hk.Conv2D(16, 3, padding='VALID'),
                        jx.nn.silu,
                        hk.Flatten(),
                        hk.Linear(128),
                        jx.nn.silu
                    ])
        pi_layer = hk.Linear(self.num_actions)
        V_layer = hk.Linear(1)
        phi = phi(s)
        return pi_layer(phi), V_layer(phi)

def AC_loss(last_states, actions, rewards, states, terminals, beta, gamma, num_actors, num_actions):
    network = AC_network(num_actions)

    pi_logit_curr, V_curr = network(states)
    pi_logit_last, V_last = network(last_states)

    entropy = -jnp.sum(jnp.log(pi_logit_last)*pi_logit_last, axis=1)
    loss = jnp.mean(0.5*(gamma*jnp.where(terminals,0,jx.lax.stop_gradient(V_curr))+rewards-V_last)**2-\
           0.5*(jx.lax.stop_gradient(gamma*jnp.where(terminals,0,V_curr)+rewards-V_last)*jx.nn.log_softmax(pi_logit_last)[jnp.arange(num_actors),actions])-\
           beta*entropy)
    return loss

# def AC_loss(last_states, actions, rewards, states, terminals, beta, gamma, num_actors, num_actions):
#     phi = hk.Sequential([
#         hk.Conv2D(16, 3, padding='VALID'),
#         jx.nn.silu,
#         hk.Flatten(),
#         hk.Linear(128),
#         jx.nn.silu
#     ])

#     pi_layer = hk.Linear(num_actions)
#     V_layer = hk.Linear(1)

#     V_curr = V_layer(phi(states))
#     pi_last = pi_layer(phi(last_states))
#     V_last = V_layer(phi(last_states))

#     entropy = -jnp.sum(jnp.log(pi_last)*pi_last, axis=1)
#     loss = jnp.mean(0.5*(gamma*jnp.where(terminals,0,jx.lax.stop_gradient(V_curr))+rewards-V_last)**2-\
#            0.5*(jx.lax.stop_gradient(gamma*jnp.where(terminals,0,V_curr)+rewards-V_last)*jnp.log(pi_last[jnp.arange(num_actors),actions]))-\
#            beta*entropy)
#     return loss
# AC_loss = jit(AC_loss, static_argnames=('num_actors', 'num_actions'))

def sample_actions(states, key, num_actions):
    network = AC_network(num_actions)
    pi_logit, _ = network(states)
    output = jx.random.categorical(key, pi_logit)
    return output

# def sample_action(logit, key):
#     output = jx.random.categorical(key, logit)
#     return output
# sample_action = jit(sample_action)

class AC_agent():
    def __init__(self,
                 key,
                 in_channels,
                 num_actions,
                 num_actors,
                 alpha,
                 beta,
                 gamma,
                 gamma_rms, 
                 epsilon_rms):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.num_actors = num_actors
        self.num_actions = num_actions

        self.optimizer = optimizers.rmsprop_optimizer(alpha,gamma_rms,epsilon_rms)

        self.loss = hk.without_apply_rng(hk.transform(AC_loss))
        self.sample = hk.without_apply_rng(hk.transform(sample_actions))
        dummy_last_states = jnp.zeros((num_actors,10,10,in_channels))
        dummy_actions = jnp.zeros(num_actors, dtype=int)
        dummy_rewards = jnp.zeros(num_actors)
        dummy_terminals = jnp.zeros(num_actors, dtype=bool)
        dummy_states = jnp.zeros((num_actors,10,10,in_channels))

        self.key, subkey = jx.random.split(key)
        self.params = self.loss.init(subkey, dummy_last_states, dummy_actions, dummy_rewards, dummy_states, dummy_terminals, self.beta, self.gamma, num_actors, num_actions)

    def act(self, states):
        states = jnp.stack(states)
        self.key, subkey = jx.random.split(self.key)
        return self.sample.apply(self.params, states, subkey, self.num_actions)


    def update(self, states, actions, rewards, next_states, terminals):
        states = jnp.stack(states)
        rewards = jnp.stack(rewards)
        next_states = jnp.stack(next_states)
        terminals = jnp.stack(terminals)
        grads = grad(self.loss.apply)(self.params, states, actions, rewards, next_states, terminals, self.beta, self.gamma, self.num_actors, self.num_actions)
        self.params = self.optimizer(self.params, grads)
        # self.params = jx.tree_multimap(self.optimizer, self.params, grads) 
         

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="A2C.out")
parser.add_argument("--model", "-m", type=str, default="A2C.model")
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config = json.load(f)

num_actors = config["num_actors"]
alpha = config["alpha"]
beta = config["beta"]
gamma = config["gamma"]
gamma_rms = config["gamma_rms"]
epsilon_rms = config["epsilon_rms"]
game = config["game"]
num_frames = config["num_frames"]

envs = [Environment(game) for i in range(num_actors)]

valid_actions = envs[0].minimal_action_set()
key, subkey = jx.random.split(key)
agent = AC_agent(subkey,
                 envs[0].state_shape()[2],
                 len(valid_actions),
                 num_actors,
                 alpha,
                 beta,
                 gamma,
                 gamma_rms,
                 epsilon_rms)

states = [env.state().astype(float) for env in envs]
last_states = None
terminals = [False]*num_actors
rewards = [0]*num_actors
t=0
while t < num_frames:
    actions = [valid_actions[a] for a in agent.act(states)]
    rewards = []
    new_terminals = []
    for env, term, action in zip(envs, terminals, actions):
        if(not term):
            r, term = env.act(action)
        else:
            r = 0
            term = False
            env.reset()
        rewards+=[r]
        new_terminals+=[term]
    terminals = new_terminals
    last_states = states
    states = [env.state().astype(float) for env in envs]
    #TODO: Ensure state is not used following termination
    agent.update(last_states, actions, rewards, states, terminals)
    t += 1
    

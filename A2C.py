import jax as jx
import jax.numpy as jnp
from jax import grad, jit, vmap

import haiku as hk

import json

import argparse

import time

# import optimizers
from jax.experimental import optimizers

from minatar import Environment

class AC_network(hk.Module):
    def __init__(self, num_actions, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions

    def __call__(self, s):
        phi = hk.Sequential([
                        hk.Conv2D(16, 3, padding='VALID'),
                        jx.nn.relu,
                        hk.Flatten(),
                        hk.Linear(128),
                        jx.nn.relu
                    ])
        pi_layer = hk.Linear(self.num_actions)
        V_layer = hk.Linear(1)
        phi = phi(s)
        return pi_layer(phi), V_layer(phi)[:,0]

def AC_loss(params, network, last_states, actions, rewards, curr_states, terminals, beta, gamma, num_actors, num_actions):
    pi_logit_curr, V_curr = network(params,curr_states)
    pi_logit_last, V_last = network(params,last_states)

    critic_loss = jnp.mean(0.5*(gamma*jx.lax.stop_gradient(jnp.where(terminals,0,V_curr))+rewards-V_last)**2)
    entropy = -jnp.sum(jx.nn.log_softmax(pi_logit_last)*jx.nn.softmax(pi_logit_last), axis=1)
    actor_loss = jnp.mean(-0.5*(jx.lax.stop_gradient(gamma*jnp.where(terminals,0,V_curr)+rewards-V_last)*jx.nn.log_softmax(pi_logit_last)[jnp.arange(num_actors),actions])-beta*entropy)
    loss = critic_loss+actor_loss
    return loss

def sample_actions(params, network, states, key, num_actions):
    pi_logit, _ = network(params,states)
    output = jx.random.categorical(key, pi_logit)
    return output

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
        self.t = 0

        opt_init, self.opt_update, self.get_params = optimizers.rmsprop(alpha,gamma_rms,epsilon_rms)

        network = hk.without_apply_rng(hk.transform(lambda s: AC_network(num_actions)(s)))
        self.net_apply = network.apply
        self.key, subkey = jx.random.split(key)
        dummy_states = jnp.zeros((num_actors,10,10,in_channels))
        params = network.init(subkey,dummy_states)

        self.opt_state = opt_init(params)

        self.sample = jit(sample_actions, static_argnums=(1,9,10))
        self.opt_update = jit(self.opt_update)
        self.loss_grad = jit(grad(AC_loss), static_argnums=(1,9,10))

    def act(self, states):
        states = jnp.stack(states)
        self.key, subkey = jx.random.split(self.key)
        return self.sample(self.params(), self.net_apply, states, subkey, self.num_actions)

    def params(self):
        return self.get_params(self.opt_state)

    def update(self, last_states, actions, rewards, curr_states, terminals):
        last_states = jnp.stack(last_states)
        rewards = jnp.stack(rewards)
        curr_states = jnp.stack(curr_states)
        terminals = jnp.stack(terminals)
        grads = self.loss_grad(self.params(), self.net_apply, last_states, actions, rewards, curr_states, terminals, self.beta, self.gamma, self.num_actors, self.num_actions)
        self.opt_state = self.opt_update(self.t, grads, self.opt_state)
        self.t += 1
         

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
returns = []
curr_returns = [0.0]*num_actors
avg_return = 0.0
termination_times = []
t=0
t_start = time.time()
while t < num_frames:
    actions = agent.act(states)
    rewards = []
    new_terminals = []
    for i, (env, term, action) in enumerate(zip(envs, terminals, actions)):
        #continue until terminated, when terminated reset env and begin new episode immediately
        if(not term):
            #translate agent actions to world actions
            r, term = env.act(valid_actions[action])
            curr_returns[i] += r
        else:
            r = 0
            term = False
            termination_times += [t]
            returns += [curr_returns[i]]
            avg_return = 0.99 * avg_return + 0.01 * curr_returns[i]
            curr_returns[i] = 0
            env.reset()
        rewards += [r]
        new_terminals += [term]
    terminals = new_terminals
    last_states = states
    states = [env.state().astype(float) for env in envs]
    #TODO: Ensure state is not used following termination
    agent.update(last_states, actions, rewards, states, terminals)
    t += 1
    #print logging info periodically
    if(t%100==0):
        print("Avg return: " +str(jnp.around(avg_return, 2))+" | Frame: "+str(t)+" | Time per frame: " +
                         str((time.time()-t_start)/t))
    

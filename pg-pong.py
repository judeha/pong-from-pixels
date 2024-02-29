# %%
import numpy as np
import pickle
import gym
import gc
from torch.utils.tensorboard import SummaryWriter

gc.enable()
gc.collect()

# %%
# HELPER FUNCTIONS

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype('float').ravel()

# %%
# MODEL INITIALIZATION

# Hyperparams
H = 200 # n hidden layer neurons
batch_size = 10 # n episodes before param update
learning_rate = 1e-5
gamma = 0.99 # discount factor
b1 = 0.9
b2 = 0.999
epsilon = 1e-7
resume = True # resume from previous checkpoint?
path = 'save_7500'
render = False

# Model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open(f'checkpoints/{path}.p','rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization, centers around 0
    model['W2'] = np.random.randn(H) / np.sqrt(H)

# Tensorboard logging
# Writer outputs to ./runs/
writer = SummaryWriter('exp4')
def log_reward(episode,r):
  writer.add_scalar('reward:',r,episode)

# Tracks gradients, 1st moment, and 2nd moment over a batch
gradient_buffer = { k : np.zeros_like(v) for k,v in model.items() }
m = { k : np.zeros_like(v) for k,v in model.items() }
c = { k : np.zeros_like(v) for k,v in model.items() }


# %%
# PG FUNCTIONS

def discount_rewards(r):
  """ take 1D float array of rewards over episode and compute discounted reward """
  discounted_r = np.zeros_like(r)
  G = 0

  # Working backwards from the terminal state
  for t in reversed(range(0, r.size)):
    # (Loosely) update value at each state
    G = gamma * G + r[t]
    discounted_r[t] = G

  return discounted_r

def policy_forward(x):
  """ given frame, return probability of action 2 """
  # Layer 1
  h = np.dot(model['W1'], x)
  # ReLU
  h[h<0] = 0
  # Layer 2: get logits
  logp = np.dot(model['W2'], h)
  # Sigmoid
  p = sigmoid(logp)

  return p, h # Return probability of action 2, hidden state

def policy_backward(h, x, pgrad):
  """
  backward pass that gets policy gradients
  
  h: hidden states
  x: observed states
  pgrad: policy gradients

  """
  # Second layer gradients
  dW2 = np.dot(h.T, pgrad).ravel()
  dh = np.outer(pgrad, model['W2'])
  dh[h <= 0] = 0

  # First layer gradients
  dW1 = np.dot(dh.T, x)
  return {'W1':dW1, 'W2':dW2}

# %%
# INITIALIZE GYM

env = gym.make("Pong-v4")
observation, info = env.reset()

prev_x = None # Used for differencing
xs,hs,pgrads,rs = [],[],[],[] # Observed, hidden, policy gradient, rewards
xsp,hsp,pgradsp,rsp = [],[],[],[] # TODO
running_reward = 8
reward_sum = 0
episode_number = 7501

# %%
while True:
  if render: env.render()

  # Preprocess and difference observation
  current_x = prepro(observation)
  x = current_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = current_x

  # Feed forward policy network
  aprob, h = policy_forward(x)
  # Sample action from returned probability
  action = 2 if np.random.uniform() < aprob else 3

  # Record intermediates
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # "fake label"
  pgrads.append(y - aprob) # policy gradient

  # Take a step
  observation, reward, terminated, truncated, info = env.step(action)
  # Record rewards
  reward_sum += reward
  rs.append(reward)

  # If episode terminated
  if terminated or truncated:
    episode_number += 1

    # Stack intermediaries
    ixs = np.vstack(xs)
    ihs = np.vstack(hs)
    ipgrads = np.vstack(pgrads)
    irs = np.vstack(rs)
    xs,hs,pgrads,rs = [],[],[],[]

    # Get discounted rewards
    discounted_r = discount_rewards(irs)
    # Normalize rewards
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)

    # Calculate gradients (using Advantage and Policy Gradients)
    ipgrads *= discounted_r
    grad = policy_backward(ihs, ixs, ipgrads)

    # Accumulate gradients over batch
    for k in model:
      gradient_buffer[k] += grad[k]

    # At end of batch: update model
    if episode_number % batch_size == 0:
      # For each layer of weights
      for k,v in model.items():
        # Get summed gradient
        g = gradient_buffer[k]

        # ADAM optimizer
        m[k] = b1 * m[k] + (1 - b1) * g
        c[k] = b2 * c[k] + (1 - b2) * g**2

        m_hat = m[k] / (1 - b1**episode_number)
        c_hat = c[k] / (1 - b2**episode_number)

        # Update model
        model[k] += learning_rate * m_hat / (np.sqrt(c_hat) + epsilon)

        # Reset gradient buffer
        gradient_buffer[k] = np.zeros_like(v)

    # Book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %.2f. running mean: %.2f' % (reward_sum, running_reward))
    # Log reward
    log_reward(episode_number, running_reward)
    # Save model at checkpoints
    if episode_number % 500 == 0:
      pickle.dump(model, open(f'checkpoints/save_{episode_number}.p', 'wb'))

    # Reset episode after temination
    reward_sum = 0
    prev_x = None
    observation, info = env.reset()



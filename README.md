# I try gradient descent on pong-from-pixels and it does not go well
Refactoring of "Deep Reinforcement Learning: Pong from Pixels" from Andrej Karpathy with gradient descent comparisons

## motivation
bored! bored!!

## Overview of gradient descent algorithms
I liked this [blog post](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam) by Lili Jiang, which visually explains the different version of gradient optimization. TLDR:

- Vanilla gradient descent gets caught in local minima -> Momentum uses historical gradients to accelerate the algorithm and give it enough "momentum" to keep it moving in a general direction.
- AdaGrad takes momentum and squares it, creating the second moment. This allows it to focus on features that have been updated less. It computes an "adaptive" gradient for each feature, making it better for sparse features.
- RMSProp speeds up AdaGrad by decaying the squared momentum and preventing it from blowing up.
- Adam is generally thought of as combining RMSProp (second moment) with momentum (first moment) and generally converges faster and works with sparse features.

### 1. Vanilla gradient descent
```math
\Delta = -\gamma * \text{gradient}
```

Reminder: the gradient is computed in backprop, where $`L=y-\hat{y}`$. We take the derivative of loss respective to $`\hat{y}`$, which is a series of functions $`f_1(f_2(...(x)))`$, where each $`f`$ is a layer of the neural network. Rather than calculating the gradient analytically, we calculate it empirically (with the actual values of the weights). A gradient of 4 for $`w`$ tells us that increasing the weight will increase the loss by a rate of 4 in the current formulation.

Intuition: The blog explains normal gradient descent “like using a flashlight to find your way in the dark.” At any point in time, we can only see the few feet ahead of us that is illuminated by the flashlight, which I found to be a very helpful analogy.

### 2. Momentum
```math
M_t = \text{sum of gradient}_t = \text{gradient} + \rho * \text{sum of gradient}_{t-1}
\Delta = -\gamma * M
\Delta = -\gamma * \text{gradient} + \rho * \Delta_{t-1}
```

Intuition: This is very similar to vanilla gradient descent, except now my $`\Delta`$ accounts for the current gradient as well as the previous gradients, weighted by $`\rho`$. The addition of past gradients acts as "momentum," almost like a sort of memory that tells me what general direction I've been heading in. $`\rho`$ tells us how much to care about past vs present gradients, acting as "friction" that eventually slows the momentum effect of previous timesteps. If $`\rho=0`$, then I only care about the current gradient and it ends up being the same as vanilla descent. If $`\rho=1`$, then I may end up with too much minimum and overshoot my goal. Ideally, a good $`\rho`$ will let you escape local minima without overshooting the global minima you're aiming for.

### 3. AdaGrad
```math
M'_t = \text{sum of gradient squared}_t = \text{gradient}^2 + \text{sum of gradient}_{t-1}
\Delta = -\gamma * \frac{\text{gradient}}{\sqrt{M'}}
```

Intuition: AdaGrad takes the idea of momentum and then squares the sum of past gradients. Why square? Squaring forces the momentum to grow faster, which indicates which features have been updated most in the past. The idea is that dividing by squared momentum will dis-incentivize updating features that have already gone through lots of updates (and thus have high accumulated momentum) and instead focus on features that need more updating. In other words, Ada updates features at different rates, which can help avoid saddle points.

### 4. RMSProp
```math
C_t = \text{sum of gradient squared}_t = (1 - \rho) * \text{gradient} + \rho * \text{sum of gradient squared}
\Delta_t = -\gamma * \frac{\text{gradient}}{\sqrt{C_t}}
```

Intuition: AdaGrad is slow because the squared terms blow up fast and increase computation time. RMSProp is basically AdaGrad, but faster, because it keeps the squared term using the $`\rho`$ decay rate. The smaller $`\rho`$ is (usually between 0 and 1), the faster, although you trade off the momentum effect. The original pong from pixels code uses RMSProp, which uses a cache to store previous second moment/sum of gradient squared information.

<p align="center">
  <img src="assets/images/exp1.png" width="350" title="Experiment 1">
  <!-- <img src="your_relative_path_here_number_2_large_name" width="350" alt="accessibility text"> -->
</p>

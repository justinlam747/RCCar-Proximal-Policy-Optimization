# complete project walkthrough

this document explains every part of the project, why each design choice was made, what the tradeoffs are, and how everything connects together.

---

## table of contents

1. [what this project does](#what-this-project-does)
2. [the big picture: how rl training works](#the-big-picture-how-rl-training-works)
3. [file-by-file breakdown](#file-by-file-breakdown)
4. [the environment in detail](#the-environment-in-detail)
5. [the track system](#the-track-system)
6. [the car model](#the-car-model)
7. [the observation space](#the-observation-space)
8. [the action space](#the-action-space)
9. [the reward function and its 8 iterations](#the-reward-function-and-its-8-iterations)
10. [the training pipeline](#the-training-pipeline)
11. [ppo algorithm explained](#ppo-algorithm-explained)
12. [hyperparameter choices and tradeoffs](#hyperparameter-choices-and-tradeoffs)
13. [evaluation system](#evaluation-system)
14. [physics simulation choices](#physics-simulation-choices)
15. [interesting design patterns](#interesting-design-patterns)
16. [known limitations and future improvements](#known-limitations-and-future-improvements)

---

## what this project does

this is a reinforcement learning project that trains a virtual car to drive itself around a racetrack. the car has no hardcoded rules about how to drive - it learns entirely from trial and error by receiving rewards for good behavior and penalties for bad behavior.

the core loop is:
1. the car observes the world through simulated sensors (lidar + imu)
2. a neural network decides what to do (accelerate, brake, turn)
3. the physics engine simulates what happens
4. a reward function scores the result
5. the neural network updates its weights to get better rewards
6. repeat millions of times

by the end of training, the neural network has learned a driving policy - a mapping from sensor readings to steering commands that keeps the car on the track.

---

## the big picture: how rl training works

```
┌─────────────┐     action      ┌──────────────┐
│             │ ──────────────> │              │
│  PPO Agent  │                 │  Environment │
│  (brain)    │ <────────────── │  (world)     │
│             │  obs, reward    │              │
└─────────────┘                 └──────────────┘
      │                               │
      │ update weights                │ pybullet physics
      │ every 4096 steps              │ track boundaries
      │                               │ lidar ray casting
      ▼                               ▼
┌─────────────┐                 ┌──────────────┐
│ MLP Network │                 │ Track Gen    │
│ [128, 128]  │                 │ Reward Calc  │
│ ~20k params │                 │ Collision    │
└─────────────┘                 └──────────────┘
```

the agent and environment talk to each other through the gymnasium interface. the agent sends actions, the environment returns observations and rewards. this is the standard API that lets you swap out different algorithms (ppo, sac, a2c) without changing the environment, or swap environments without changing the algorithm.

**why this matters:** by following gymnasium's interface, we get access to the entire stable-baselines3 ecosystem - pre-built algorithms, callbacks, logging, evaluation tools - for free. we just had to build the environment correctly.

---

## file-by-file breakdown

### core files

| file | lines | what it does |
|------|-------|--------------|
| `environment.py` | 1166 | the gymnasium environment - defines observations, actions, rewards, physics integration, track detection, lidar |
| `train.py` | 366 | sets up ppo with hyperparameters, wraps the environment, configures callbacks, runs training |
| `evaluate.py` | 256 | loads a trained model, runs episodes, collects statistics, plots results |
| `test_environment.py` | 237 | validates the environment works before training (basic ops, full episodes, visualization) |

### setup files

| file | what it does |
|------|--------------|
| `setup/track (3).py` | procedural track generation - creates a different racetrack shape from a seed |
| `setup/controls (1).py` | tank-drive physics controller - converts velocity commands to pybullet forces |
| `setup/track_config (1).yaml` | all configuration: track geometry, physics constants, spawn position |
| `setup/car (1).urdf` | the car's physical model - dimensions, mass, wheel positions |

### output files

| file | what it does |
|------|--------------|
| `models/best_model/` | the best model found during training (saved automatically) |
| `models/checkpoints/` | periodic model snapshots every 25k steps |
| `models/ppo_car_final.zip` | the model at the very end of training |
| `logs/` | tensorboard logs for monitoring training |
| `log.md` | hand-written notes documenting 8 iterations of reward function design |

**tradeoff - file naming:** the setup files have spaces and numbers in their names (e.g., `track (3).py`). this is unusual for python because normal `import` statements don't work with spaces. the project works around this with `importlib.util.spec_from_file_location()`. the tradeoff is slightly more complex import code in exchange for preserving the original file naming convention.

---

## the environment in detail

the environment (`environment.py`) is the heart of the project. it implements `gym.Env` which requires four things:

1. **observation_space**: what the agent can see
2. **action_space**: what the agent can do
3. **reset()**: start a new episode
4. **step(action)**: take one action and return the result

### initialization flow

when `SelfDrivingCarEnv.__init__()` runs:

```
1. load yaml config
2. connect to pybullet (GUI or headless)
3. set gravity to 0.0 (top-down sim, no falling)
4. set timestep to 1/30 second
5. create Track object → generates procedural racetrack
6. spawn track cylinders in pybullet
7. calculate car spawn position (center of track at segment 0)
8. calculate spawn heading (pointing toward segment 1)
9. load car URDF into pybullet
10. define observation space (13D) and action space (2D)
11. initialize progress tracking variables
```

**design choice - spawn position calculation:** rather than hardcoding where the car starts, the environment calculates it dynamically from the track geometry. it averages the inner and outer boundary points at segment 0 to find the center of the track, then uses segment 1 to determine which direction to face. this means the spawn position automatically adapts to any track shape.

**tradeoff:** this is slightly more complex than a hardcoded position, but means the environment works correctly even if you change the track configuration.

### the step() method - one tick of the simulation

```python
def step(self, action):
    # 1. increment step counter
    # 2. clip action values to [-1, 1]
    # 3. convert action to velocities (action * max_velocity)
    # 4. get car's forward direction from rotation matrix
    # 5. transform local velocity to world coordinates
    # 6. apply velocities directly via resetBaseVelocity()
    # 7. enforce fixed height (prevent bouncing)
    # 8. run one pybullet physics step
    # 9. compute observation (lidar + imu)
    # 10. compute reward
    # 11. check if episode is over
    # 12. return (obs, reward, terminated, truncated, info)
```

**design choice - direct velocity control:** instead of applying forces/torques to the car (which would require tuning friction, motor torque, etc.), the code directly sets the car's velocity each step using `resetBaseVelocity()`. this is essentially saying "the car is going this fast right now" rather than "push the car with this force."

**tradeoff:** this makes the physics much simpler and more predictable (no wheel slip, no momentum from forces), but it's less physically realistic. the car can change direction instantly. for a learning project focused on RL rather than physics, this is the right call - it removes a huge source of complexity that would make the reward function much harder to design.

**design choice - fixed height enforcement:** every step checks if the car has drifted from its spawn height (z = 0.0675m) and resets it if so. this prevents the car from bouncing or getting stuck on track boundary cylinders.

**tradeoff:** this is a bit of a hack - it fights the physics engine rather than working with it. a more robust solution would be to use a 2D physics engine entirely. but pybullet only does 3D, so constraining the third dimension manually is the pragmatic approach.

---

## the track system

### how tracks are generated

the track generation in `setup/track (3).py` creates a closed racetrack shape through a multi-stage pipeline:

```
Step 1: Generate base angles
  100 evenly spaced angles [0, 2π)

Step 2: Warp angles
  Apply sinusoidal stretching with 3 harmonics
  Creates sections where points are packed (tight turns)
  and sections where points are sparse (straights)

Step 3: Build base oval
  Place points on an ellipse (not circle)
  Major axis = base_radius × 1.4
  Minor axis = base_radius / 1.4
  This creates a more natural racetrack shape

Step 4: Compute outward vectors
  For each point, find the direction pointing away from origin
  These are used to push points inward/outward

Step 5: Apply feature field (9 features)
  Add Gaussian bumps and dips along the track
  45% are outward bumps (creates straights by pushing track out)
  55% are inward dips (creates tight hairpin turns)

Step 6: Build chicanes (3 S-bends)
  Paired bumps: one outward, one inward, close together
  Creates characteristic S-bend shapes

Step 7: Apply high-frequency variation
  Harmonics 2-5 of sine waves
  Adds subtle asymmetry so the track doesn't feel artificially smooth

Step 8: Enforce no self-intersection
  Check if the resulting shape crosses itself
  If it does, scale down the variation by 0.8x up to 8 times
  Worst case: fall back to the clean oval

Step 9: Offset to create inner/outer boundaries
  Take the centerline and push it ±half_width along normals
  This creates the two edges of the track
```

**design choice - procedural generation with seed:** using a seed means the same track is generated every time with the same seed, which is critical for reproducible training. but you can change the seed to train on different tracks. this is the best of both worlds - deterministic for debugging, variable for generalization.

**tradeoff - complexity vs variety:** the track generation has a lot of parameters (9 features, 3 chicanes, 3 angle harmonics, high-freq variation). this creates interesting and varied tracks, but it also means some seeds might produce tracks that are too easy or too hard. simpler generation (like a perfect oval) would be more predictable but less interesting.

**the naming inversion:** there's a subtle but important detail - `track.inner_points` is actually the OUTER boundary of the track, and `track.outer_points` is the INNER boundary. this happens because of how offset normals work: offsetting by `-half_width` pushes outward (away from center), and `+half_width` pushes inward. the environment code handles this correctly in `_is_on_track()` with a comment explaining the swap.

**tradeoff:** this naming is confusing, but fixing it would require changes throughout the codebase. the alternative would be to swap the signs in `_offset_curve()`, but that would change the normal direction convention. the comment-and-move-on approach is pragmatic.

### how track boundaries work in physics

the track boundaries are made of small cylinders connecting successive boundary points:

```
point[0] --- cylinder --- point[1] --- cylinder --- point[2] ...
```

- 100 cylinders for the inner boundary
- 100 cylinders for the outer boundary
- each cylinder has radius 0.0125m (very thin)
- each cylinder has mass 0 (static, immovable)
- white color for visibility

**tradeoff - cylinders vs mesh:** using individual cylinders rather than a mesh creates tiny gaps at the joints between segments. a mesh would be watertight but harder to generate procedurally. the gaps are so small they don't affect gameplay. this is a good example of "good enough" engineering.

---

## the car model

the car is defined in `setup/car (1).urdf` using the URDF (Unified Robot Description Format) standard:

```
body dimensions: 0.25m x 0.125m x 0.075m (length x width x height)
body mass: 0.15625 kg
4 wheels:
  radius: 0.0375m
  width: 0.025m
  mass: 0.0078125 kg each
  continuous joints (can spin freely)
  positioned at corners of the body
```

**design choice - small, lightweight car:** the car is only 25cm long in the simulation. combined with zero gravity, this means the physics is more like a top-down 2D game than a realistic car simulation. the light mass means forces have large effects, which is fine since we're setting velocities directly anyway.

**design choice - functional wheels:** the wheels exist in the URDF and can rotate, but they don't actually drive the car. the car moves by having its velocity set directly. the wheels are purely visual - they make the car look like a car in the GUI.

**tradeoff:** this is a deliberate simplification. real car physics (tire friction, suspension, weight transfer, differential steering) would make the problem much harder to learn and much harder to design rewards for. since the goal is to learn RL concepts, not car physics, this simplification is correct.

---

## the observation space

the agent sees 13 numbers each step:

```
index  name                    range      source
-----  ----                    -----      ------
0      heading cosine          [-1, 1]    IMU (which way car faces)
1      heading sine            [-1, 1]    IMU (which way car faces)
2      forward velocity        [-1, 1]    wheel encoder (how fast)
3      angular velocity        [-1, 1]    IMU gyro (how fast turning)
4-12   lidar distances (x9)    [0, 1]     lidar scanner (wall distances)
```

### why cos/sin for heading instead of raw angle

raw heading angle goes from -pi to +pi, with a discontinuity at +/-pi (turning slightly past pi jumps to -pi). neural networks hate discontinuities - they create sharp gradients that destabilize learning. by using cos(heading) and sin(heading), the heading is represented as a smooth, continuous circle. facing east is (1,0), north is (0,1), west is (-1,0), south is (0,-1). every transition is smooth.

**tradeoff:** we use 2 numbers instead of 1, slightly increasing the observation space. this is a trivially small cost for a major stability improvement.

### why normalized velocities

the raw forward velocity can be up to 2.0 m/s and angular velocity up to 2.0 rad/s. dividing by the maximum and clipping to [-1, 1] puts everything on the same scale. neural networks learn faster when all inputs are in similar ranges because the gradient updates affect all weights proportionally.

**tradeoff:** none significant. normalization is almost always the right choice for neural network inputs.

### the lidar system in detail

```
                    ray 4 (forward, 0 deg)
                         |
            ray 3 (-22.5 deg) |  ray 5 (+22.5 deg)
                  \      |      /
        ray 2 (-45 deg)\    |    /ray 6 (+45 deg)
                     \   |   /
      ray 1 (-67.5 deg) \ | /  ray 7 (+67.5 deg)
                        \|/
    ray 0 (-90 deg) ---- CAR ---- ray 8 (+90 deg)
```

9 rays spanning the front 180 degree hemisphere. each ray tests against every segment of both track boundaries (200 segments total). for each ray:
1. compute ray direction from car heading + ray angle offset
2. for each boundary segment, find the intersection point using ray-segment math
3. keep the closest hit distance
4. normalize by max range (5.0m), clamp to [0, 1]

a value of 1.0 means no wall within 5 meters. a value near 0.0 means a wall is very close.

**design choice - 9 rays, 180 degree coverage:** 9 rays is enough to distinguish a wall on the left from a wall on the right, and to see turns coming. 180 degrees means the car can see to both sides but not behind it. this matches what a real car's forward-facing sensors would see.

**tradeoff - computation cost:** every step casts 9 rays against 200 segments = 1,800 intersection tests. this is the most expensive part of `_get_observation()`. alternatives:
- fewer rays (e.g., 5): faster but less spatial resolution
- spatial indexing: faster but more complex code
- pybullet's built-in ray casting: uses GPU but couples us tighter to pybullet

the current approach is simple and fast enough for training (~30 fps effective).

**design choice - no rear coverage:** the car can't see behind it. this is intentional - there's nothing useful behind the car on a racetrack. giving rear vision would add 9 more observations without helping the agent.

**design choice - no global position:** the agent has no idea where it is on the track in absolute terms. it only knows its heading, speed, and wall distances. this is a deliberate choice to make the policy more generalizable - a car trained this way should work on any track shape, not just the one it trained on.

**tradeoff:** without global position, the agent can't learn track-specific strategies like "brake hard before the hairpin at the 3 o'clock position." it has to react purely to what it sees. this makes learning harder but the result more robust.

### ray-segment intersection math

the `_ray_segment_intersection()` method solves a 2D linear system:

```
ray:     P = origin + t * dir        (t > 0 means forward)
segment: Q = A + s * (B - A)         (0 <= s <= 1 means on segment)
```

set P = Q and solve for t and s using the cross product:
```
cross = dir.x * seg.y - dir.y * seg.x
t = (diff.x * seg.y - diff.y * seg.x) / cross
s = (diff.x * dir.y - diff.y * dir.x) / cross
```

if `cross ~ 0`, the ray and segment are parallel (no hit). the `t > 0.01` check prevents the ray from hitting at its own origin.

---

## the action space

the agent outputs 2 continuous numbers:

```
action[0]: forward/backward  [-1, 1]  -> multiplied by max_linear_velocity (2.0 m/s)
action[1]: left/right turn   [-1, 1]  -> multiplied by max_angular_velocity (2.0 rad/s)
```

**design choice - continuous actions:** discrete actions (e.g., forward/left/right) would be simpler but produce jerky movement. continuous actions allow smooth curves and partial throttle. ppo handles continuous action spaces well through gaussian policy distributions.

**design choice - 2D action space:** the car has two degrees of freedom: speed and turn rate. this matches the tank-drive control model where the car can move forward/backward and rotate in place. there's no separate brake action - negative forward_input is reverse.

**tradeoff - no brake vs reverse:** in a real car, braking and reversing are different (braking slows you down, reversing drives backward). here, setting action[0] to -1 means "go backward at full speed," which the agent is unlikely to learn as useful. a separate brake action would add complexity but might make learning more natural. in practice, the agent learns to just use positive forward values and steer.

### action-to-velocity conversion

```python
linear_velocity = forward_input * 2.0   # m/s in car's forward direction
angular_velocity = turn_input * 2.0     # rad/s around vertical axis
```

the linear velocity is then transformed from local to world coordinates using the car's rotation matrix:

```python
world_vel_x = linear_velocity * car_forward[0]
world_vel_y = linear_velocity * car_forward[1]
```

this means action[0] = 1.0 always drives the car forward relative to where it's facing, regardless of its orientation in the world.

---

## the reward function and its 8 iterations

the reward function is the most critical and hardest part of any RL project. this project went through 8 iterations, documented in `log.md`. each iteration reveals a common RL pitfall.

### iteration 1: naive speed reward

```
survival: +0.5/step, velocity: +1.0 * speed, off-track: -50
```

**what happened:** the car learned to go as fast as possible and crashed after 1/4 of the track. the velocity reward dominated everything - the optimal strategy was to floor it, even if that meant crashing.

**lesson:** if speed is the biggest reward, the agent will prioritize speed over everything else, including staying alive.

### iteration 2: survival-focused

```
survival: +1.0/step, centering: +0.5, velocity: x0.1, off-track: -100
```

**what happened:** the car stopped moving entirely. it discovered that sitting still on the track gives +1.0/step indefinitely with zero risk.

**lesson:** if survival is the biggest reward and there's no penalty for inaction, the agent will do nothing. this is a classic RL exploit called "reward hacking."

### iteration 3: minimum speed penalty

```
added: -0.3 if speed < 0.5
```

**what happened:** the car moved forward then backward repeatedly. it maintained "speed" (the instantaneous velocity was high) without actually going anywhere.

**lesson:** instantaneous velocity is a flawed metric. the agent can show high velocity while making zero progress by oscillating.

### iteration 4: angle-based progress

```
replaced velocity with: angle_change * 10.0
```

**what happened:** not tested in this iteration, but the idea was right - measure actual progress around the track rather than speed.

**lesson:** progress-based rewards are much harder to hack than velocity-based ones, because going back and forth cancels out.

### iteration 5: research-based reward

```
time penalty: -0.1/step, progress: +1.0 (clipped), off-track: -100, lap bonus: +100
```

**what happened:** the car spiraled inward toward the center. the angle-from-origin metric rewarded turning toward the center because that changes the angle rapidly without following the track.

**lesson:** "progress" must be carefully defined. angle from origin works for a perfect circle but not for arbitrary track shapes. the metric must account for the actual track direction.

### iteration 6: track tangent velocity

```
survival: +0.5, velocity along track tangent: +1.0 (clipped), off-track: -100
```

**what happened:** the car oversteered and hugged walls. it followed the track but in an aggressive, crash-prone way.

**lesson:** velocity along the track tangent is a better progress metric, but without centering incentive, the car takes the tightest possible racing line and crashes on sharp turns.

### iteration 7: priority-based reward

```
survival: +1.0, centering: +0.3, speed: +0.2, steering penalty: -0.2, off-track: -100
```

**what happened:** this was a more balanced approach with clear priority ordering: stay alive > stay centered > go fast > steer gently.

**lesson:** reward components should have explicit priority ordering. the ratios between them determine what the agent prioritizes.

### iteration 8: the current reward (in environment.py)

the final reward structure uses angular progress as the primary signal:

```python
# PRIMARY: angular progress around track center
delta_angle = current_angle - last_angle  # how much angle changed
# wrap to [-pi, pi] to handle the discontinuity at +/-pi
progress_reward = delta_angle * 50.0      # dominant reward signal

# SECONDARY: centering bonus
min_edge_dist / half_width * 0.1          # small bonus for staying centered

# TERMINAL: off-track penalty
-10.0                                     # plus episode ends (losing all future rewards)
```

**why this works:** the progress reward dominates at ~0.83/step at full speed (delta_angle ~ 0.017 rad x 50 = 0.83), while centering gives only 0.1/step max. this means the agent's primary objective is making progress, with a gentle nudge to stay centered.

**why -10 for off-track instead of -100:** the episode also terminates when going off-track. this means the agent loses all future rewards (potentially thousands of points over the remaining steps). the -10 is just a small extra signal - the real punishment is the lost future.

**the angle wrapping trick:**
```python
if delta_angle > np.pi:
    delta_angle = delta_angle - 2.0 * np.pi
if delta_angle < -np.pi:
    delta_angle = delta_angle + 2.0 * np.pi
```

`atan2` returns values in [-pi, pi]. when the car crosses the -pi/+pi boundary, the raw delta would be ~ 2pi (a huge false reward or penalty). wrapping catches this and converts it to the true small delta.

### reward engineering meta-lessons

1. **never make the easiest thing the most rewarding** - if survival pays the most, the agent will just survive
2. **measure outcomes not actions** - reward progress, not speed
3. **the ratio between rewards matters more than absolute values** - a +1000 reward means nothing if the penalty is +2000
4. **termination is a powerful implicit penalty** - ending the episode costs future rewards
5. **every metric can be hacked** - speed (oscillate), angle (spiral), survival (sit still). your job is to make the correct behavior the easiest way to maximize reward
6. **iterate, don't design** - no one gets the reward function right on the first try. log your iterations.

---

## the training pipeline

### environment wrapping

```python
env = SelfDrivingCarEnv(config_path, render_mode=None)
env = Monitor(env)           # logs episode rewards/lengths
env = DummyVecEnv([env])     # vectorized wrapper (required by SB3)
```

**why Monitor:** records `ep_rew_mean` (average episode reward) and `ep_len_mean` (average episode length) which are the two most important training metrics. without Monitor, you'd have no visibility into training progress.

**why DummyVecEnv:** stable-baselines3 requires all environments to be vectorized, even if there's only one. DummyVecEnv runs environments sequentially (as opposed to SubprocVecEnv which uses multiprocessing). for a single environment, there's no performance difference.

**tradeoff - single vs multiple environments:** training with multiple parallel environments (e.g., 8) would be ~8x faster because you collect experience 8x faster. the tradeoff is higher memory usage and more complex debugging. for a learning project, a single environment is simpler to understand.

### the PPO model

```python
model = PPO(
    policy="MlpPolicy",          # feedforward neural network
    env=train_env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[128, 128]),
    device="cuda",
    tensorboard_log=logs_dir,
)
```

**total parameters: ~20,000** (13 inputs -> 128 -> 128 -> 2 outputs for policy, plus a parallel value head)

### callbacks

**EvalCallback** (every 10,000 steps):
- creates a separate environment instance
- runs 5 episodes with deterministic actions (no exploration noise)
- if the mean reward is the best seen so far, saves the model
- this gives you the best model regardless of when training ended

**CheckpointCallback** (every 25,000 steps):
- unconditionally saves the model
- creates files like `ppo_car_25000_steps.zip`
- acts as backup in case of crashes or power loss

**tradeoff - eval frequency:** evaluating every 10,000 steps means 20 evaluations during 200k training. more frequent eval gives finer-grained best-model selection but slows training (5 eval episodes x 3000 max steps = up to 15,000 extra steps per eval). 10,000 is a reasonable middle ground.

### training output

```
models/
├── best_model/
│   └── best_model.zip     # the one you probably want to use
├── checkpoints/
│   ├── ppo_car_25000_steps.zip
│   ├── ppo_car_50000_steps.zip
│   └── ...
└── ppo_car_final.zip       # state at end of training

logs/
└── PPO_1/                  # tensorboard logs
    └── events.out.tfevents.*
```

---

## ppo algorithm explained

PPO (Proximal Policy Optimization) is the training algorithm. here's what it does at a conceptual level:

### the policy network

the policy network takes 13 observation values and outputs:
- **mean** of a gaussian distribution for each action (2 values)
- **log standard deviation** for each action (2 values, learned)

to pick an action, it samples from these gaussians. during training, the randomness helps explore. during evaluation, it uses the mean directly (deterministic).

### the value network

a separate head (sharing the first layers) predicts "how much total reward will the agent get from this state onward?" this is called the value function V(s). it's used to compute advantages.

### the training loop

```
repeat for 200,000 total steps:
    1. ROLLOUT: run the policy for 4096 steps, recording:
       - observations, actions, rewards, dones, values, log_probs

    2. COMPUTE ADVANTAGES using GAE:
       A_t = sum of (gamma * lambda)^k * delta_{t+k}
       where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
       advantages tell us "was this action better or worse than expected?"

    3. UPDATE for 10 epochs:
       shuffle the 4096 transitions into mini-batches of 128
       for each mini-batch:
         a. compute the policy ratio: pi_new(a|s) / pi_old(a|s)
         b. clip the ratio to [1-epsilon, 1+epsilon] where epsilon=0.2
         c. policy loss = -min(ratio * advantage, clipped_ratio * advantage)
         d. value loss = (V(s) - returns)^2
         e. entropy bonus = -ent_coef * entropy(pi)
         f. total loss = policy_loss + vf_coef * value_loss - entropy_bonus
         g. backpropagate and clip gradients to max_norm=0.5
```

### why PPO specifically?

**vs DQN (Deep Q-Network):** DQN only works with discrete actions. our car needs continuous control.

**vs SAC (Soft Actor-Critic):** SAC is often better for continuous control but requires a replay buffer (more memory) and is more sensitive to hyperparameters. PPO is simpler and more forgiving.

**vs A2C (Advantage Actor-Critic):** A2C updates after every step or small batch, making it less stable. PPO's clipping mechanism makes updates much safer.

**vs TRPO (Trust Region Policy Optimization):** TRPO was PPO's predecessor. it uses a KL divergence constraint (mathematically elegant but computationally expensive). PPO approximates the same thing with a simple clip operation. nearly identical results, much simpler code.

**tradeoff:** PPO is not the most sample-efficient algorithm (SAC usually learns faster per sample), but it's the most robust. it rarely diverges, works with minimal tuning, and handles both discrete and continuous actions. this makes it the default choice for new RL projects.

---

## hyperparameter choices and tradeoffs

### learning_rate = 3e-4

this is the "adam default" and a common starting point for RL. it controls how much the network weights change per gradient update.

- **higher (e.g., 1e-3):** learns faster but risks overshooting and destabilizing
- **lower (e.g., 1e-5):** more stable but may not learn fast enough in 200k steps
- **3e-4** is the standard safe choice

### n_steps = 4096

how many steps to collect before each policy update. affects the bias-variance tradeoff of advantage estimation.

- **higher:** more stable gradients, better advantage estimates, but slower iteration
- **lower (e.g., 256):** faster iteration but noisier gradients
- **4096** is on the larger side, chosen because the sensor-only observation space makes the problem harder and benefits from more data per update

**tradeoff:** with 200k total steps and 4096 per rollout, we get only ~49 policy updates total. that's not many. more updates (smaller n_steps) could help, but might be noisier.

### batch_size = 128

how many transitions per gradient step within each epoch.

- **must divide n_steps evenly:** 4096 / 128 = 32 gradient steps per epoch
- **larger batches:** more stable but fewer updates per epoch
- **smaller batches:** more updates but noisier

### n_epochs = 10

how many times to reuse the collected data.

- **higher:** more sample efficient (squeeze more learning from each rollout)
- **lower:** less risk of overfitting to the rollout data
- **10** is standard PPO. going higher risks "policy collapse" where the policy changes too much from its rollout behavior

### gamma = 0.995

discount factor - how much future rewards matter relative to immediate ones.

- **0.99:** at step t, a reward 100 steps later is worth 0.99^100 = 0.366 of its face value
- **0.995:** at step t, a reward 100 steps later is worth 0.995^100 = 0.606 of its face value
- **0.995** was chosen because without global position, the agent needs to plan further ahead using only local sensor data. higher gamma means distant rewards still influence current decisions.

**tradeoff:** higher gamma makes the value function harder to learn (it has to predict further into the future), but gives better long-term behavior.

### gae_lambda = 0.95

controls the bias-variance tradeoff in advantage estimation.

- **1.0:** pure monte carlo (high variance, zero bias) - uses actual returns
- **0.0:** pure TD (low variance, high bias) - uses only one-step bootstrap
- **0.95:** mostly uses longer-horizon returns, with slight bias toward short-term

### clip_range = 0.2

PPO's signature feature. limits how much the policy can change per update.

```
ratio = pi_new(a|s) / pi_old(a|s)
clipped_ratio = clip(ratio, 1-0.2, 1+0.2) = clip(ratio, 0.8, 1.2)
```

this means the new policy can be at most 20% more likely or 20% less likely to take any given action compared to the old policy. this prevents catastrophic forgetting.

### ent_coef = 0.02

entropy bonus - rewards the policy for being uncertain (exploring).

- **higher (e.g., 0.1):** more exploration, slower convergence
- **lower (e.g., 0.001):** less exploration, risk of getting stuck
- **0.02** is moderately high, reflecting the need for exploration in a sensor-only setting where the agent can't "see" the whole track

### policy network = [128, 128]

two hidden layers with 128 neurons each.

- **wider (e.g., [256, 256]):** more capacity but harder to train, more prone to overfitting
- **narrower (e.g., [64, 64]):** faster but might not have enough capacity for spatial reasoning from lidar
- **deeper (e.g., [64, 64, 64]):** more layers can learn hierarchical features but add gradient issues
- **[128, 128]** is a common middle ground for continuous control tasks

### total_timesteps = 200,000

- each episode can be up to 3000 steps
- 200k steps ~ 67 full episodes (if no early termination)
- in practice, many episodes end early (car goes off track), so the agent sees more episodes
- this is enough for basic driving but not enough for truly optimal racing lines

**tradeoff:** more steps = better performance but longer training time. 200k is a good starting point for iteration speed.

---

## evaluation system

`evaluate.py` loads a trained model and tests it:

```python
model = PPO.load(model_path)
for episode in range(num_episodes):
    obs = env.reset()
    while not done:
        action = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
```

**key detail - deterministic=True:** during training, the policy samples from a gaussian (adding random exploration). during evaluation, we use the mean of the gaussian directly. this gives the best possible performance from the learned policy.

the evaluator collects:
- per-episode rewards and lengths
- mean and standard deviation
- generates matplotlib plots (histogram of rewards, line plot of episode lengths)

---

## physics simulation choices

### why pybullet

- free and open source (vs MuJoCo which was paid until recently)
- good python bindings
- supports URDF loading
- headless mode (DIRECT) for fast training
- gui mode for visualization

**tradeoff vs other engines:**
- **MuJoCo:** faster, better contact physics, but was historically paid. now free but pybullet is still simpler to set up
- **Box2D:** truly 2D which would be more appropriate, but doesn't support URDF and has fewer features
- **Unity ML-Agents:** better graphics but heavy setup, requires unity editor

### zero gravity

```yaml
gravity: 0.0
```

the simulation runs with no gravity. this means the car floats at its spawn height. combined with the fixed-height enforcement in `step()`, this creates effectively 2D physics in a 3D engine.

**why not just use a 2D engine?** pybullet's URDF loading, collision detection, and visualization are very convenient. the overhead of fighting the third dimension is smaller than the overhead of reimplementing these features in a 2D framework.

### timestep = 1/30 second

```yaml
time_step: 0.03333333333333333
```

each `step()` call advances the simulation by 1/30 of a second. at 3000 max steps per episode, that's 100 seconds of simulated driving time.

**tradeoff:** smaller timesteps (e.g., 1/240 typical for pybullet) give more accurate physics but require more steps per episode. since we're setting velocities directly (not simulating forces), physics accuracy isn't critical, so 1/30 is fine.

---

## interesting design patterns

### 1. importlib for spaces in filenames

```python
spec = importlib.util.spec_from_file_location("track", "setup/track (3).py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Track = module.Track
```

this is the python way to import from non-standard file paths. it creates a module spec, builds an empty module, and executes the file's code into it.

### 2. ray-casting polygon test

the `_point_in_polygon()` method uses the classic ray-casting algorithm:
- shoot a ray from the test point going rightward to infinity
- count how many polygon edges it crosses
- odd = inside, even = outside

this works for any polygon shape (convex or concave) and is O(n) where n is the number of edges. it's the standard algorithm used in GIS, game engines, and computational geometry.

### 3. separate termination vs truncation

gymnasium distinguishes between:
- **terminated:** the episode ended because of the agent's actions (went off track, flipped)
- **truncated:** the episode ended because of time limit (3000 steps)

this distinction matters for value estimation. when terminated, future value is 0 (bad thing happened). when truncated, future value should be estimated from the value function (the agent was doing fine, just ran out of time).

### 4. debug visualization

the `_draw_debug_visuals()` method renders a HUD overlay in the pybullet GUI:
- speed, step reward, episode reward, step counter, lap progress, on-track status
- 9 lidar rays with color gradient (cyan at sides, green forward)
- uses pybullet's `addUserDebugText` and `addUserDebugLine`
- replaces previous debug items by ID to avoid accumulation

this is only active in `render_mode="human"` and doesn't affect training performance.

### 5. the monitor-callback-tensorboard pipeline

```
Monitor -> logs ep_rew_mean, ep_len_mean to files
EvalCallback -> runs eval episodes, saves best model, logs to tensorboard
CheckpointCallback -> periodic model saves
TensorBoard -> reads log files, shows interactive graphs
```

this is the standard SB3 monitoring stack. each component does one thing and they compose cleanly.

---

## known limitations and future improvements

### current limitations

1. **single environment training:** only one environment at a time. using 4-8 parallel environments with SubprocVecEnv would be 4-8x faster.

2. **fixed track per training run:** the agent trains on one track. it may not generalize to different tracks without domain randomization (training on many different seeds).

3. **no curriculum learning:** the agent faces the full difficulty from step 1. starting with a simple oval and gradually increasing complexity could speed up learning.

4. **no observation normalization:** SB3 provides VecNormalize which standardizes observations based on running statistics. this often improves training stability.

5. **angle-based progress on non-circular tracks:** the progress metric uses angle from origin, which assumes a roughly circular track. very elongated or irregularly shaped tracks could give misleading progress signals.

6. **lidar is computed in python:** the 1,800 ray-segment intersection tests per step are done in a python loop. this could be vectorized with numpy or offloaded to pybullet's built-in ray casting for a significant speedup.

7. **no reward for completing a full lap:** the current reward function has no explicit lap completion bonus. this would provide a strong sparse signal for long-term planning.

### potential improvements

- **parallel training environments** for faster data collection
- **domain randomization** (random track seeds per episode) for generalization
- **curriculum learning** (start simple, increase difficulty)
- **observation normalization** with VecNormalize
- **recurrent policy** (LSTM) to handle partial observability
- **reward shaping** with potential-based functions for faster convergence
- **hyperparameter tuning** with optuna or similar
- **imitation learning** from a human driver as a warm start

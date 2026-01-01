# important code snippets

the most important pieces of code in this project, explained line by line with why they matter.

---

## 1. the observation builder

**file:** `environment.py` lines 529-588
**why it matters:** this is what the agent "sees." if the observation is bad, no amount of training will help.

```python
def _get_observation(self):
    car_pos, car_orn = p.getBasePositionAndOrientation(
        self.car_id, physicsClientId=self.physics_client
    )
    car_vel, car_ang_vel = p.getBaseVelocity(
        self.car_id, physicsClientId=self.physics_client
    )

    heading = self._get_car_heading(car_orn)

    rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn))
    rot_matrix = rot_matrix.reshape((3, 3))
    car_forward = rot_matrix[:, 0]

    forward_vel = car_vel[0] * car_forward[0] + car_vel[1] * car_forward[1]
    angular_vel = car_ang_vel[2]

    normalized_forward_vel = np.clip(
        forward_vel / self.max_linear_velocity, -1.0, 1.0
    )
    normalized_angular_vel = np.clip(
        angular_vel / self.max_angular_velocity, -1.0, 1.0
    )

    car_pos_2d = np.array([car_pos[0], car_pos[1]])
    lidar_distances = self._cast_lidar_rays(car_pos_2d, heading)

    observation = np.concatenate([
        np.array([
            np.cos(heading),
            np.sin(heading),
            normalized_forward_vel,
            normalized_angular_vel,
        ], dtype=np.float32),
        lidar_distances,
    ])

    return observation
```

**line by line:**

- `car_pos, car_orn = p.getBasePositionAndOrientation(...)` - asks pybullet where the car is and which way its rotated. car_pos is [x, y, z], car_orn is a quaternion [x, y, z, w]
- `car_vel, car_ang_vel = p.getBaseVelocity(...)` - asks pybullet how fast the car is moving and spinning. car_vel is [vx, vy, vz], car_ang_vel is [wx, wy, wz]
- `rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))` - converts the quaternion to a 3x3 rotation matrix. quaternions are compact but hard to use directly. the rotation matrix tells us which way each axis of the car points in world coordinates
- `car_forward = rot_matrix[:, 0]` - the first column of the rotation matrix is the car's local X axis in world coordinates. this is the "forward" direction because URDF convention puts the front of the car along X
- `forward_vel = car_vel[0] * car_forward[0] + car_vel[1] * car_forward[1]` - dot product of world velocity with forward direction. this gives us how fast the car moves in the direction its facing. sideways drift doesnt count. this is critical because the agent needs to know if its actually going forward, not just moving in some random direction
- `np.clip(forward_vel / self.max_linear_velocity, -1.0, 1.0)` - normalize to [-1, 1]. dividing by max velocity turns m/s into a percentage. clipping guarantees the value stays in bounds even if physics glitches push velocity past the max
- `np.cos(heading), np.sin(heading)` - encode the heading angle as two smooth values instead of one discontinuous one. heading goes from -pi to +pi and jumps at the boundary. cos/sin are smooth everywhere. this prevents the neural network from seeing a huge input discontinuity when the car faces west

**what would break if you changed this:**
- removing cos/sin and using raw heading: training would destabilize near the -pi/+pi boundary
- removing velocity normalization: the network would treat velocity as more important than lidar just because the numbers are bigger
- using world velocity instead of forward velocity: the agent couldnt tell if it was going forward or sideways

---

## 2. the reward function

**file:** `environment.py` lines 590-688
**why it matters:** the reward function is the ONLY way the agent knows what good behavior is. every design choice here directly shapes what the car learns to do.

```python
def _compute_reward(self, action):
    car_pos, car_orn = p.getBasePositionAndOrientation(
        self.car_id, physicsClientId=self.physics_client
    )
    car_vel, _ = p.getBaseVelocity(
        self.car_id, physicsClientId=self.physics_client
    )

    reward = 0.0
    is_on_track = self._is_on_track(car_pos)

    # off track penalty
    if is_on_track == False:
        reward = reward - 10.0
        return reward

    # progress reward (PRIMARY)
    current_angle = self._get_progress_angle(car_pos)
    delta_angle = current_angle - self.last_angle

    if delta_angle > np.pi:
        delta_angle = delta_angle - 2.0 * np.pi
    if delta_angle < -np.pi:
        delta_angle = delta_angle + 2.0 * np.pi

    progress_reward = delta_angle * 50.0
    reward = reward + progress_reward

    self.last_angle = current_angle
    self.total_progress = self.total_progress + delta_angle

    # centering bonus (minor)
    dist_inner, dist_outer, closest_idx = self._get_distances_to_track(car_pos)
    track_width = self.track.track_width

    if dist_inner < dist_outer:
        min_edge_dist = dist_inner
    else:
        min_edge_dist = dist_outer

    half_width = track_width / 2.0
    centering_score = min_edge_dist / half_width

    if centering_score > 1.0:
        centering_score = 1.0
    if centering_score < 0.0:
        centering_score = 0.0

    reward = reward + centering_score * 0.1

    return reward
```

**the three reward components explained:**

**off-track penalty (-10.0):** returns immediately with -10. the episode also ends (in `_is_terminated()`), which means the agent loses all future rewards too. if the car was earning ~0.83/step and had 2000 steps left, going off track costs -10 + the lost 1660 points of future reward. the termination does most of the punishing, not the -10.

**progress reward (delta_angle * 50.0):** measures how much the car's angle from the origin changed since last step. `_get_progress_angle()` calls `np.arctan2(y, x)` which gives the angle of the car's position relative to the track center. if the car drives counterclockwise around the track, this angle increases. the multiplier of 50.0 makes a typical step worth ~0.83 reward (delta_angle ~ 0.017 rad at full speed on a ~4m radius track). this is the dominant signal.

**the angle wrapping code:**
```python
if delta_angle > np.pi:
    delta_angle = delta_angle - 2.0 * np.pi
if delta_angle < -np.pi:
    delta_angle = delta_angle + 2.0 * np.pi
```
this is critical. `arctan2` returns [-pi, pi]. when the car crosses the boundary (e.g., from angle 3.1 to -3.1), the raw delta would be -6.2, which would give a massive false penalty of -310. the wrapping converts this to the true delta of +0.08. without this code, the car would be punished every time it crosses the west-facing direction and training would fail.

**centering bonus (centering_score * 0.1):** find whichever edge is closer (inner or outer), divide by half the track width. a car perfectly centered gets centering_score = 1.0 (worth +0.1). a car touching an edge gets ~0.0. this is intentionally small (0.1 vs 0.83 for progress) so it gently nudges the car toward the middle without overriding the drive to make progress.

---

## 3. the step function

**file:** `environment.py` lines 814-933
**why it matters:** this is the core simulation loop. every single training step goes through here.

```python
def step(self, action):
    self.current_step = self.current_step + 1

    forward_input = float(action[0])
    turn_input = float(action[1])

    if forward_input < -1.0:
        forward_input = -1.0
    if forward_input > 1.0:
        forward_input = 1.0
    if turn_input < -1.0:
        turn_input = -1.0
    if turn_input > 1.0:
        turn_input = 1.0

    linear_velocity = forward_input * self.max_linear_velocity
    angular_velocity = turn_input * self.max_angular_velocity

    car_pos, car_orn = p.getBasePositionAndOrientation(
        self.car_id, physicsClientId=self.physics_client
    )

    rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn))
    rot_matrix = rot_matrix.reshape((3, 3))
    car_forward = rot_matrix[:, 0]

    linear_velocity_world_x = linear_velocity * car_forward[0]
    linear_velocity_world_y = linear_velocity * car_forward[1]

    p.resetBaseVelocity(
        objectUniqueId=self.car_id,
        linearVelocity=[linear_velocity_world_x, linear_velocity_world_y, 0.0],
        angularVelocity=[0.0, 0.0, angular_velocity],
        physicsClientId=self.physics_client
    )

    fixed_height = self.config['spawn']['position'][2]
    height_diff = car_pos[2] - fixed_height
    if abs(height_diff) > 0.001:
        p.resetBasePositionAndOrientation(
            self.car_id,
            [car_pos[0], car_pos[1], fixed_height],
            car_orn,
            physicsClientId=self.physics_client
        )

    p.stepSimulation(physicsClientId=self.physics_client)

    observation = self._get_observation()
    reward = self._compute_reward(action)
    terminated = self._is_terminated()
    truncated = self._is_truncated()

    return observation, reward, terminated, truncated, info
```

**the important parts:**

**action clipping:** the neural network outputs values from a gaussian distribution, which can go beyond [-1, 1]. clipping ensures the car never gets a command outside its valid range. without this, a network output of 5.0 would mean 10 m/s velocity instead of the intended 2 m/s max.

**local-to-world velocity transform:** `linear_velocity * car_forward[0]` converts "go forward at 2 m/s" into world-space components. if the car faces northeast, car_forward might be [0.707, 0.707, 0]. so 2 m/s forward becomes [1.414, 1.414, 0] in world coordinates. this is why the car always moves in the direction its facing.

**resetBaseVelocity vs applyExternalForce:** this is a huge design decision. `resetBaseVelocity` says "the car is NOW moving at this speed." `applyExternalForce` says "push the car with this force." with forces, the car has momentum, inertia, and takes time to accelerate/decelerate. with velocity setting, the car responds instantly. we chose velocity setting because it makes the RL problem much simpler - the agent doesn't need to learn about momentum.

**the z-velocity is locked to 0.0** in the resetBaseVelocity call, and angular velocity is locked to [0, 0, angular_velocity] (only rotation around the vertical axis). this prevents the car from moving up/down or tumbling.

**fixed height enforcement:** if pybullet's physics cause the car to drift vertically (e.g., from colliding with a track cylinder), this code snaps it back. the 0.001 threshold means it only intervenes when the drift is noticeable, avoiding unnecessary position resets every step.

**the order matters:** velocity is set BEFORE `stepSimulation()`. pybullet uses the current velocity to compute the next position. then we observe the result AFTER the simulation step. this ensures the observation reflects the consequences of the action.

---

## 4. lidar ray casting

**file:** `environment.py` lines 312-343
**why it matters:** lidar is how the car "sees" the walls. its the primary spatial awareness mechanism.

```python
def _cast_lidar_rays(self, car_pos_2d, heading):
    distances = np.ones(self.LIDAR_NUM_RAYS, dtype=np.float32)

    for i in range(self.LIDAR_NUM_RAYS):
        ray_angle = heading + np.radians(self.LIDAR_ANGLES_DEG[i])
        ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])

        min_t = self.LIDAR_MAX_RANGE

        for boundary in [self.track.inner_points[:, :2], self.track.outer_points[:, :2]]:
            n = len(boundary)
            for j in range(n):
                k = (j + 1) % n
                t = self._ray_segment_intersection(
                    car_pos_2d, ray_dir, boundary[j], boundary[k]
                )
                if t is not None and t < min_t:
                    min_t = t

        distances[i] = min_t / self.LIDAR_MAX_RANGE
        if distances[i] > 1.0:
            distances[i] = 1.0

    return distances
```

**how it works:**

1. start with all distances = 1.0 (max range, no wall detected)
2. for each of the 9 rays:
   - calculate the ray's world angle by adding the car's heading to the ray's offset angle
   - convert angle to a direction vector [cos, sin]
   - test this ray against EVERY segment of BOTH boundaries (inner and outer)
   - keep the closest hit
   - normalize by max range (5.0m) so output is [0, 1]

**the ray angles are:** `[-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]` degrees relative to the car's heading. this fans out from left (-90) through forward (0) to right (+90).

**`(j + 1) % n`** wraps around so the last point connects back to the first, forming a closed loop. without the modulo, the last segment would be missing and the car could "see through" the gap.

**the nested loop structure:** 9 rays x 2 boundaries x 100 segments = 1800 intersection tests per step. this is the most computationally expensive part of the environment. each test is the `_ray_segment_intersection` call below.

---

## 5. ray-segment intersection

**file:** `environment.py` lines 935-963
**why it matters:** this is the math that makes lidar work. its called 1800 times per step.

```python
def _ray_segment_intersection(self, ray_origin, ray_dir, seg_a, seg_b):
    seg_dx = seg_b[0] - seg_a[0]
    seg_dy = seg_b[1] - seg_a[1]

    cross = ray_dir[0] * seg_dy - ray_dir[1] * seg_dx

    if abs(cross) < 1e-10:
        return None

    diff_x = seg_a[0] - ray_origin[0]
    diff_y = seg_a[1] - ray_origin[1]

    t = (diff_x * seg_dy - diff_y * seg_dx) / cross
    s = (diff_x * ray_dir[1] - diff_y * ray_dir[0]) / cross

    if t > 0.01 and 0 <= s <= 1:
        return t

    return None
```

**the math:**

we have two parametric lines:
```
ray:     P = origin + t * dir        (the lidar beam)
segment: Q = seg_a + s * (seg_b - seg_a)   (a piece of track boundary)
```

set them equal and solve for t and s:
```
origin + t * dir = seg_a + s * (seg_b - seg_a)
```

this is a 2x2 linear system. the cross product `ray_dir[0] * seg_dy - ray_dir[1] * seg_dx` is the determinant of the system matrix. if its near zero, the ray and segment are parallel (no intersection possible).

- **t** = distance along the ray to the intersection. must be > 0.01 (not behind the car, and not at the car's exact position)
- **s** = position along the segment. must be in [0, 1] meaning the hit is actually on the segment, not on its extension

**why t > 0.01 instead of t > 0:** prevents the ray from "hitting" at its own origin point. if the car is exactly on a boundary point, t would be ~0 and every ray would report a wall at distance 0. the 0.01 buffer (1cm) prevents this.

---

## 6. point-in-polygon (track detection)

**file:** `environment.py` lines 468-527
**why it matters:** this decides whether the car is on or off the track. getting this wrong means either false deaths (car punished while on track) or false safety (car rewarded while off track).

```python
def _point_in_polygon(self, point, polygon):
    x = point[0]
    y = point[1]
    n = len(polygon)
    inside = False
    j = n - 1

    for i in range(n):
        xi = polygon[i][0]
        yi = polygon[i][1]
        xj = polygon[j][0]
        yj = polygon[j][1]

        edge_crosses_y = (yi > y) != (yj > y)

        if edge_crosses_y:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside

        j = i

    return inside
```

**the algorithm (ray casting):**

imagine standing at the test point and looking east (to the right). count how many polygon edges cross your line of sight:
- 0 crossings = outside
- 1 crossing = inside
- 2 crossings = outside (went in and came back out)
- 3 crossings = inside
- etc.

odd = inside, even = outside

**`(yi > y) != (yj > y)`** checks if the edge straddles the test point's y coordinate. if both endpoints are above or both are below, the edge cant cross our horizontal ray.

**`x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi`** finds the x coordinate where the edge crosses our horizontal line. this is linear interpolation along the edge.

**`if x < x_intersection`** only counts crossings to the RIGHT of the test point (the ray goes rightward to infinity).

**`inside = not inside`** flips the boolean each time we cross an edge. starts at False (outside). after 1 crossing its True (inside). after 2 its False again, etc.

**used in `_is_on_track()`:**
```python
inside_larger = self._point_in_polygon(pos_2d, self.track.inner_points[:, :2])
inside_smaller = self._point_in_polygon(pos_2d, self.track.outer_points[:, :2])
is_on = inside_larger and (inside_smaller == False)
```

the track is a donut. the car must be inside the outer boundary (larger polygon) AND outside the inner boundary (smaller polygon). the confusing naming (`inner_points` = outer boundary) is because of how track offsets are computed.

---

## 7. track centerline generation

**file:** `setup/track (3).py` lines 76-87
**why it matters:** this is the pipeline that creates the entire track shape. every other piece of track geometry derives from this.

```python
def _generate_centerline_points(self):
    base_angles = np.linspace(0, 2 * np.pi, self.num_segments, endpoint=False)
    warped_angles = self._warp_angles(base_angles)
    base_points = self._build_base_oval(warped_angles)
    outward = self._compute_outward_vectors(base_points)
    variation = self._compose_variation_profile(warped_angles)
    variation = self._limit_variation(variation)
    points = base_points + outward * variation[:, None]
    points = self._enforce_no_self_intersection(points, base_points, outward, variation)
    z = np.full((points.shape[0], 1), self.line_height)
    return np.hstack([points, z])
```

**each line is a stage of the pipeline:**

1. `np.linspace(0, 2*pi, 100)` - create 100 evenly spaced angles around a circle
2. `_warp_angles()` - stretch and compress these angles using sine harmonics. this makes some sections have points packed close together (tight turns) and others spread out (straights)
3. `_build_base_oval()` - place points on an ellipse instead of a circle. major axis = radius * 1.4, minor = radius / 1.4. this gives the oval racetrack shape
4. `_compute_outward_vectors()` - for each point, find which direction points away from the origin. these are the directions we'll push points to add track features
5. `_compose_variation_profile()` - create a 1D array of how much to push each point inward or outward. combines gaussian bumps (features), paired bumps (chicanes), and sine waves (ripples)
6. `_limit_variation()` - make sure no point gets pushed so far that the track self-intersects or gets too narrow
7. `base_points + outward * variation[:, None]` - actually push each point. `variation[:, None]` turns the 1D array into a column so it broadcasts with the 2D outward vectors
8. `_enforce_no_self_intersection()` - check if the result crosses itself. if it does, scale down the variation by 0.8x repeatedly (up to 8 times) until it doesnt
9. add z coordinates and return

**key insight:** `variation[:, None]` is a numpy broadcasting trick. variation is shape (100,) and outward is shape (100, 2). the `[:, None]` reshapes variation to (100, 1), which then multiplies element-wise with each column of outward. this scales each outward vector by its corresponding variation amount.

---

## 8. angle warping

**file:** `setup/track (3).py` lines 89-102
**why it matters:** this is what makes the track feel like a real racetrack with straights and turn complexes instead of a smooth oval.

```python
def _warp_angles(self, base_angles):
    weights = np.ones_like(base_angles)
    harmonics = max(1, self.angle_warp_harmonics)
    strength = max(0.0, self.angle_warp_strength)
    if strength > 0:
        for harmonic in range(1, harmonics + 1):
            amplitude = strength * self._rng.uniform(0.5, 1.0) / harmonic
            phase = self._rng.uniform(0, 2 * np.pi)
            weights += amplitude * np.sin(harmonic * base_angles + phase)
    weights = np.clip(weights, 0.15, None)
    cumulative = np.cumsum(weights)
    cumulative *= (2 * np.pi / cumulative[-1])
    return cumulative
```

**the idea:** instead of placing 100 points evenly around the track, we redistribute them so some sections have more points (denser = tighter curves) and some have fewer (sparser = longer straights).

- `weights = np.ones_like(base_angles)` - start with uniform weights (all 1.0)
- the harmonic loop adds sine waves to the weights. where the sine is positive, weight goes up (more points = tighter turn). where its negative, weight goes down (fewer points = longer straight)
- `amplitude / harmonic` means higher harmonics contribute less, keeping the overall shape smooth
- `np.clip(weights, 0.15, None)` prevents weights from going to zero or negative, which would cause points to stack up or reverse direction
- `np.cumsum(weights)` converts weights to cumulative positions
- `cumulative *= (2 * np.pi / cumulative[-1])` rescales so the total still wraps around to exactly 2pi

---

## 9. the training setup

**file:** `train.py` lines 236-259
**why it matters:** these are the knobs that control how the agent learns. wrong settings here and training either fails or takes forever.

```python
policy_kwargs = dict(net_arch=[128, 128])

model = PPO(
    policy="MlpPolicy",
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
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=logs_dir,
    device="cuda",
)
```

**what each line actually does:**

- `policy="MlpPolicy"` - use a feedforward neural network (not CNN, not LSTM). takes the 13 observation numbers, passes through hidden layers, outputs 2 action values. "Mlp" = multi-layer perceptron
- `net_arch=[128, 128]` - two hidden layers with 128 neurons each. the network is: 13 -> 128 -> 128 -> 2 (actions) with a parallel 128 -> 128 -> 1 (value) head
- `n_steps=4096` - collect 4096 steps of experience before updating the network. during these steps, the network does NOT change. its like taking notes for a while, then studying all your notes at once
- `batch_size=128` - when studying the notes, look at 128 at a time. 4096 / 128 = 32 gradient updates per epoch
- `n_epochs=10` - study the same notes 10 times. 32 * 10 = 320 gradient updates per rollout. reusing data is sample efficient but risks overfitting to stale data
- `gamma=0.995` - a reward 100 steps from now is worth 0.995^100 = 60.6% of its face value. high gamma means the agent cares about the long term
- `clip_range=0.2` - PPO's key feature. the new policy can be at most 20% different from the old policy for any given action. prevents catastrophic updates
- `ent_coef=0.02` - gives a small bonus for being uncertain (exploring). without this, the agent might commit to a mediocre strategy too early and never discover better ones
- `device="cuda"` - run neural network computations on the GPU. the environment still runs on CPU (pybullet is CPU-only)

---

## 10. environment wrapping

**file:** `train.py` lines 53-78
**why it matters:** without proper wrapping, you get no training metrics and SB3 refuses to run.

```python
def _make_env():
    env = SelfDrivingCarEnv(config_path=config_path, render_mode=render_mode)
    if log_dir is not None:
        env = Monitor(env, log_dir)
    else:
        env = Monitor(env)
    return env

env_fns = [_make_env]
env = DummyVecEnv(env_fns)
```

**the wrapping chain:**

1. `SelfDrivingCarEnv(...)` - the raw environment
2. `Monitor(env)` - wraps the environment to record episode rewards and lengths. every time an episode ends, Monitor logs the total reward and number of steps. this is how `ep_rew_mean` gets computed. without Monitor, you have zero visibility into training progress
3. `DummyVecEnv([_make_env])` - wraps in a "vectorized" environment. SB3 requires this even for a single env. the list `[_make_env]` contains factory functions (not environments) - this is important because DummyVecEnv calls the function itself to create the env. if you passed the env directly instead of a function, it would break

**why a factory function:** `DummyVecEnv` expects callables (functions that create environments) not environment instances. this is because if you used `SubprocVecEnv` for parallel training, each subprocess needs to create its OWN environment instance. the factory pattern makes this possible.

---

## 11. the spawn position calculation

**file:** `environment.py` lines 194-244
**why it matters:** if the car spawns in the wrong place or facing the wrong direction, every episode starts broken.

```python
inner_pt = self.track.inner_points[0]
outer_pt = self.track.outer_points[0]

spawn_x = (inner_pt[0] + outer_pt[0]) / 2.0
spawn_y = (inner_pt[1] + outer_pt[1]) / 2.0
spawn_z = self.config['spawn']['position'][2]
spawn_pos = [spawn_x, spawn_y, spawn_z]

inner_next = self.track.inner_points[1]
outer_next = self.track.outer_points[1]

next_x = (inner_next[0] + outer_next[0]) / 2.0
next_y = (inner_next[1] + outer_next[1]) / 2.0

dx = next_x - spawn_x
dy = next_y - spawn_y
heading = np.arctan2(dy, dx)
spawn_orn = p.getQuaternionFromEuler([0, 0, heading])
```

**the logic:**

1. take the first point of each boundary (segment 0)
2. average them to find the center of the track at that point
3. do the same for segment 1
4. the direction from segment 0's center to segment 1's center is the track direction
5. convert that direction to a heading angle with `arctan2`
6. convert the heading to a quaternion (pybullet's rotation format)

this means the car always spawns centered on the track, facing the direction the track goes. it adapts automatically to any track shape - no hardcoded coordinates needed.

---

## 12. the importlib trick

**file:** `environment.py` lines 33-55
**why it matters:** without this, python cannot import files with spaces in their names.

```python
def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_track_module = _import_from_path("track", os.path.join(_THIS_DIR, "setup", "track (3).py"))
Track = _track_module.Track
```

normal python imports use the file name as the module name. `import track` looks for `track.py`. but our file is `track (3).py` which has spaces and parentheses - both illegal in python identifiers. you cant write `import track (3)`.

`importlib` bypasses this:
1. `spec_from_file_location("track", "setup/track (3).py")` - create a module spec with the name "track" pointing to the file, regardless of filename
2. `module_from_spec(spec)` - create an empty module object
3. `spec.loader.exec_module(module)` - run the file's code inside the module object, populating it with classes and functions
4. `_track_module.Track` - pull out the Track class we need

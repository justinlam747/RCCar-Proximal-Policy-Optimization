# reward function iteration log

this document tracks each iteration of the reward function and the reasoning behind changes.

---

## iteration 1: original reward (baseline)

**reward structure:**
- survival bonus: +0.5 per step on track
- forward velocity: +1.0 * velocity
- slow penalty: -1.0 if speed < 0.3
- edge penalty: -0.5 if too close to edges
- off track: -50

**observed behavior:**
- car moved forward but crashed at 1/4 of the track
- movement was not smooth, car was jerky
- agent learned to go fast to maximize velocity reward

**problem identified:**
- velocity reward was too dominant
- agent was incentivized to go fast, which caused crashes
- survival bonus was too small compared to speed bonus

---

## iteration 2: survival-focused reward

**changes made:**
- survival bonus: increased from +0.5 to +1.0 per step
- centering bonus: added up to +0.5 for staying in middle of track
- forward velocity: reduced multiplier from 1.0 to 0.1
- removed slow penalty
- off track: increased from -50 to -100
- max episode steps: increased from 1000 to 3000

**reasoning:**
- prioritize survival over speed
- centering helps the car stay safe on the track
- reduced speed incentive to prevent reckless driving
- more time to complete a lap

**observed behavior:**
- car stopped moving and just sat in place
- agent found the optimal strategy was to just survive without moving
- oscillating/wiggling in place to avoid the track detection

**problem identified:**
- no penalty for staying still
- agent exploited the survival reward by not moving

---

## iteration 3: added wiggle penalty

**changes made:**
- added penalty: -0.3 if forward velocity < 0.5
- increased velocity multiplier from 0.1 to 0.2

**reasoning:**
- force the agent to maintain minimum forward speed
- prevent the wiggling/oscillating exploit

**observed behavior:**
- car moved forward and backward repeatedly
- agent learned to go forward (get speed bonus) then backward
- this way it stayed in place but appeared to be moving

**problem identified:**
- forward velocity is instantaneous, not cumulative
- going forward then backward cancels out but still gets rewards

---

## iteration 4: progress-based reward

**changes made:**
- replaced velocity reward with angle progress reward
- track angle around the track center
- reward = angle_change * 10.0
- positive angle change = correct direction
- negative angle change = going backwards = penalty

**reasoning:**
- cant hack progress because going back and forth cancels out
- only sustained movement in one direction gets rewarded
- measures actual progress around the lap

**observed behavior:**
- unknown (not yet tested)

**problem identified:**
- needed to research proven reward ratios from existing projects

---

## iteration 5: research-based reward (current)

**sources consulted:**
- openai gym carracing environment documentation
- research papers on rl reward shaping
- github projects for racing rl agents

**key learnings from research:**
1. use progress-based rewards, not velocity-based
2. small time penalty prevents sitting still
3. large off-track penalty (10x larger than positive rewards)
4. clip speed/progress rewards to prevent reckless behavior
5. lap completion bonus for sparse long-term rewards

**final reward structure:**
- time penalty: -0.1 per step (encourages efficiency)
- progress reward: +1.0 max per step (clipped)
- off track penalty: -100
- lap completion bonus: +100

**ratios explained:**
- time penalty is small so it doesnt dominate (-0.1 vs +1.0)
- progress is clipped to prevent going too fast
- off track penalty is 100x the step reward
- lap bonus equals 100 steps of max progress

**expected behavior:**
- agent must make progress to overcome time penalty
- going too fast doesnt give extra reward (clipped at 1.0)
- off track is heavily punished
- completing a lap gives big bonus

**observed behavior:**
- car spiraled into the wall after 97 steps
- mean reward was -111 (negative) and -221 during training
- the car turned inward constantly

**problem identified:**
- angle-based progress rewards turning toward the center
- the track may go clockwise but we assumed counterclockwise
- spiraling inward changes the angle in the "rewarded" direction
- angle from origin doesnt account for track shape

---

## iteration 6: track tangent velocity (current)

**changes made:**
- removed angle-based progress entirely
- calculate track tangent direction at cars position
- reward velocity projected onto track tangent
- added survival bonus back (+0.5)
- kept time penalty (-0.1)
- kept off track penalty (-100)

**reward structure:**
- survival bonus: +0.5 per step
- time penalty: -0.1 per step
- velocity along track: up to +1.0 (clipped)
- off track: -100

**reasoning:**
- velocity along track tangent only rewards movement in correct direction
- spiraling or turning doesnt help because tangent points along the track
- the car must be aligned with the track to get reward
- survival bonus ensures positive reward for staying on track

**expected behavior:**
- car should follow the track because thats where the tangent points
- turning away from track direction reduces reward
- going backwards gives negative reward
- staying on track while moving forward maximizes reward

**observed behavior:**
- car oversteered constantly
- hugged the walls
- crashed after going around some of the track

**problem identified:**
- no penalty for sharp steering
- no incentive to stay centered (away from walls)
- velocity reward was too high relative to survival

---

## iteration 7: priority-based reward (current)

**changes made:**
- increased survival bonus from +0.5 to +1.0 (highest priority)
- added centering bonus (+0.3) to stay away from walls
- added steering penalty (-0.2 max) to prevent oversteering
- reduced speed bonus to +0.2 max (secondary priority)
- removed time penalty

**reward structure:**
- survival: +1.0 per step (priority 1)
- centering: +0.3 max (prevents wall hugging)
- speed: +0.2 max (priority 2)
- steering penalty: -0.2 max (prevents oversteering)
- off track: -100

**priority ratios:**
- survival (1.0) > centering (0.3) + speed (0.2) + steering (0.2)
- car is always rewarded for staying on track
- additional small rewards for good behavior
- penalty for excessive steering

**expected behavior:**
- car should prioritize survival over speed
- soft steering should be preferred
- car should stay in the middle of the track
- smooth driving should emerge

---

## reward ratio summary table

| version | survival | velocity | progress | time | off track | lap bonus |
|---------|----------|----------|----------|------|-----------|-----------|
| v1      | +0.5     | +1.0x    | none     | none | -50       | none      |
| v2      | +1.0     | +0.1x    | none     | none | -100      | none      |
| v3      | +1.0     | +0.2x    | none     | none | -100      | none      |
| v4      | +1.0     | none     | +10x angle | none | -100    | none      |
| v5      | none     | none     | +20x angle (clipped) | -0.1 | -100 | +100 |
| v6      | +0.5     | track tangent (clipped) | none | -0.1 | -100 | none |

---

## lessons learned

1. **survival rewards alone cause exploit** - agent will find ways to survive without doing the task

2. **velocity rewards are hackable** - going back and forth appears as movement but makes no progress

3. **progress-based rewards work** - measure actual distance/angle traveled, not speed

4. **clipping prevents reckless behavior** - cap the max reward per step

5. **time penalty encourages action** - small negative per step prevents sitting still

6. **ratios matter more than absolute values** - the relative size of rewards determines behavior

7. **off track penalty must be large** - should outweigh any possible positive reward exploitation

8. **sparse rewards help long-term planning** - lap completion bonus guides the agent toward the goal

---

## iteration 8: anti-wiggle with minimum speed (current)

**changes made:**
- reduced total timesteps from 200k to 50k for faster iteration
- forward velocity reward: +0.5 max (main driver, doubled from 0.2)
- survival bonus: reduced from +1.0 to +0.3
- centering bonus: reduced from +0.3 to +0.2
- minimum speed penalty: NEW -0.5 max if speed < 0.3
- steering penalty: reduced from -0.2 to -0.1

**reward structure:**
- forward velocity: +0.5 max (primary reward)
- minimum speed penalty: -0.5 max if too slow (anti-wiggle)
- survival: +0.3 per step
- centering: +0.2 max
- steering penalty: -0.1 max
- off track: -100

**key insight:**
- the wiggle happened because survival was the biggest reward
- car could survive without moving, so it did
- new structure: movement is the main reward
- penalty for going slow forces the car to drive forward
- reduced steering penalty allows necessary turning on oval

**expected behavior:**
- car must maintain speed > 0.3 to avoid penalties
- wiggling gives no forward velocity = penalty
- driving forward gives the biggest reward
- should complete laps instead of sitting still

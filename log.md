# reward function iteration log

## iteration 1: basic reward
- survival: +0.5, velocity: +1.0x, off track: -50
- car crashed at 1/4 of the track, too fast

## iteration 2: survival focused
- survival: +1.0, centering: +0.5, velocity: +0.1x, off track: -100
- car stopped moving entirely, exploited survival

## iteration 3: wiggle penalty
- added penalty -0.3 if speed < 0.5, velocity +0.2x
- car went forward/backward repeatedly

## iteration 4: angle progress
- replaced velocity with angle change * 10.0
- not tested yet

## iteration 5: research based
- time penalty: -0.1, progress: +1.0 max (clipped), off track: -100, lap bonus: +100
- based on carracing env and papers

## iteration 5 results
- car spiraled into the wall after 97 steps
- mean reward -111
- angle from origin rewards turning inward, not following track
- spiraling changes angle in the rewarded direction

## iteration 6: track tangent velocity
- removed angle progress, reward velocity along track tangent
- survival +0.5, time -0.1, tangent velocity up to +1.0, off track -100
- car oversteered, hugged walls, crashed partway through

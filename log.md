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

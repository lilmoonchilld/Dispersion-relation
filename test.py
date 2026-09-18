import mpmath as mp
mp.mp.dps = 15
import dispersion
from dispersion import CFG, initialize_branches

# Try generating seeds and seeing how long it takes to find one
import time
dispersion.INITIAL_GUESS_NORMALIZED = [1.2]
t0 = time.time()
print(f"Finding branch...")
dispersion.initialize_branches(CFG)
print(f"Time: {time.time()-t0}")

import re

with open('dispersion.py', 'r') as f:
    content = f.read()

# Replace RootRecord definition
old_root_record = """@dataclass
class RootRecord:
    m: int
    omega: mp.mpc
    omega_norm: mp.mpc
    sign: int
    radial_k: mp.mpf | None
    n: int | None
    K2: mp.mpc
    k_eff2_r0: mp.mpc | None
    Lambda: mp.mpc
    residual: mp.mpf"""

new_root_record = """@dataclass
class RootRecord:
    m: int
    omega: mp.mpc
    omega_norm: mp.mpc
    omega_real: mp.mpf
    omega_imag: mp.mpf
    sign: int
    n: int | None
    mode_family: str
    radial_behavior: str
    K2: mp.mpc
    Lambda: mp.mpc
    k_eff2_r1: mp.mpc | None
    k_eff2_r0: mp.mpc | None
    k_eff2_r2: mp.mpc | None
    radial_k_r0: mp.mpc | None
    radial_nodes: int | None
    residual: mp.mpf"""

content = content.replace(old_root_record, new_root_record)

with open('dispersion.py', 'w') as f:
    f.write(content)
print("Replaced RootRecord.")

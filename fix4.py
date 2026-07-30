with open("swmhd.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "f\"$\hat" in line or "f\"$\gamma" in line or "f\"$\Omega" in line:
        lines[i] = line.replace("f\"$", "rf\"$\\").replace("}\n", "}\\n")

with open("swmhd.py", "w") as f:
    f.writelines(lines)

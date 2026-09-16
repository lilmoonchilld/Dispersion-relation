with open("swmhd_variable_depth.py", "r") as f:
    content = f.read()

# Replace p.r_grid_cache with r_grid explicitly in function signature to avoid state leakage
old_sig = "def assign_branches_global(modes, m, p: Params, N_max=10):"
new_sig = "def assign_branches_global(modes, m, r_grid, p: Params, N_max=10):"
content = content.replace(old_sig, new_sig)

old_call = "L_in, L_out = calculate_boundary_localization(mode[\"eta\"], p.r_grid_cache, p)"
new_call = "L_in, L_out = calculate_boundary_localization(mode[\"eta\"], r_grid, p)"
content = content.replace(old_call, new_call)

with open("swmhd_variable_depth.py", "w") as f:
    f.write(content)

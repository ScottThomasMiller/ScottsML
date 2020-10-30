
# simple pendulum simulator:
time python3  -u spgen.py  \
  --split_ratio 0.1 --save_dir save \
  --tspan 0 10  --dsr 0.1  --trajectories 1000 \
  --hamiltonian="(p**2)/2.0 + (1.0 - cos(q))" --state_symbols q p \
  --num_bodies 1 --name "spdata"



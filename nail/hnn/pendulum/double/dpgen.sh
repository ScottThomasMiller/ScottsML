
time python3  -u dpgen.py \
  --hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))" \
  --state_symbols 'q1' 'q2' 'p1' 'p2' --save_dir save --split_ratio 0.1 \
  --tspan 0 10 --dsr 0.0001  --trajectories 100 --name "dp-dataset-dsr1e-04-tspan0_10-traj100-xy-p1pi"


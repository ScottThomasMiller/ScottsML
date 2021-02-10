
time python3  -u dpgen.py \
  --hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))" \
  --state_symbols 'q1' 'q2' 'p1' 'p2' --save_dir save --split_ratio 0.1 \
  --tspan 0 100 --dsr 0.01  --trajectories 125 --name "dp-dataset-dsr1e-02-tspan0_100-traj125-xy-p1pi"


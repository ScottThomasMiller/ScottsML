
time python3  -u dpgen.py \
  --hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))" \
  --state_symbols 'q1' 'q2' 'p1' 'p2' \
  --tspan 0 1000 --dsr 0.1  --trajectories 1000 --name "dp-dataset-dsr1e-01-tspan0_1000-traj1000-xy-p1pi-test"


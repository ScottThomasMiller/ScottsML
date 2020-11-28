
time python3  -u dpmain2.py \
  --hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))" \
  --epochs 200000 --master_port=11031 --state_symbols 'q1' 'q2' 'p1' 'p2' \
  --early_stop 9.99e-05 --activation_fn Tanh --save_dir save --model baseline  \
  --tspan 0 100 --dsr 0.01  --trajectories 100 --name "dp-dataset-dsr1e-02-tspan0_100-traj125-xy-p1pi" \
  --batch_size 1024 --hidden_dim  512 512  --train_pct 100 --test_pct 4   \

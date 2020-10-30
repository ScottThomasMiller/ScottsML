
# simple pendulum
time python3  -u spmain.py \
  --hamiltonian="(p**2)/2.0 + (1.0 - cos(q))" --state_symbols q p \
  --epochs 16 --master_port=11031 \
  --activation_fn Tanh  --learn_rate 1e-03 \
  --tspan 0 7  --dsr 0.0546875  --trajectories 160 --name "sp-dataset-dsr1e-03-tspan0_7-traj160-xy-ixs" \
  --batch_size 1 --hidden_dim  32 32 --train_pct 100 --test_pct 100 


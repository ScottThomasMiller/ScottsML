python3 dpforecast.py --seed 929233  \
  --epochs 16 \
  --state_symbols 'q1' 'q2' 'p1' 'p2'  \
  --name "dp-dataset-dsr1e-02-tspan0_100-traj125-xy-p1pi" \
  --hamiltonian "-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))" \
  --test_pct 1 \
  --master_port 11571 \
  --activation_fn 'Tanh' \
  --learn_rate 1e-03  \
  --tspan 0 100 \
  --dsr 0.01 \
  --batch_size 1  \
  --hidden_dim 32 32  \
  --train_pct 0 \
  --save_dir ~/github/jupyter/save \
  --load_model lowest_hnn_model_20210212_070031_e7c4b204-6d29-11eb-b6c5-1e00e214aa95.trch
  #--load_model lowest_hnn_model_20210211_211103_8e9f6a4a-6cd7-11eb-ba09-1e00e214aa95.trch
  #--load_model lowest_hnn_model_20210210_171057_d9806ba4-6bec-11eb-8296-5cf3fc21f0d4.trch
  #--load_model lowest_hnn_model_20210209_103415_4481a2da-6aec-11eb-bdeb-000e1e857040.trch
 

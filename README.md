# ETH-DL-Project

Avalanche documentation: [here](https://avalanche-api.continualai.org/en/v0.6.0/index.html)
Start [here](https://avalanche.continualai.org/getting-started/learn-avalanche-in-5-minutes)

# scripts 

Important: to save results, create '/storage' folder in project root directory!

Example script: 

```
cd src 
python train.py --task projected_training \
--epochs 4 \
--num_hidden_layers 3 --hidden_sizes 200 200 200 --activation 'tanh' \
--warm_up_epochs 1 \
--algo SGD \
--plot_losses \
--lr 0.01 \
--loss MSE \
--seed 42      
``` 

Debug mode runs everything only on two batches. To switch to debug mode add '--debug'.

For other parameters see `src/train.py -> arg_parser()`.

# Using WANDB 

Checkout [Quickstart](https://docs.wandb.ai/quickstart). All you need to do is create account and login (wandb login). There are no logs in `--debug` mode. Also, if you want to disable loggin use `wandb offline`. Then go to website and see all the nice graphs :) 
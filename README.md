# ETH-DL-Project

Avalanche documentation: [here](https://avalanche-api.continualai.org/en/v0.6.0/index.html)
Start [here](https://avalanche.continualai.org/getting-started/learn-avalanche-in-5-minutes)

# scripts 

Important: to save results, create '/storage' folder in project root directory!

Example script: 

```
cd src 
python train.py --task projected_training \
--epochs 10 \
--hidden_layers 128 \
--warm_up_epochs 10 \
--algo SGD \
--plot_losses         
``` 

Debug mode runs everything only on two batches. To switch to debug mode add '--debug'.

For other parameters see `src/train.py -> arg_parser()`.
# MAHPPO

PyTorch implementation of the paper: Multi-Agent Collaborative Inference via DNN Decoupling: Intermediate Feature Compression and Edge Learning, which is currently under review.



## Requirements

```
pip install -r requirements.txt
```



## Step 1. 

Train a classification model, and autoencoders at beforehand selected partitioning points.

1. Training of the classification model:

   - change the `default_root` value in `dataset\config.py` to the path saving Caltech-101 dataset in your PC.

   - training the model with the following command

     ```python
     python train_model.py
     ```

2. Training of autoencoders

   - train the autoencoder at a specified partitioning point

     ```python
     python train_ae.py
     ```
     
   - finetune
   
     ```python
     python train_ae.py --finetune
     ```



## Step 2.

Evaluate the latency and energy consumption of inference on Jetson Nano and power monitor.

If you want to implement the evaluation with yourself, remember to set your Jetson Nano with the following commands:

```shell
# select power mode
sudo /usr/sbin/nvpmodel -m 1

# turn off DVFS
sudo jetson_clocks
```

or, you can refer to our evaluated results in `env\data.py`.



## Step 3.

Train the DRL agent making offloading decisions.

```
python train_agent.py
```



## Reference

If you find our code useful for your research, please cite our paper.

```
@article{hao2022multi,
  title={Multi-agent collaborative inference via dnn decoupling: Intermediate feature compression and edge learning},
  author={Hao, Zhiwei and Xu, Guanyu and Luo, Yong and Hu, Han and An, Jianping and Mao, Shiwen},
  journal={IEEE Transactions on Mobile Computing},
  year={2022},
  publisher={IEEE}
}
```

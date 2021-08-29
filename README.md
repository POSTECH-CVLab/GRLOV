# adaptation-on-grasping
Domain adaptation on robot grasping 


## Environment
- environment
```
- Windows10
- nvidia-dirver 461.72
- cuda 11.1
- cuDNN 8.0.5
- pytorch 1.7.1
- pybullet 2.6.4
```
- installation 
```
git clone <url_repo> <dir_name>
conda create -n <env_name> python=3.8
conda activate <env_name> 
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

- robot configuration       
https://github.com/BarisYazici/deep-rl-grasping     
```
pytest test
```

## Algorithm
- DDPG pseudo algorithm
<p align="center">
<img src="demo/ddpg_pseudo.png" width="350px">
</p>

## Demo 
- ddpg (pendulum)
<p align="center">
<img src="demo/pendulum.png" width="350px">
</p>
<p align="center">
<img src="demo/ddpg_pendulum.png" width="400px" height="200px">
</p>

- ddpg (pybullet/gripper)    
  achieved upto 0.9 success rate (among latest 10 trial)
<p align="center">
<img src="demo/ddpg_grasping.gif" width="350px">
</p>  
<p align="center">
<img src="demo/ddpg_grasping_reward.png" width="400px">
</p>
<p align="center">
<img src="demo/ddpg_grasping_success.png" width="400px">
</p>

## Contributors
이주용 김진모 최서영 한주완 조원준 안석준
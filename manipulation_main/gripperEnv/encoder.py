import numpy as np
import sys, os; sys.path.append(os.path.abspath(os.path.join('..', 'DDPG')))
from manipulation_main.common import io_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def embed_state(obs):
    obs = torch.from_numpy(obs[:,:,:3])
    obs = obs.unsqueeze(0).permute(0, 3, 1, 2)
    obs = Variable(obs).to(device).float()
    return obs

'''
class Encoder(nn.Module):
    """ Vanilla CNN """

    def __init__(self, config,
                 model_dir='./checkpoints'):
        super(Encoder, self).__init__()
        model_dir = os.path.expanduser(model_dir)
        self.model_path = os.path.join(model_dir, 'encoder.pth')
        network = config['network']

        encoders = []
        for layer in network:
            encoders.append(nn.Conv2d(in_channels=layer['in_channel'], 
                                      out_channels=layer['out_channel'], 
                                      kernel_size=layer['kernel_size'], 
                                      stride=layer['stride'], 
                                      padding=layer['padding']))
            encoders.append(nn.ReLU(True))
            encoders.append(nn.MaxPool2d(2, stride=2))
        self.encoder = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.encoder(x)
        return x
'''
        
class AutoEncoder(nn.Module):
    """ Vanilla autoencode r"""

    def __init__(self, config,
                 model_dir='./checkpoints'):
        super(AutoEncoder, self).__init__()
        model_dir = os.path.expanduser(model_dir)
        self.model_path = os.path.join(model_dir, 'encoder.pth')
        config = io_utils.load_yaml(config['sensor']['encoder_dir'])
        network = config['network']

        encoders = []
        for layer in network:
            encoders.append(nn.Conv2d(in_channels=layer['in_channel'], 
                                      out_channels=layer['out_channel'], 
                                      kernel_size=layer['kernel_size'], 
                                      stride=layer['stride'], 
                                      padding=layer['padding']))
            encoders.append(nn.ReLU(True))
            encoders.append(nn.MaxPool2d(2, stride=2))
        self.encoder = nn.Sequential(*encoders)

        decoders = []
        for layer in reversed(network):
            decoders.append(nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True))
            decoders.append(nn.ConvTranspose2d(in_channels=layer['out_channel'], 
                                               out_channels=layer['in_channel'], 
                                               kernel_size=layer['kernel_size'],  
                                               stride=layer['stride'],
                                               padding=layer['padding']
                            ))
            decoders.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*decoders)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = x.detach().cpu().numpy()
        return x

    def load_weight(self, model_path=None):
        if model_path is None:
            self.encoder.load_state_dict(torch.load(self.model_path))
        else:
            self.encoder.load_state_dict(torch.load(model_path))
        return

    def save_weight(self, model_path=None):
        if model_path is None:
            torch.save(self.encoder.state_dict(), self.model_path)
        else:
            torch.save(self.encoder.state_dict(), model_path)
        return

    def train(self, inputs, targets, batch_size, epochs, model_dir):
        raise NotImplemented

if __name__ == '__main__':
    import pybullet as p
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from itertools import count
    
    import gym
    import manipulation_main
    from manipulation_main.common import io_utils

    config = io_utils.load_yaml("config/gripper_grasp.yaml")
    env = gym.make("gripper-env-v0", config=config)
    total_timestep=100_000

    img_h, img_w, img_c = 64, 64, 5
    action_shape = env.action_space.shape
    action_min = env.action_space.low
    action_max = env.action_space.high
    model = AutoEncoder(config).to(device)
    
    '''
    num_epochs = 30
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=1e-3,
                                weight_decay=1e-5)
    for epoch in range(num_epochs):
        total_loss, total_obs = 0, 1
        last_obs = env.reset()
        with tqdm(total=total_timestep) as pbar:
            for t in count():
                # update progress bar
                pbar.n = t
                pbar.refresh()

                if t >= total_timestep:
                    break
                action = np.random.rand(5)
                action = action * (action_max - action_min) + action_min
                obs, reward, done, _ = env.step(action)
                
                # ===================imge========================= 
                obs = torch.from_numpy(obs[:,:,:3])
                obs = obs.unsqueeze(0).permute(0, 3, 1, 2)
                obs = Variable(obs).to(device).float()
                # ===================forward=====================
                output = model(obs)
                loss = criterion(output, obs)
                total_loss += loss.data
                total_obs += 1
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if done:
                    obs = env.reset()
                    print(f'epoch [{epoch+1}/{num_epochs}], loss:{total_loss/total_obs:.4f}')
                    #print(reward)
                last_obs = obs
        model.save_weight()
    '''

    #plt.ion()
    model.load_weight("./checkpoints/encoder_000018.pth")
    while True:
        action = np.random.rand(5)
        action = action * (action_max - action_min) + action_min
        obs, reward, done, _ = env.step(action)
        obs = embed_state(obs)        
        emb = model.encode(obs)
        
        print(emb); break
        #plt.imshow(emb[0][0], cmap='gray')
        #plt.show()
        #plt.pause(0.1) 



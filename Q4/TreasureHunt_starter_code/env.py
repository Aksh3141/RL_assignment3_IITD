import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pdb 
from grid import Grid 
from PIL import Image
import imageio
from tqdm import tqdm



UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class TreasureHunt(gym.Env):

    def __init__(self, locations):
        
        self.locations = locations

        #the grid size
        self.n = 10
        self.num_treasures = len(self.locations['treasure'])
        self.num_states = self.n*self.n*(2**self.num_treasures)
        self.num_actions = 4
        self._action_delta = [[0,1],[0,-1],[-1,0],[1,0]]
        self.action_name = ['up','down','left','right']

        #the observation and action space
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

        #the 0th state
        self.state = 0

        #the treasure indicator
        self.treasure_from_index, self.index_from_treasure = self._get_treasure_indicator()

        #get transition matrix
        self.T = self._generate_tmatrix()

        #the reward matrix
        self.reward = self._generate_reward()
    
    def reset(self):
        self.state = np.random.randint(self.num_states)
        return self.state
    
    def locations_from_state(self, state):
        
        treasure_index = state // 100
        treasure_indicator = self.treasure_from_index[treasure_index]
        treasure_locations = []
        for i in range(self.num_treasures):
            if(treasure_indicator[i] == '1'):
                treasure_locations.append(self.locations['treasure'][i])
        ship_location = state % 100
        ship_location = (ship_location // 10, ship_location % 10)
        return ship_location, treasure_locations 

    def _get_treasure_indicator(self):
        
        treasure_indicator = []
        treasure_index = dict()
        for n in range(2**self.num_treasures):
    
            treasure_indicator_n = bin(n)[2:][::-1]
            treasure_indicator_n = treasure_indicator_n + ''.join(['0' for i in range(self.num_treasures - len(treasure_indicator_n))])
            treasure_indicator.append(treasure_indicator_n)

            treasure_index[treasure_indicator_n] = n 
        return treasure_indicator, treasure_index

    
    def _get_pos_ts(self, x, y):

        out = np.arange(0, 2**self.num_treasures)
        if((x,y) in self.locations['treasure']):
            ind = self.locations['treasure'].index((x,y))
            for i in range(2**self.num_treasures):
                tind = self.treasure_from_index[i]
                if(tind[ind] == '1'):
                    tind = list(tind)
                    tind[ind] = '0'
                    tind = ''.join(tind)
                    out[i] = self.index_from_treasure[tind]
        return out

    def step(self, action):
        reward = self.reward[self.state]
        next_state = self.T[self.state, action]
        self.state = np.random.multinomial(1, next_state).nonzero()[0][0]
        return self.state, reward 
    
    def is_land(self, x, y):
        return ((x,y) in self.locations['land'])

    
    def _get_grid_locations(self, state_id):

        locations = dict()
        treasure_index = state_id // (self.n*self.n)
        state_id = state_id % (self.n*self.n)

        x, y = state_id // self.n, state_id % self.n 

        locations['pirate'] = self.locations['pirate']
        locations['fort'] = self.locations['fort']
        locations['ship'] = [(x,y)]
        locations['land'] = self.locations['land']
        locations['treasure'] = []
        treasure_indicator = self.treasure_from_index[treasure_index]
        for i in range(self.num_treasures):
            # Removed the ship position check so treasures are always visible
            if(treasure_indicator[i] == '0'):
                continue
            locations['treasure'].append(self.locations['treasure'][i])
        return locations

    def render(self, state_id = None, path = 'state.jpeg', return_image = True):
        if(state_id is None):
            state_id = self.state
        locations = self._get_grid_locations(state_id)
        grid = Grid(locations)
        if(return_image):
            image = grid.show(return_image = True)
            return image

    def visualize_policy_execution(self, policy, path = 'output.gif'):
        self.state = (2**self.num_treasures - 1)*self.n*self.n
        images = [self.render(return_image = True)]
        for i in tqdm(range(100)):
            action = np.argmax(policy[self.state])
            self.state, _ = self.step(action)
            images.append(self.render(return_image = True))
        
        pil_images = [Image.fromarray(arr.astype('uint8')) for arr in images]
        imageio.mimsave(path, pil_images, duration=2)
    
    def get_policy_rewards(self, policy):
        self.state = (2**self.num_treasures - 1)*self.n*self.n
        rewards = []
        for i in range(100):
            action = np.argmax(policy[self.state])
            self.state, reward = self.step(action)
            rewards.append(reward)
        
        return np.array(rewards)
    



    def visualize_policy(self, policy, path = 'policy_vis.png'):

        for i in range(2**self.num_treasures):
            state_id = i*self.n*self.n
            policy_i = policy[state_id: state_id + self.n*self.n]
            locations = self._get_grid_locations(state_id)
            del locations['ship']
            policy_i = policy_i.argmax(-1).reshape(self.n, self.n)
            
            for j in range(4):
                policy_ij = (policy_i == j).nonzero()
                policy_ij = [(x,y) for x,y in zip(policy_ij[0], policy_ij[1])]
                locations[f'{self.action_name[j]}_arrow'] = policy_ij
            
            grid = Grid(locations)
            pathi = path.split('.')
            pathi = pathi[:-1] + [f"_{i}"] + [pathi[-1]]
            pathi = '.'.join(pathi)
            grid.show(pathi)


    def _generate_tmatrix(self):

        T = np.zeros((2**self.num_treasures, self.n, self.n, self.num_actions, 2**self.num_treasures, self.n, self.n))
        for x in range(self.n):
            for y in range(self.n):

                #if wall then ignore
                if(self.is_land(x,y)):
                    for i in range(2**self.num_treasures):
                        T[i,x,y,:,i,x,y] = 1
                    continue
                

                #it there is treasure at x,y then change the state
                pos_ts = self._get_pos_ts(x,y)

                #iterate over actions
                for a in range(4):

                    #the prob to distribute
                    prob = 1

                    #find the new state
                    nx = x + self._action_delta[a][0]
                    ny = y + self._action_delta[a][1]

                    #is valid?
                    if(nx < self.n and nx >= 0 and ny < self.n and ny >= 0):
                        if(not self.is_land(nx,ny)):
                            for i in range(2**self.num_treasures):
                                T[i,x,y,a,pos_ts[i],nx,ny] = 0.9
                            prob -= 0.9
                    

                    #now to distribute find all valid states
                    valid_states = [(x,y)]
                    for a_v in range(4):

                        #ignore the desired stated
                        if(a_v == a):
                            continue

                        nx = x + self._action_delta[a_v][0]
                        ny = y + self._action_delta[a_v][1]
                        #if valid append
                        if(nx < self.n and nx >= 0 and ny < self.n and ny >= 0):
                            if(self.is_land(nx,ny)):
                                continue
                            valid_states.append((nx, ny))
                    
                    #distribute the probability
                    for (nx,ny) in valid_states:
                        for i in range(2**self.num_treasures):
                            T[i,x,y,a,pos_ts[i],nx,ny] = prob / len(valid_states)

        T[:,self.n-1,self.n-1,:] = 0
        for i in range(2**self.num_treasures):
            T[i,self.n-1,self.n-1,:,i,self.n-1,self.n-1] = 1   

        T = T.reshape(self.num_states, 4, -1)
        return T
    
    def _generate_reward(self):

        reward = np.zeros((2**self.num_treasures,self.n,self.n)) - 0.1
        for i, tloc in enumerate(self.locations['treasure']):
            for j, tind in enumerate(self.treasure_from_index):
                if(tind[i] == '1'):
                    reward[j,tloc[0],tloc[1]] = 2
    
        for ploc in self.locations['pirate']:
            reward[:,ploc[0],ploc[1]] = -1
        
        floc = self.locations['fort'][0]
        reward[:,floc[0], floc[1]] = 0.01
        return reward.reshape(-1)
    

class TreasureHunt_v2:

    def __init__(self, num_treasures = 2, num_pirates = 2, num_lands = 10, locations = None):

        #save the args 
        self.num_lands = num_lands
        self.num_pirates = num_pirates
        self.num_treasures = num_treasures

        if(locations is None):
            locations = dict()
            loc = self.sample_random_locations(self.num_treasures, self.num_pirates, self.num_lands)
            locations['treasure'] = loc[:self.num_treasures]
            locations['pirate'] = loc[self.num_treasures: self.num_treasures + self.num_pirates]
            locations['land'] = loc[self.num_pirates + self.num_treasures : ]
            locations['ship'] = [(0,0)]
            locations['fort'] = [(9,9)]

        #make the grid for the pirate and land
        self.land_spatial_state = np.zeros((10,10))
        for x,y in locations['land']:
            self.land_spatial_state[x,y] = 1
        self.pirate_spatial_state = np.zeros((10,10))
        for x,y in locations['pirate']:
            self.pirate_spatial_state[x,y] = 1

        #get the model
        self.env = TreasureHunt(locations)
        self.state = self.index_to_spatial(self.env.state)
    
    def render(self, state, path = 'state.jpeg', return_image = True):

        state_id = self.spatial_to_index(state) 
        image = self.env.render(state_id, path, return_image)
        if(return_image):
            return image 
    
    def visualize_policy_execution(self, policy, path = 'output.gif'):
        self.env.visualize_policy_execution(policy, path)

    def visualize_policy(self, policy, path = 'policy_vis.png'):
        self.env.visualize_policy(policy, path)

    def spatial_to_index(self, state):
        treasure_indicator = []
        for (x,y) in self.env.locations['treasure']:
            if(state[2,x,y] == 1):
                treasure_indicator.append('1')
            else:
                treasure_indicator.append('0')
        treasure_indicator = ''.join(treasure_indicator)
        index = self.env.index_from_treasure[treasure_indicator]
        index = 100*index + np.argmax(state[3])
        return index 

    def reset(self):

        loc = self.sample_random_locations(self.num_treasures, self.num_pirates, self.num_lands)
        locations = dict()
        locations['ship'] = [(0,0)]
        locations['fort'] = [(9,9)]
        locations['treasure'] = loc[:self.num_treasures]
        locations['pirate'] = loc[self.num_treasures: self.num_treasures + self.num_pirates]
        locations['land'] = loc[self.num_pirates + self.num_treasures : ]

        #make the grid for the pirate and land
        self.land_spatial_state = np.zeros((10,10))
        for x,y in locations['land']:
            self.land_spatial_state[x,y] = 1
        self.pirate_spatial_state = np.zeros((10,10))
        for x,y in locations['pirate']:
            self.pirate_spatial_state[x,y] = 1

        self.env = TreasureHunt(locations)
        self.env.reset()
        self.state = self.index_to_spatial(self.env.state)
        return self.state

    def index_to_spatial(self, index):
        ship_location, treasure_locations = self.env.locations_from_state(index)
        spatial_state = np.zeros((4, 10, 10))
        spatial_state[0] = self.land_spatial_state
        spatial_state[1] = self.pirate_spatial_state
        for (x,y) in treasure_locations:
            spatial_state[2,x,y] = 1
        spatial_state[3,ship_location[0], ship_location[1]] = 1
        return spatial_state > 0
    
    def step(self, action):

        nstate_id, r = self.env.step(action)
        self.state = self.index_to_spatial(self.env.state)
        return self.state, r 

    def get_policy_rewards(self, policy):
        return self.env.get_policy_rewards(policy)


    def sample_random_locations(self, num_treasures, num_pirates, num_lands):
        
        locations = [] 
        possible_locations = np.arange(1,99).tolist()
        locations_id = np.random.choice(possible_locations, num_treasures, replace = False).tolist()
        locations += locations_id   

        possible_locations = list(set(possible_locations) - set(locations_id))
        locations_id = np.random.choice(possible_locations, num_pirates, replace = False).tolist()
        locations += locations_id   
        
        possible_locations = list(set(possible_locations) - set(locations_id))
        locations_id = np.random.choice(possible_locations, num_lands, replace = False).tolist()
        locations += locations_id  

        loc = []
        for i in locations:
            loc.append((i // 10, i % 10))
        return loc
    
    def get_all_states(self):

        states = []
        for i in range(self.env.num_states):
            states.append(self.index_to_spatial(i))
        return np.stack(states, axis = 0)

# locations = {
#     'ship': [(0,0)],
#     'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),(0,7),(0,8),(0,9),(1,7),(1,8),(2,7)],
#     'fort': [(9,9)],
#     'pirate': [(4,7),(8,5)],
#     'treasure': [(4,0),(1,9)]
# }

# th = TreasureHunt(locations)
# T = th.T
# tt = T.sum(-1)
# pdb.set_trace()

                
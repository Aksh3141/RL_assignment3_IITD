import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config
from copy import copy

class DiscretePortfolioOptEnv(gym.Env): 
    '''
    Portfolio Optimization Problem 

    Instance: Multi-Period Asset Allocation Problem, Dantzing & Infager, 1993 

    The Portfolio Optimization (PO) Problem is a problem that seeks to optimize 
    the distribution of assets in a financial portfolio to with respect to a desired 
    financial metric (e.g. maximal return, minimum risk, etc.). 

    In this particular instance by Dantzing & Infager, the optimizer begins with a 
    quantity of cash and has the opportunity to purchase or sell other assets in each  
    of 10 different investement periods. Each transaction incurs a cost and prices of 
    the assets are subject to change over time. Cash value is consant (price = 1). 
    The objective is to maximize the amount of wealth (i.e. the sum total of asset values)
    at the end of the total investment horizon.

    The episodes proceed by the optimizer deciding whether to buy or sell each asset
    in each time period. The episode ends when either all 10 periods have passed or 
    if the amount of any given asset held becomes negative.  

 
    

    Actions:
        Type: Box (num_assets)
        "asset 1 transaction amount" (idx 0): x in {-2,-1,0,1,2}: Buy (positive) or sell (negative) x shares of asset 1; 


    Reward:
        Change in total wealth from previous period or [-max(asset price of all assets) *  maximum transaction size]
        if an asset quantity becomes negative, at which 
        point the episode ends.

    Starting State:
        Starting amount of cash and wealth and prices. 

    Episode Termination:
        Negative asset quantity or traversal of investment horizon. 
    '''    
    def __init__(self, *args, **kwargs):
        self.num_assets = 5 # Number of assets 
        self.initial_cash = 50 # Starting amount of capital 
        self.step_limit = 10 # Investment horizon
        self.lot_size=2 
        self.holding_limit=[10 for i in range(self.num_assets)]

        self.cash = copy(self.initial_cash)

        #Transaction costs proportional to amount bought 
        # self.buy_cost = np.array([1, 1, 1])
        # self.sell_cost = np.array([1, 1, 1])
        # self.buy_cost = np.array([0.5,0.5,0.25,0.5,0.75])
        # self.sell_cost = np.array([0.5,0.5,0.75,0.5,0.25])
        self.buy_cost = np.array([0.25,0.25,0.25,0.25,0.25])
        self.sell_cost = np.array([0.25,0.25,0.25,0.25,0.25])
        

        # if "prices" in kwargs:
        #     prices=kwargs["prices"]
        #     asset1mean=np.array(prices)
        #     asset1mean=prices.reshape(1,-1)
        # else:  
        #     print("Asset prices not found. Using default prices. Please check your arguments!")
        #     asset1mean = np.array([4, 1, 4, 5 , 6, 6, 6, 6, 6, 6]).reshape(1, -1)  # Up

        # self.asset_price_means = np.vstack([asset1mean])
        # if "variance" in kwargs:
        #     var=kwargs["variance"]
        #     print(f"Setting price variance to {var}")
        # else:
        #     print("Using Deterministic transitions")
        #     var=0
        # self.asset_price_var = np.ones((self.asset_price_means.shape)) * var
        
        # Cash on hand, asset prices, num of shares, time_step
        self.obs_length = 1 + 2 * self.num_assets

        self.observation_space = spaces.Box(-20000, 20000, shape=(self.obs_length,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1*self.lot_size for l in range(self.num_assets)]),high=np.array([1*self.lot_size for l in range(self.num_assets)]),
                          dtype=np.int32)
                
        self.seed()
        self.reset()
        
        
    def _RESET(self):
        self.step_count = 0
        self.asset_prices = self._generate_asset_prices()
        self.holdings = np.zeros(self.num_assets)
        self.cash = copy(self.initial_cash)
        self.state = np.hstack([
            self.initial_cash,
            self.asset_prices[:, self.step_count],
            self.holdings,
            self.step_count,],
            dtype=np.float32)
        return self.state
    
    def _generate_asset_prices(self):
        # asset_prices = np.array([max(0,int(self.np_random.normal(mu, sig))) for mu, sig in 
        #     zip(self.asset_price_means.flatten(), self.asset_price_var.flatten())]
        #     ).reshape(self.asset_price_means.shape)

        mean_asset_prices=[np.array([-i*0.1+3, -0.1*i+8, -0.1*i+4, -0.1*i+5, -0.1*i+1]) for i in range(self.step_limit)]  # Change mean at each time step
        cov_asset_prices=np.array([[ 0.92565536, -0.89439923, -0.4671399 , -0.89535766,  0.07970912],
                    [-0.89439923,  2.6241132 ,  0.84302314,  2.08803254, -0.06652505],
                    [-0.4671399 ,  0.84302314,  0.9864559 ,  1.03902951,  0.18637103],
                    [-0.89535766,  2.08803254,  1.03902951,  2.50151988,  0.13640179],
                    [ 0.07970912, -0.06652505,  0.18637103,  0.13640179,  0.31039514]])
        # cov_asset_prices=1*np.eye(5,5)
        asset_prices = np.array([np.random.multivariate_normal(mean_asset_price, cov_asset_prices) for mean_asset_price in mean_asset_prices])
        asset_prices=np.transpose(asset_prices)
        
        # Zero out negative asset prices and all following prices - implies
        # equity is bankrupt and worthless.
        zero_vals = np.vstack(np.where(asset_prices<0))
        cols = np.unique(zero_vals[0])
        for c in cols:
            first_zero = zero_vals[1][np.where(zero_vals[0]==c)[0].min()]
            asset_prices[c,first_zero:] = 0
        
        return asset_prices.astype(int)
    
    def _STEP(self, action):
        assert self.action_space.contains(action)
    
        asset_prices = self.asset_prices[:, self.step_count].copy()
        
        
        for idx, a in enumerate(action):
            if a == 0:
                continue
            # Sell a shares of asset
            elif a < 0:
                a = np.abs(a)
                if a > self.holdings[idx]:
                    a = self.holdings[idx]
                self.holdings[idx] -= a
                # self.cash += asset_prices[idx] * a * (1 - self.sell_cost[idx])
                self.cash += (asset_prices[idx] - self.sell_cost[idx])*a
                # print(f"sell cost {(asset_prices[idx] - self.sell_cost[idx])*a}")
            # Buy a shares of asset
            elif a > 0:
                if self.holdings[idx]+a<=self.holding_limit[idx]:
                    # purchase_cost = asset_prices[idx] * a * (1 + self.buy_cost[idx])
                    purchase_cost = (asset_prices[idx] + self.buy_cost[idx])*a
                    # print(f"purchase_cost : {purchase_cost}")
                    if self.cash < purchase_cost:
                        a = np.floor(self.cash / ((asset_prices[idx] + self.buy_cost[idx])*a))
                        purchase_cost = (asset_prices[idx] + self.buy_cost[idx])*a
                        # print(f"purchase_cost : {purchase_cost}")
                    self.holdings[idx] += a
                    self.cash -= purchase_cost
                else:
                    pass ##Do nothing if max holdings of the asset is exceeded
                
        # Return total portfolio value at the end of the horizon as reward
        if self.step_count + 1 == self.step_limit: 
            reward = np.dot(asset_prices, self.holdings) + self.cash
        else: 
            reward = 0 
        self.step_count += 1

        # Finish if 10 periods have passed - end of investment horizon 
        if self.step_count >= self.step_limit:
            done = True
        else:
            self._update_state()
            done = False
            
        return self.state, reward, done, {}
    
    def _update_state(self):


        self.state = np.hstack([
            self.cash,
            self.asset_prices[:, self.step_count],
            self.holdings,
            self.step_count,
        ], dtype=np.float32)

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

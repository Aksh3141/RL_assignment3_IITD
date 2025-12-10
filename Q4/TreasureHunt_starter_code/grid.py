
import numpy as np
from PIL import Image 
import cv2
import pdb

class Grid:

    def __init__(self, 
        locations = dict(),
        N=10):
        
        self.B = 3 
        self.color = [0.2588, 0.4039, 0.6980]
        self.brightness = 1.8
        self.grid = []
        self.P = 5
        self.W = 100 
        self.M = N 
        self.N = N
        self.ew = self.W + self.P//2

        self.locations = locations
        self.icons = dict()
        for i in self.locations:
            self.icons[i] = self.load_image(f"images/{i}.png")

        #generate the grid
        self.grid = self.generate_grid()
        
    def load_image(self, path):

        image = Image.open(path)
        image = cv2.resize(np.array(image), (self.W,self.W))
        image = image.astype('float') / 255
        return image
        
    def get_real_coordinates(self, i, j):

        x = j
        y = self.N - 1 - i
        return (x,y)
    
    def put_icon(self, grid, icon):

        gg = icon[:,:,3:]*icon[:,:,:3]
        grid[self.P//2:-1*self.P//2, self.P//2:-1*self.P//2] = gg + (1-icon[:,:,3:])*grid[self.P//2:-1*self.P//2, self.P//2:-1*self.P//2]
        return grid

    def generate_grid(self):
        grid = []

        #iterate over each grid cell
        for i in range(self.M):
            
            #the row
            row_i = []

            #for each row
            for j in range(self.N):
                
                #the i and j th grid cell
                grid_ij = np.ones((self.W+self.P, self.W+self.P, 3))

                #paint with colors  
                for c in range(3):
                    grid_ij[:,:,c] = self.color[c]*(0.7 + (self.brightness - 0.9)/60*(i+30-j))

                #see if any icon should be here
                real_coordinates = self.get_real_coordinates(i,j)
                for k in self.locations:
                    if(real_coordinates in self.locations[k]):
                        grid_ij = self.put_icon(grid_ij, self.icons[k])
                 
                #make white
                grid_ij[:self.P//2,:,:] = 0
                grid_ij[-self.P//2:,:,:] = 0

                grid_ij[:,:self.P//2,:] = 0
                grid_ij[:,-self.P//2:,:] = 0
                
                #append to row_i
                row_i.append(grid_ij)
        
            #append to grid
            grid.append(row_i)
        return grid

    def get_grid(self, x, y):
        grid = self.grid[self.M - y - 1][x]
        return grid 

    def draw_one_step_grid(self, sp, dp, grid, color):
        #print(sp, dp)
        if(sp[0] == dp[0]):

            grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],:] *= 0
            for c in range(3):
                grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],c] = color[c]
            #print(grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],0].shape)
        else:
            grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,:] *= 0
            for c in range(3):
                grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,c] = color[c]
            #print(grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,0].shape)
        
    def draw_one_step(self, sp, dp, color):

        delta_x = dp[0] - sp[0] 
        delta_y = dp[1] - sp[1]
        
        sgrid = self.get_grid(sp[0]-1, sp[1]-1) 
        dgrid = self.get_grid(dp[0]-1, dp[1]-1)
        if(delta_x == delta_y):
            return
        if(delta_x == -1):
            self.draw_one_step_grid((self.ew//2 + 1,0),(self.ew//2 + 1, self.ew//2+1), sgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.ew//2+1,self.W + self.P), dgrid, color)
            
        elif(delta_x == 1):
            self.draw_one_step_grid((self.ew//2+1, 0),(self.ew//2 + 1, self.ew//2 + 1), dgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.ew//2+1, self.W + self.P), sgrid, color)
        elif(delta_y == -1):
            self.draw_one_step_grid((0 ,self.ew//2 + 1),( self.ew//2 + 1, self.ew//2+1), dgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.W + self.P, self.ew//2+1), sgrid, color)
            
        elif(delta_y == 1):
            self.draw_one_step_grid((0, self.ew//2+1),(self.ew//2 + 1, self.ew//2 + 1), sgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.W + self.P,self.ew//2+1), dgrid, color)
            
        return

    def draw_path(self, sequence, color = [0,0,1]):

        start_x, start_y = sequence[0]
        start_grid = self.get_grid(start_x-1,start_y-1)
        start_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2] *= 0
        start_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2,0] += 1
        

        end_x, end_y = sequence[-1]
        end_grid = self.get_grid(end_x-1,end_y-1)
        end_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2] *= 0
        end_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2,1] += 1
        
        for i in range(1,len(sequence)):
            self.draw_one_step(sequence[i-1],sequence[i], color)
        
    
    # def draw_arrow(self, x, y, direction):



    
    def show(self, path = "demo.png", return_image = False):

        grid = [np.concatenate(row_i, axis = 1) for row_i in self.grid]
        grid = np.concatenate(grid, axis = 0)
        grid = np.clip(grid, 0, 1)
        grid_big = np.zeros((grid.shape[0] + 2*self.B, grid.shape[1] + 2*self.B, 3))
        grid_big[:,:,0] = 1
        grid_big[:,:,1] = 1
        grid_big[self.B:-1*self.B,self.B:-1*self.B] = grid

        grid = (grid_big*255).astype('uint8')
        if(return_image):
            return grid
        grid = Image.fromarray(grid)
        grid.save(path)
    
    def clear(self):
        self.grid = self.generate_grid()

# locations = {
#     'ship': [(0,0)],
#     'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),(0,7),(0,8),(0,9),(1,7),(1,8),(2,7)],
#     'fort': [(9,9)],
#     'pirate': [(4,7),(8,5)],
#     'treasure': [(4,0),(1,9)],
#     'arrow': [(3,1)],
# }
# tt = Grid(locations)
# tt.show()
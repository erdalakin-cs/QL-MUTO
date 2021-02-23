import numpy as np
import time
import sys
import random
import math

UNIT = 1   # each unit
Grid_H = 10  # grid height
Grid_W = 10  # grid width

building_number = 20
buildings = [] #red id: 1 r: 125
dBuildings = [] #dark red id: 3 r: 150
yBuildings = [] #yellow id: 2 r: 25
emptyS = []  #Green id: 0 r: 0
seaS = []  #Blue

originx = 4
originy = 4

adimsayisi = 0


#T = [[1,1,1,1,1,1,1,1,2,0], [2,2,2,2,2,1,1,2,2,0], [3,2,2,3,3,3,3,3,2,0], [3,2,2,3,0,0,0,3,2,0], [3,2,2,2,0,0,0,3,2,0], [3,2,2,3,3,3,3,3,2,0], [3,3,2,3,0,0,0,0,2,0], 
#[1,3,3,3,3,3,3,1,1,3], [1,1,1,1,3,3,1,1,1,1], [1,1,1,1,3,3,1,1,1,1], [4,3,3,4,4,4,4,4,2,4] ]

#rewardTable = [[0]*Grid_H]*Grid_W

class Grid(object):
    def __init__(self):
        #super(self).__init__()
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)
        self.list_of_observed = []
        self.last_state_of_drones = []
        self.Grid_Hself = 10
        self.Grid_Wself = 10
        self.Ttemp = [[0]*Grid_H for _ in range(Grid_W)]
        self.T = [[1,1,1,1,1,1,1,1,2,0], [2,2,2,2,2,1,1,2,2,0], [3,2,2,3,3,3,3,3,2,0], [3,2,2,3,0,0,0,3,2,0], [3,2,2,2,0,0,0,3,2,0], [3,2,2,3,3,3,3,3,2,0], [3,3,2,3,0,0,0,0,2,0], 
[1,3,3,3,3,3,3,1,1,3], [1,1,1,1,3,3,1,1,1,1], [1,1,1,1,3,3,1,1,1,1] ]
        self.Ttemp = [[1,1,1,1,1,1,1,1,2,0], [2,2,2,2,2,1,1,2,2,0], [3,2,2,3,3,3,3,3,2,0], [3,2,2,3,0,0,0,3,2,0], [3,2,2,2,0,0,0,3,2,0], [3,2,2,3,3,3,3,3,2,0], [3,3,2,3,0,0,0,0,2,0], 
[1,3,3,3,3,3,3,1,1,3], [1,1,1,1,3,3,1,1,1,1], [1,1,1,1,3,3,1,1,1,1] ]

        self.rewardTable = [[0]*Grid_H for _ in range(Grid_W)]
        self.rewardTableTemp = [[0]*Grid_H for _ in range(Grid_W)]

        self.GreenPoints = 0
        self.LRedPoints = 100
        self.DRedPoints = 150
        self.YellowPoints = 25

        self.GreenPointsTotal = 0
        self.LRedPointsTotal = 0
        self.DRedPointsTotal = 0
        self.YellowPointsTotal = 0

        self._build_Grid()



    def _build_Grid(self):

        self.GreenPointsTotal = 0
        self.YellowPointsTotal = 0
        self.DRedPointsTotal = 0
        self.LRedPointsTotal = 0
        for x in range(Grid_H):
            for y in range(Grid_W):
                if self.T[x][y] == 0: #Green
                    self.rewardTable[x][y] = self.GreenPoints
                    self.rewardTableTemp[x][y] = self.GreenPoints
                    self.GreenPointsTotal += 1#self.GreenPoints
                elif self.T[x][y] == 1: #red
                    self.rewardTable[x][y] = self.LRedPoints
                    self.rewardTableTemp[x][y] = self.LRedPoints
                    self.LRedPointsTotal += 1#self.LRedPoints
                elif self.T[x][y] == 2: #yellow
                    self.rewardTable[x][y] = self.YellowPoints
                    self.rewardTableTemp[x][y] = self.YellowPoints
                    self.YellowPointsTotal += 1# self.YellowPoints    
                elif self.T[x][y] == 3: #dark red
                    self.rewardTable[x][y] = self.DRedPoints
                    self.rewardTableTemp[x][y] = self.DRedPoints
                    self.DRedPointsTotal += 1#self.DRedPoints      


        self.rewardTable[originx][originy] = 0 #base station
        self.rewardTableTemp[originx][originy] = 0 #base station

        

    def reset(self, drone):
        drone.x = originx
        drone.y = originy
        return [originx,originy]



    def lastReset(self):
        for x in range(Grid_H):
            for y in range(Grid_W):
                if self.T[x][y] == -1:
                    self.rewardTable[x][y] = -30


        self.rewardTable[originx][originy] = 0


    def step(self, action, actionCount, drone, list_of_drones):

        if action == 0: #up
            if drone.x > 0:
                drone.x -= 1
        elif action == 1: #down
            if drone.x < Grid_H-1:
                drone.x += 1
            
        elif action == 2: #right
            if drone.y < Grid_W-1:
                drone.y += 1
        elif action == 3: #left
            if drone.y > 0:
                drone.y -= 1

        dist = self.dist_to_closest(drone, list_of_drones)
        done = False
        observed_ = False
        if dist > 3 or actionCount> 3*(drone.d_id+1):
            reward = self.rewardTable[drone.x][drone.y]
            done = True
        else:
            reward = self.rewardTable[drone.x][drone.y]
            if reward == -30:
                observed_ = True
        return [drone.x, drone.y], reward, done, observed_


    def dist_to_closest(self,drone, list_of_drones):
        x1 = drone.x
        y1 = drone.y
        dist = 5000
        for dr in list_of_drones:
            
            if dr.d_id != drone.d_id:

                
                x2 = dr.x
                y2 = dr.y
                y = y2 - y1
                x = x2 - x1
                tempDist = math.sqrt(x*x + y*y)
                
                if tempDist < dist:
                    dist = tempDist

        if dist > 4.50:
            print(dist)
        return dist


#env = Grid()

#print(env.reset()[1])








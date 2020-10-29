import numpy as np
import matplotlib  
import matplotlib.style 
import matplotlib.pyplot as plt
import sys
from map_woCanvas import Grid
from RL_main_woCanvas import QLearningTable
import numpy as np
from collections import defaultdict 
import pandas as pdb
from dataclasses import dataclass
from collections import defaultdict
matplotlib.style.use('ggplot')

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

episode_number = 5000
drone_number = 5
decreasingValue = 0.00025
listOfDrones = []

totalReward = [0]*episode_number
@dataclass
class drone:
    d_id: int
    x: int
    y: int
    epis: int
    path: defaultdict(list)
    totalPoint: int
    pointEachEpis:[0]*episode_number
    pointEachEpisForPrint:[0]*episode_number

def update():

    RL.eps = 1.0

    for episode in range(episode_number):
        env._build_Grid()
        actionCount = 0
        for d_id in range(1,drone_number+1):
            obs = env.reset(listOfDrones[d_id])
            drone = listOfDrones[d_id]
            
            while True:


                action = RL.choose_action(str(obs))

                obs_, reward, done, observed_ = env.step(action,actionCount,listOfDrones[d_id], listOfDrones)
            
                listOfDrones[d_id].pointEachEpis[episode] += reward
                #totalReward[episode] += reward
                actionCount += 1

                RL.learn(str(obs), action, reward, str(obs_))

                #if obs_ not in drone.path[episode]:
                drone.path[episode].append(obs_)

                env.rewardTable[obs_[0]][obs_[1]] = -50
                obs = obs_

                if done:
                    actionCount = 0
                    break

            #First reset Grid then assign the observed cells
            env._build_Grid()
            for location in drone.path[episode]:
                env.rewardTable[location[0]][location[1]] = -150
                totalReward[episode] += env.rewardTableTemp[location[0]][location[1]]
                listOfDrones[d_id].pointEachEpisForPrint[episode] += env.rewardTableTemp[location[0]][location[1]]


    
        if RL.eps >= decreasingValue:
            RL.eps = RL.eps - decreasingValue
        elif RL.eps < decreasingValue:
            RL.eps = 0.0



        
def createLastPositions(T):

    #T = [[1,1,1,1,1,1,1,1,2,0], [2,2,2,2,2,1,1,2,2,0], [3,2,2,3,3,3,3,3,2,0], [3,2,2,3,0,0,0,3,2,0], [3,2,2,2,0,0,0,3,2,0], [3,2,2,3,3,3,3,3,2,0], [3,3,2,3,0,0,0,0,2,0], 
#[1,3,3,3,3,3,3,1,1,3], [1,1,1,1,3,3,1,1,1,1], [1,1,1,1,3,3,1,1,1,1], [4,3,3,4,4,4,4,4,2,4] ]

    Grid_H = 10
    Grid_W = 10
    UNIT = 30

    root = tk.Tk()

    canvas = tk.Canvas(root, bg='white',
                        height=Grid_H * UNIT,
                        width=Grid_W * UNIT)

    root.title("RL-QL | LR = "+ str(RL.lr)+ " N= " + str(drone_number))
    
    # create grids
    for c in range(0, Grid_W * UNIT, UNIT):
        x0, y0, x1, y1 = c, 0, c, Grid_H * UNIT
        canvas.create_line(x0, y0, x1, y1)
    for r in range(0, Grid_H * UNIT, UNIT):
        x0, y0, x1, y1 = 0, r, Grid_W * UNIT, r
        canvas.create_line(x0, y0, x1, y1)

    origin = np.array([15, 15]) 
    # buildings
    x = 0
    y = 0

    for x in range(Grid_H):
        for y in range (Grid_W):

            building_center = origin + np.array([UNIT * x, UNIT*y])
            #print (x,"-",y, ": ",T[x][y])
            if T[x][y] == 0:
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='blue')

            elif T[x][y]== 1:
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='red2')
            elif T[x][y] == 2:
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='yellow')

            elif T[x][y]== 3:
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='red4')

            elif T[x][y] == 4:
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='pink')

            elif T[x][y] == -1:
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='grey')
            
            elif T[x][y] == -2: #birinci Drone
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='black')
            elif T[x][y] == -3: #ikinci Drone
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='purple')
            elif T[x][y] == -4: #ucuncu Drone
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='cyan')

            elif T[x][y] == -5: #besinci Drone
                canvas.create_rectangle(
                    building_center[0] - 15, building_center[1] - 15,
                    building_center[0] + 15, building_center[1] + 15,
                    fill='green')

        # create Ground Station
    origin = np.array([135, 135]) 
    canvas.create_rectangle(
        origin[0] - 15, origin[1] - 15,
        origin[0] + 15, origin[1] + 15,
        fill='snow')
    canvas.pack()
    root.mainloop()
if __name__== "__main__":

    print (tf.VERSION)
    print (sys.version)
    env = Grid()
    for i in range(drone_number+1):
        dr = drone(i,4,4,0,defaultdict(list),0,[0]*episode_number,[0]*episode_number)
        listOfDrones.append(dr)

    RL = QLearningTable(actions=list(range(4)))
    createLastPositions(env.T)
    update()

    totalEffectiveCells = 0
            
    L = []
    totalCoveredCells = 0
    totalCoveredEffectiveCells = 0
    totalPoints = env.DRedPointsTotal*env.DRedPoints + env.LRedPointsTotal*env.LRedPoints + env.GreenPointsTotal*env.GreenPoints + env.YellowPointsTotal*env.YellowPoints
    DR = 0
    LR = 0
    YL = 0
    GR = 0
    file1 = open('resultsQL.txt','w')
    file2 = open('EpisodeResults.txt', 'w')
    file2.write("Episode\tN=1\tN=2\tN=3\tN=4\tN=5\n")
    for i in range(0,episode_number, 1):
        val = str(i+1) #+ "\t" + str(totalReward[i])
        
        for j in range(1,drone_number+1):
            total = 0
            for k in range(1,j+1):
                total += listOfDrones[k].pointEachEpisForPrint[i]
            val = val + "\t"+ str(total)
        val = val + "\n"
        file2.write(val)

    file2.close()
    file3 = open('EpisodeReward.txt', 'w')
    file3.write("Episode\tN=1\tN=2\tN=3\tN=4\tN=5\n")
    for i in range(0,episode_number, 1):
        val = str(i+1) #+ "\t" + str(totalReward[i])
        
        for j in range(1,drone_number+1):
            total = 0
            for k in range(1,j+1):
                total += listOfDrones[k].pointEachEpis[i]
            val = val + "\t"+ str(total)
        val = val + "\n"
        file3.write(val)

    file3.close()
    
    for i in range(1,drone_number+1):

        #print(i,".drone")
        index = np.argmax(np.array(listOfDrones[i].pointEachEpis))
        #print(index)
        #print(listOfDrones[i].pointEachEpis[index])
        #print(listOfDrones[i].path[index])
        #totalPoints += listOfDrones[i].pointEachEpis[index]
        
        for obj in listOfDrones[i].path[index]:
            
            if env.T[obj[0]][obj[1]] >= 0: #Onceki dronun pozisyonunu bozmamak icin
                if env.Ttemp[obj[0]][obj[1]] == 0 or env.Ttemp[obj[0]][obj[1]] == 4: 
                    totalCoveredEffectiveCells += env.GreenPoints
                    GR += 1 #env.GreenPoints
                elif env.Ttemp[obj[0]][obj[1]] == 1: 
                    totalCoveredEffectiveCells += env.LRedPoints
                    LR += 1#env.LRedPoints
                elif env.Ttemp[obj[0]][obj[1]] == 2: 
                    totalCoveredEffectiveCells += env.YellowPoints
                    YL += 1#env.YellowPoints
                elif env.Ttemp[obj[0]][obj[1]] == 3: 
                    totalCoveredEffectiveCells += env.DRedPoints
                    DR += 1#env.DRedPoints
                
                env.T[obj[0]][obj[1]] = -1
                totalCoveredCells += 1

        #Coloring last positions of the drones
        if env.T[obj[0]][obj[1]] == -2:
            if obj[0] -1 > 0 and env.T[obj[0]-1][obj[1]] != -2:
                env.T[obj[0]-1][obj[1]] = -2
            elif obj[0] +1 < env.Grid_Hself and env.T[obj[0]+1][obj[1]] != -2:
                env.T[obj[0]+1][obj[1]] = -2
            elif obj[1] +1 < env.Grid_Wself and env.T[obj[0]][obj[1]+1] != -2:
                env.T[obj[0]][obj[1]+1] = -2
            elif obj[1] -1 > 0 and env.T[obj[0]][obj[1]-1] != -2:
                env.T[obj[0]][obj[1]-1] = -2
        else:
            env.T[obj[0]][obj[1]] = -2
        # if i == 1:
        #     env.T[obj[0]][obj[1]] = -2
        # elif i == 2:
        #     env.T[obj[0]][obj[1]] = -3
        # elif i == 3:
        #     env.T[obj[0]][obj[1]] = -4
        # elif i == 4:
        #     env.T[obj[0]][obj[1]] = -5
#    for i in range(episode_number):
#        print(totalReward[i])

        L = []

        L.append(str(i)+"\t")
        L.append(str(totalCoveredCells)+"\t")
        #L.append(str(totalCoveredEffectiveCells)+"\t")
        L.append(str(round(100*(totalCoveredEffectiveCells/totalPoints),2))+"\t")
        L.append(str(round(100*(DR/env.DRedPointsTotal),2))+"\t")
        L.append(str(round(100*(LR/env.LRedPointsTotal),2))+"\t")
        L.append(str(round(100*(YL/env.YellowPointsTotal),2))+"\t")
        L.append(str(round(100*(GR/env.GreenPointsTotal),2))+"\n")
        


        if (i == 1):
            file1.write("Drones\tCovered Area Rate\tCovered Effective Area Rate\tHigh Risk Covered Rate\tMedium Risk Covered Rate\tLow Risk Covered Rate\tEmpty Area Covered Rate \n")
        file1.writelines(L)
    file1.close()    
    createLastPositions(env.T)
    


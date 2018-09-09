"""
    # mobile environment that update channel and association
    """
import numpy as np
import random
import itertools
import math
from channel import *
from ue_mobility import *
import copy
import numpy as np
import sys

matplotlib.rcParams.update({'font.size': 14})

# defining the number of steps
MAXSTEP = 2000
UE_STEP = 1

N_ACT = 5   #number of actions of a single agent

MAX_UE_PER_GRID = 1 # Maximum number of UE per grid

# relative hight of the BSs to UEs in m assuming plain terrain
H_BS = 10
# min distance between BSs in Grids
MIN_BS_DIST = 2

R_BS = 50
#
BS_STEP = 2


class MobiEnvironment:
    
    def __init__(self, nBS, nUE, grid_n=200, mobility_model = "group", test_mobi_file_name = ""):
        self.nBS = nBS
        self.nUE = nUE
        self.bs_h = H_BS
        
        # x,y boundaries
        self.grid_n = grid_n
        
        [xMin, xMax, yMin, yMax] = [1, self.grid_n, 1, self.grid_n]
        boundaries = [xMin, xMax, yMin, yMax]
#        xBS = np.array([int(xMin +1), int(xMax -1), int(xMin+1), int(xMax-1)])
#        yBS = np.array([int(yMin +1), int(yMax -1), int(yMax-1), int(yMin+1)])
        xBS = np.array([int(xMax/4), int(xMax/4), int(xMax*3/4), int(xMax*3/4)])
        yBS = np.array([int(yMax/4), int(yMax*3/4), int(yMax/4), int(yMax*3/4)])


        self.boundaries = boundaries
        self.mobility_model = mobility_model
        print "mobility model: ", mobility_model, " grid size ", grid_n
        #       bsLoc is 3D, bsLocGrid is 2D heatmap
        #self.bsLoc, self.bsLocGrid = GetRandomLocationInGrid(self.grid_n, self.grid_n, self.nBS, H_BS, MIN_BS_DIST)
        self.initBsLoc = np.array([xBS, yBS, np.ones((np.size(xBS)))*self.bs_h], dtype=int).T
        self.initBsLocGrid = GetGridMap(self.grid_n, self.grid_n, self.initBsLoc[:,:2])
        self.bsLoc = copy.deepcopy(self.initBsLoc)
        self.bsLocGrid = copy.deepcopy(self.initBsLocGrid)

#        self.ueLoc, self.ueLocGrid = GetRandomLocationInGrid(self.grid_n, self.grid_n, self.nUE)
        self.ueLoc = []
        self.ueLocGrid = []
        
        self.mm = []
        
        #mobility trace used for testing
        self.test_mobi_trace = []

        if self.mobility_model == "random_waypoint":
            self.mm = random_waypoint(nUE, dimensions=(self.grid_n, self.grid_n), velocity=(1, 1), wt_max=1.0)
        elif self.mobility_model == "group":
#            self.mm = tvc([10,10,10,10], dimensions=(self.grid_n, self.grid_n), velocity=(1, 1.), aggregation=[0.5,0.2], epoch=[1000,1000])
            self.mm = reference_point_group([10,10,10,10], dimensions=(self.grid_n, self.grid_n), velocity=(0, 1), aggregation=0.8)
            for i in range(200):
                next(self.mm)
                i += 1 #repeat in reset
        elif self.mobility_model == "in_coverage":
            self.ueLoc, self.ueLocGrid = GetRandomLocationInCellCoverage(self.grid_n, self.grid_n, R_BS,  self.bsLoc, self.nUE)
        elif self.mobility_model == "read_trace":
            print "testing with mobility trace ", test_mobi_file_name
            assert test_mobi_file_name
            self.ueLoc_trace = np.load(test_mobi_file_name)
            
            self.ueLoc = self.ueLoc_trace[0]
            self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        
        else:
            sys.exit("mobility model not defined")

        if (self.mobility_model == "random_waypoint") or (self.mobility_model == "group"):
            positions = next(self.mm)
            #2D to 3D
            z = np.zeros((np.shape(positions)[0],0))
            self.ueLoc =  np.concatenate((positions, z), axis=1).astype(int)
            self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
     
        self.channel = LTEChannel(self.nUE, self.nBS, self.boundaries, self.ueLoc, self.bsLoc)
        self.association = self.channel.GetCurrentAssociationMap(self.ueLoc)

        
        self.action_space_dim = N_ACT**self.nBS
        self.observation_space_dim = self.grid_n * self.grid_n * (nBS + 1) * MAX_UE_PER_GRID

        self.state = np.zeros((nBS + 1, self.grid_n, self.grid_n ))
        self.step_n = 0
    

    def SetBsH(self, h):
        self.bs_h = h
    
    
    def reset(self):
        #         Get random locations for bs and ue
        #self.bsLoc, self.bsLocGrid = GetRandomLocationInGrid(self.grid_n, self.grid_n, self.nBS, H_BS, MIN_BS_DIST)

        self.bsLoc = copy.deepcopy(self.initBsLoc)
        self.bsLocGrid = copy.deepcopy(self.initBsLocGrid)
        
        if (self.mobility_model == "random_waypoint") or (self.mobility_model == "group"):
            positions = next(self.mm)
            #2D to 3D
            z = np.zeros((np.shape(positions)[0],0))
            self.ueLoc =  np.concatenate((positions, z), axis=1).astype(int)
            self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        elif self.mobility_model == "read_trace":
            print "reseting mobility trace "
            self.ueLoc = self.ueLoc_trace[0]
            self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        else:
            self.ueLoc, self.ueLocGrid = GetRandomLocationInCellCoverage(self.grid_n, self.grid_n, R_BS,  self.bsLoc, self.nUE)

        #       reset channel
        self.channel.reset(self.ueLoc, self.bsLoc)
        #       reset association
        self.association = self.channel.GetCurrentAssociationMap(self.ueLoc)
        
        self.state[0] = self.bsLocGrid
        self.state[1:] = self.association
        
        # self.ueLocGrid = np.sum(self.association, axis = 0)
        # print np.array_equal(np.sum(self.association, axis = 0),  self.ueLocGrid )

        self.step_n = 0
        
        return np.array(self.state)
    
    def step(self, action , ifrender=False): #(step)
        
        positions = next(self.mm)
        #2D to 3D
        z = np.zeros((np.shape(positions)[0],0))
        self.ueLoc = np.concatenate((positions, z), axis=1).astype(int)

        self.bsLoc = BS_move(self.bsLoc, self.boundaries, action, BS_STEP, MIN_BS_DIST + BS_STEP, N_ACT)
        self.association_map, meanSINR, nOut = self.channel.UpdateDroneNet(self.ueLoc, self.bsLoc, ifrender, self.step_n)
        
        self.bsLocGrid = GetGridMap(self.grid_n, self.grid_n, self.bsLoc)
        self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        
        r_dissect = []
        
        r_dissect.append(meanSINR/20)

        r_dissect.append(-1.0 * nOut/self.nUE)
        
        self.state[0] = self.bsLocGrid
        self.state[1:] = self.association_map

#        dist_penalty = Get_loc_penalty(self.bsLoc, 25, self.nUE)
#        r_dissect.append(-dist_penalty/self.nUE *0.5)

        done = False
#        done = Get_if_collide(self.bsLoc, MIN_BS_DIST)
#        collision = done

        self.step_n += 1
        
#        if collision:
#            r_dissect.append(-1)
#        else:
#            r_dissect.append(0)

        if self.step_n >= MAXSTEP:
            done = True

        reward = max(sum(r_dissect), -1)
#        print meanSINR, " ",nOut," ", r_dissect, " ", reward

#        info = [r_dissect, self.step_n, self.ueLoc]
        info = [r_dissect, self.step_n]
        return np.array(self.state), reward, done, info
    
    def step_test(self, action , ifrender=False): #(step)
        """
            similar to step(), but write here an individual function to
            avoid "if--else" in the original function to reduce training
            time cost
            """
        
        self.ueLoc = self.ueLoc_trace[self.step_n]
        self.bsLoc = BS_move(self.bsLoc, self.boundaries, action, BS_STEP, MIN_BS_DIST + BS_STEP, N_ACT)
        self.association_map, meanSINR, nOut = self.channel.UpdateDroneNet(self.ueLoc, self.bsLoc, ifrender, self.step_n)
        
        self.bsLocGrid = GetGridMap(self.grid_n, self.grid_n, self.bsLoc)
        self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        
        r_dissect = []
        
        r_dissect.append(meanSINR/20)
        
        r_dissect.append(-1.0 * nOut/self.nUE)

        self.state[0] = self.bsLocGrid
        self.state[1:] = self.association_map
        
        done = False
        
        self.step_n += 1
        
        if self.step_n >= MAXSTEP:
            done = True
    
        reward = max(sum(r_dissect), -1)
        
        info = [r_dissect, self.step_n, self.ueLoc]
        return np.array(self.state), reward, done, info


    def render(self):
        fig = figure(1, figsize=(20,20))
        for bs in range(self.nBS):
            subplot(self.nBS +1, 2, bs+1)
            title("bs " + str(bs) + "ue distribution")
            imshow(self.association[bs], interpolation='nearest', origin='lower')
            
            subplot(self.nBS +1, 2, self.nBS+ bs+1)
            title("bs " + str(bs) + "ue sinr distribution")
            imshow(self.association_sinr[bs], interpolation='nearest', origin='lower')

    def plot_sinr_map(self):
        fig = figure(1, figsize=(100,100))
        subplot(1, 2, 1)
        
        sinr_all = self.channel.GetSinrInArea(self.bsLoc)
        imshow(sinr_all, interpolation='nearest', origin='lower', vmin= -50, vmax = 100)
        colorbar()
        xlabel('x[m]')
        ylabel('y[m]')
        title('DL SINR [dB]')
        
        subplot(1,2,2)
        hist(sinr_all, bins=100, fc='k', ec='k')
        ylim((0,20))
        xlabel("SINR")
        ylabel("number of UEs")
        xlim(-100, 100)
        
        show()
        fig.savefig("sinr_map.pdf")
        
        np.save("sinr_map",sinr_all)

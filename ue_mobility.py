# Python code for 2D random walk, fixed direction, and group reference point mobility model.
import numpy as np
#import pylab
import random
import math
from numpy.random import rand

def WalkNRandomSteps(initCoordinates, boundaries, stepLen, nSteps):
    #creating two array for containing x and y coordinate
    #of size equals to the number of size and filled up with 0's
    [initX, initY] = initCoordinates
    x = np.ones(nSteps) * initX
    y = np.ones(nSteps) * initY
    [xMin, xMax, yMin, yMax] = boundaries
    
    for i in range(1, nSteps):
        x[i] = x[i - 1]
        y[i] = y[i - 1]
        val = random.randint(1, 4)
        
        if val == 1:
            if x[i] + stepLen >= xMax:
                x[i] = xMin
            else:
                x[i] = x[i] + stepLen
    
        elif val == 2:
            if x[i] - stepLen <= xMin:
                x[i] = xMax
            else:
                x[i] = x[i] - stepLen

        elif val == 3:
            if y[i] + stepLen >= yMax:
                y[i] = yMin
            else:
                y[i] = y[i] + stepLen
        else:
            if y[i] - stepLen <= yMin:
                y[i] = yMax
            else:
                y[i] = y[i] - stepLen

    return (x, y)

def WalkToFixedDirection(initCoordinates, boundaries, stepLen, nSteps, direction):
    #creating two array for containing x and y coordinate
    #of size equals to the number of size and filled up with 0's
    [initX, initY] = initCoordinates
    x = np.ones(nSteps) * initX
    y = np.ones(nSteps) * initY
    [xMin, xMax, yMin, yMax] = boundaries
    
    for i in range(1, nSteps):
        x[i] = x[i - 1]
        y[i] = y[i - 1]
        val = direction%4 + 1
        
        if val == 1:
            if x[i] + stepLen >= xMax:
                x[i] = xMin
            else:
                x[i] = x[i] + stepLen
    
        elif val == 2:
            if x[i] - stepLen <= xMin:
                x[i] = xMax
            else:
                x[i] = x[i] - stepLen

        elif val == 3:
            if y[i] + stepLen >= yMax:
                y[i] = yMin
            else:
                y[i] = y[i] + stepLen
        else:
            if y[i] - stepLen <= yMin:
                y[i] = yMax
            else:
                y[i] = y[i] - stepLen

    return (x, y)


def GetRandomWalkTraceInbound(numStep, numUE, stepLen, boundaries):

    margin = 5

    [xMin, xMax, yMin, yMax] = boundaries
    xInit = np.random.randint(xMin + margin, xMax - margin, size=numUE)
    yInit = np.random.randint(yMin + margin, yMax - margin, size=numUE)


    nStepX = np.zeros((numUE,numStep))
    nStepY = np.zeros((numUE,numStep))


    for userId in range(numUE):
    #     (nStepX[userId],nStepY[userId]) = WalkToFixedDirection([xInit[userId],yInit[userId]], boundaries, stepLen, nStep, userId)
        (nStepX[userId],nStepY[userId]) = WalkNRandomSteps([xInit[userId],yInit[userId]], boundaries, stepLen, numStep)

    trace = np.zeros((numStep,numUE,3)).astype(int)
    trace[:,:,:2] = np.array([nStepX,nStepY]).T

    return trace



def GetRandomLocationInCellCoverage(gridX_size, gridY_size, cell_size, bsloc, num_node):
    """
        #Generate n random locations in coverage of bsLoc ((0, gridX_size),(0, gridY_size))
        input
        1) cell size in x
        2) BS locations (3D)
        3) number of nodes (bs or ue)
        4) height (z) value - default 0
        5) minimum distance between every 2 nodes - default 0 #TODO
        
        return 1) location of nodes in format (x,y)
        2) heatmap format locations
        """
    
    nBS = np.shape(bsloc)[0]
    rand_users_loc = []
    for ue in range(num_node):
        in_bs = np.random.randint(0, nBS)
        theta = np.random.uniform(0, 2*math.pi)
        r = np.random.uniform(0, cell_size)
        
        rand_users_loc.append([bsloc[in_bs][0] + r* math.sin(theta), bsloc[in_bs][1] + r* math.cos(theta), 0])
    
    loc = np.asarray(rand_users_loc, dtype=int)

#    print loc

    grid = np.zeros((gridX_size, gridY_size))

    for n in range(num_node):
#        print loc[n][0], loc[n][1]
        grid[loc[n][0], loc[n][1]] += 1
    
    return loc, grid


def GetRandomLocationInGrid(gridX_size, gridY_size, num_node, h=0, min_dist=0):
    """
    #Generate n random locations in range ((0, gridX_size),(0, gridY_size))
    input 1) grid size in x
          2) grid size in y
          3) number of nodes (bs or ue)
          4) height (z) value - default 0
          5) minimum distance between every 2 nodes - default 0 #TODO
          
    return 1) location of nodes in format (x,y)
           2) heatmap format locations
    """
    any_too_close = True
    while (any_too_close):
        x = np.random.randint(0, gridX_size, size=num_node)
        y = np.random.randint(0, gridY_size, size=num_node)

        loc = [x,y,np.ones((num_node)) * h] #3D loc
        any_too_close = Get_if_collide(loc, min_dist)

    grid = np.zeros((gridX_size, gridY_size))

    for n in range(num_node):
       grid[x[n], y[n]] += 1

    return np.array(loc, dtype=int).T, grid

def GetGridMap(gridX_size, gridY_size, nodeLoc):
    """
        #Generate n random locations in range ((0, gridX_size),(0, gridY_size))
        input 1) grid size in x
        2) grid size in y
        3) node locations in (x,y) format
        
        return heatmap format locations
        """
    grid = np.zeros((gridX_size, gridY_size))
    
    for n in range(np.shape(nodeLoc)[0]):
        
        grid[nodeLoc[n][0], nodeLoc[n][1]] += 1
    
    return grid


def BS_move(loc, bound, action, stepLen, min_dist, n_action):
    """
        BS takes a single move based on "action" value
        loc: current BSs location
        action: action index (single number for all BS)
        stepLen: step length
        min_dist: minimum distant between BSs,
            (the BS will only move if its distance to all the
            other BSs is greater than this value.
            n_action: total number of actions for a single BS
       return: BSs locations after the step
        """
    
    nBS = np.shape(loc)[0]
#    print "location \n", loc
    act_all = Decimal_to_Base_N(action, n_action, nBS)
#    print "action", act_all
    [xMin, xMax, yMin, yMax] = bound
    
    #action 5-8 moves with longer stepLen
    stepLenLong = stepLen*2
   
    for i in range(nBS):
        
        val = act_all[i]
        [x, y, z] = loc[i]
        
        if val == 0:
            if x + stepLen < xMax:
                x = x + stepLen

        elif val == 1:
            if x - stepLen > xMin:
                x = x - stepLen

        elif val == 2:
            if y + stepLen < yMax:
                y = y + stepLen
    
        elif val == 3:
            if y - stepLen > yMin:
                y = y - stepLen
    
        # stay if val == 4

        elif val == 5:
            if x + stepLenLong < xMax:
                x = x + stepLenLong
        
        elif val == 6:
            if x - stepLenLong > xMin:
                x = x - stepLenLong

        elif val == 7:
            if y + stepLenLong < yMax:
                y = y + stepLenLong
        
        elif val == 8:
            if y - stepLenLong > yMin:
                y = y - stepLenLong


        if_collide = False

        for j in range(nBS):
            if i != j:
                dist = np.linalg.norm(loc[i]-loc[j]) # verify me

                if dist <= min_dist:
                    if_collide = True

        if not if_collide:
            loc[i] = [x, y, z]

#    print "new location \n", loc
    return loc

def UE_rand_move(loc, bound, stepLen):
    
    [xMin, xMax, yMin, yMax] = bound
    
    for i in range(np.shape(loc)[0]):
    
        [x, y, z] = loc[i]

        val = random.randint(0, 3)

        if val == 0:
            if x + stepLen >= xMax:
                x = xMin
            else:
                x = x + stepLen

        elif val == 1:
            if x - stepLen <= xMin:
                x = xMax
            else:
                x = x - stepLen

        elif val == 2:
            if y + stepLen >= yMax:
                y = yMin
            else:
                y = y + stepLen
        elif val == 3:
            if y - stepLen <= yMin:
                y = yMax
            else:
                y = y - stepLen

        loc[i] = [x, y, z]
    return loc


def Decimal_to_Base_N(num, base, digits):
    """Change decimal number ``num'' to given base
        Upto base 36 is supported.
        num: the number to be converted
        base: the base
        digits: number of output digits
        return result_array
        """
    
    result_array = np.zeros((digits))
    converted_string, modstring = "", ""
#    print num
    currentnum = num
    if not 1 < base < 37:
        raise ValueError("base must be between 2 and 36")
    if not num:
        return result_array
    while currentnum:
        mod = currentnum % base
        currentnum = currentnum // base
        converted_string = chr(48 + mod + 7*(mod > 10)) + converted_string
    
    result = np.array([int(d) for d in str(converted_string)])

    result_array[digits - len(result):] = result

    return result_array

def Get_if_collide(locations, threshold):
    """
    check if the distance between any 2 of the given locations are below the threshold
    """
    any_collide = False
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i == j:
                continue
        
            dist = np.linalg.norm(locations[i]-locations[j]) # verify me
#            in number of grids
            if dist <= threshold:
                any_collide = True

    return any_collide

def Get_loc_penalty(locations, threshold, nUE):
    """
        check if the distance between any 2 of the given locations are below the threshold
        """
    penalty = 0
    
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i == j:
                continue
        
            dist = np.linalg.norm(locations[i]-locations[j])
            #
            if dist <= threshold:
                p = nUE - nUE * dist / threshold
                penalty += p

    penalty = math.floor(penalty/2)
    return penalty

'''
    Reference Point Group Mobility model, discussed in the following paper:
    
    Xiaoyan Hong, Mario Gerla, Guangyu Pei, and Ching-Chuan Chiang. 1999.
    A group mobility model for ad hoc wireless networks. In Proceedings of the
    2nd ACM international workshop on Modeling, analysis and simulation of
    wireless and mobile systems (MSWiM '99). ACM, New York, NY, USA, 53-60.
    
    In this implementation, group trajectories follow a random direction model,
    while nodes follow a random walk around the group center.
    The parameter 'aggregation' controls how close the nodes are to the group center.
    
    Required arguments:
    
    *nr_nodes*:
    list of integers, the number of nodes in each group.
    
    *dimensions*:
    Tuple of Integers, the x and y dimensions of the simulation area.
    
    keyword arguments:
    
    *velocity*:
    Tuple of Doubles, the minimum and maximum values for group velocity.
    
    *aggregation*:
    Double, parameter (between 0 and 1) used to aggregate the nodes in the group.
    Usually between 0 and 1, the more this value approximates to 1,
    the nodes will be more aggregated and closer to the group center.
    With a value of 0, the nodes are randomly distributed in the simulation area.
    With a value of 1, the nodes are close to the group center.
    '''

U = lambda MIN, MAX, SAMPLES: rand(*SAMPLES.shape) * (MAX - MIN) + MIN
def reference_point_group(nr_nodes, dimensions, velocity=(0.1, 1.), aggregation=0.1):
    try:
        iter(nr_nodes)
    except TypeError:
        nr_nodes = [nr_nodes]
    
    NODES = np.arange(sum(nr_nodes))
    
    groups = []
    prev = 0
    for (i,n) in enumerate(nr_nodes):
        groups.append(np.arange(prev,n+prev))
        prev += n
    
    g_ref = np.empty(sum(nr_nodes), dtype=np.int)
    for (i,g) in enumerate(groups):
        for n in g:
            g_ref[n] = i
    
    FL_MAX = max(dimensions)
    MIN_V,MAX_V = velocity
    FL_DISTR = lambda SAMPLES: U(0, FL_MAX, SAMPLES)
    VELOCITY_DISTR = lambda FD: U(MIN_V, MAX_V, FD)
    
    MAX_X, MAX_Y = dimensions
    x = U(0, MAX_X, NODES)
    y = U(0, MAX_Y, NODES)
    velocity = 1.
    theta = U(0, 2*np.pi, NODES)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    
    GROUPS = np.arange(len(groups))
    g_x = U(0, MAX_X, GROUPS)
    g_y = U(0, MAX_X, GROUPS)
    g_fl = FL_DISTR(GROUPS)
    g_velocity = VELOCITY_DISTR(g_fl)
    g_theta = U(0, 2*np.pi, GROUPS)
    g_costheta = np.cos(g_theta)
    g_sintheta = np.sin(g_theta)
    
    aggregating = 200
    deaggregating = 100
    
    while True:
        
        x = x + velocity * costheta
        y = y + velocity * sintheta
        
        g_x = g_x + g_velocity * g_costheta
        g_y = g_y + g_velocity * g_sintheta
        
        if aggregating:
            for (i,g) in enumerate(groups):
                
                # step to group direction + step to group center
                x_g = x[g]
                y_g = y[g]
                c_theta = np.arctan2(g_y[i] - y_g, g_x[i] - x_g)
                
                x[g] = x_g + g_velocity[i] * g_costheta[i] + aggregation*np.cos(c_theta)
                y[g] = y_g + g_velocity[i] * g_sintheta[i] + aggregation*np.sin(c_theta)
            
            aggregating -= 1
            if aggregating == 0: deaggregating = 100
    
        else:
            for (i,g) in enumerate(groups):
                
                # step to group direction + step to group center
                x_g = x[g]
                y_g = y[g]
                c_theta = np.arctan2(g_y[i] - y_g, g_x[i] - x_g)
                
                x[g] = x_g + g_velocity[i] * g_costheta[i]
                y[g] = y_g + g_velocity[i] * g_sintheta[i]
            
            deaggregating -= 1
            if deaggregating == 0: aggregating = 10
        
        # node and group bounces on the margins
        b = np.where(x<0)[0]
        if b.size > 0:
            x[b] = - x[b]; costheta[b] = -costheta[b]
            g_idx = np.unique(g_ref[b]); g_costheta[g_idx] = -g_costheta[g_idx]
        b = np.where(x>MAX_X)[0]
        if b.size > 0:
            x[b] = 2*MAX_X - x[b]; costheta[b] = -costheta[b]
            g_idx = np.unique(g_ref[b]); g_costheta[g_idx] = -g_costheta[g_idx]
        b = np.where(y<0)[0]
        if b.size > 0:
                y[b] = - y[b]; sintheta[b] = -sintheta[b]
                g_idx = np.unique(g_ref[b]); g_sintheta[g_idx] = -g_sintheta[g_idx]
        b = np.where(y>MAX_Y)[0]
        if b.size > 0:
            y[b] = 2*MAX_Y - y[b]; sintheta[b] = -sintheta[b]
            g_idx = np.unique(g_ref[b]); g_sintheta[g_idx] = -g_sintheta[g_idx]

        # update info for nodes
        theta = U(0, 2*np.pi, NODES)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # update info for arrived groups
        g_fl = g_fl - g_velocity
        g_arrived = np.where(np.logical_and(g_velocity>0., g_fl<=0.))[0]

        if g_arrived.size > 0:
            g_theta = U(0, 2*np.pi, g_arrived)
            g_costheta[g_arrived] = np.cos(g_theta)
            g_sintheta[g_arrived] = np.sin(g_theta)
            g_fl[g_arrived] = FL_DISTR(g_arrived)
            g_velocity[g_arrived] = VELOCITY_DISTR(g_fl[g_arrived])

        yield np.dstack((x,y))[0]

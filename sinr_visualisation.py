from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import *
from IPython import display

matplotlib.rcParams.update({'font.size': 14})
#
## used to plot snapshoot of user distribution
#usrDistributionNSteps = np.zeros((nStep, gridX, gridY))
#for userId in range(nUE):
#    (nStepX[userId],nStepY[userId]) = WalkToFixedDirection([xInit[userId],yInit[userId]], boundaries, stepLen, nStep, userId)
#    for stepN in range(nStep):
#        x = int(nStepX[userId][stepN])
#        y = int(nStepY[userId][stepN])
#        usrDistributionNSteps[stepN][x][y] += 1
#
#
##plotting user distribution:
## for time in [1,2,3,nStep/2, nStep-1]:
#for time in [0]:
#    title("User Distribution at time "+ str(time) + "s")
#    #     print usrDistributionNSteps[time][np.nonzero(usrDistributionNSteps[time])]
#    imshow(usrDistributionNSteps[time].T, cmap='hot', interpolation='nearest', origin='lower')
#    xlabel("x")
#    ylabel("y")
#    show()
#
#
## plotting user trajectory (static):
#for ue in range(nUE):
#    title("Random Walk of UE " + str(ue) +"($n = " + str(nStep) + "$ steps)")
#    ylim(xMax)
#    xlim(yMax)
#    xlabel("x")
#    ylabel("y")
#    plot(nStepX[ue],nStepY[ue])
#    show()
#
#
## visualise UL DL sinrs
## For best UL SINR
#fig = figure()
#sinrDistrbution = np.zeros((xMax - xMin, yMax - yMin))
#for userId in range(nUE):
#    x = int(ueLocationAll[userId][0])
#    y = int(ueLocationAll[userId][1])
#    sinrDistrbution[y][x] = bestUlSinr[userId]
#title("Best UL SINR")
#pos = imshow(sinrDistrbution, cmap='hot', interpolation='nearest')
#fig.colorbar(pos)
#show
#savefig("BestULSINR", dpi=None, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None)
#
## For best DL SINR
#fig = figure()
#sinrDistrbution = np.zeros((xMax - xMin, yMax - yMin))
#for userId in range(nUE):
#    x = int(ueLocationAll[userId][0])
#    y = int(ueLocationAll[userId][1])
#    sinrDistrbution[y][x] = bestDlSinr[userId]
#title("Best DL SINR")
#pos = imshow(sinrDistrbution, cmap='hot', interpolation='nearest')
#fig.colorbar(pos)
#show
#savefig("BestDLSINR", dpi=None, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None)
#
#
## For individual BSs
#sinrDistrbution = np.zeros((nBS, xMax - xMin, yMax - yMin))
#
#for bsId in range(nBS):
#    for userId in range(nUE):
#        x = int(ueLocationAll[userId][0])
#        y = int(ueLocationAll[userId][1])
#        # Issue with the ueLocationAll indexing when plot the heat map.
#        # checked with distance and SINR values all okay but when plotting the heatmap, x, y are inverted
#        # plotting fixed values are also okay..
#        # using sinrDistrbution[bsId][y][x] instead of sinrDistrbution[bsId][x][y] resolves the issue
#        #         sinrDistrbution[bsId][x][y] = GetDistance(ueLocationAll[userId], bsLocationAll[bsId])
#        sinrDistrbution[bsId][y][x] = ulSinr[userId][bsId]#GetDistance(ueLocationAll[userId], bsLocationAll[bsId])
#
##plotting user distribution:
#for bsId in range(nBS):
#    x = bsLocationAll[bsId][0]
#    y = bsLocationAll[bsId][1]
#    fig = figure()
#    ax = fig.add_subplot(111)
#    ax.annotate('BS', xy=(x,y), xytext=(x, y),
#                arrowprops=dict(facecolor='black', shrink=0.05))
#    for ueId in range(2):
#        ax.annotate('UE', xy=(ueLocationAll[ueId][0],ueLocationAll[ueId][1]), xytext=(ueLocationAll[ueId][0],ueLocationAll[ueId][1]),
#                    arrowprops=dict(facecolor='white', shrink=0.05))
#        print "UE",ueId," (",ueLocationAll[ueId] ,")", ulSinr[ueId][bsId]
#
#    title("DL SINR Distribution from BS"+ str(bsId) + " (" + str(x) + ", " + str(y) + ")")
#    imshow(sinrDistrbution[bsId], cmap='hot', interpolation='nearest')
#    xlabel("x [m]")
#    ylabel("y [m]")
#    show()



def draw_UE_HO(ue_loc, numGridX, numGridY, bsLoc, ue2watch, xbestSinr, xcurrSinr, xbestBS, xcurrBS, currentTime, dlRate, ulRate, size=(5, 5), color = []):
    fig = figure(1, figsize=size)
    fig.subplots_adjust(hspace=.4)
#    subplot(4, 1, 1)
    ueValGrid = np.zeros((3, numGridX, numGridY))
    if color.any():
        for usr in xrange(len(ue_loc)):
            ueValGrid[:, ue_loc[usr][0], ue_loc[usr][1]] = color[usr]
    else:
        # problem
        ueValGrid[:, ue_loc[0], ue_loc[1]] = 1
    
    for bsId in range(len(bsLoc)):
        x = bsLoc[bsId][0]
        y = bsLoc[bsId][1]
        strBS = "BS " + str(bsId)
        text(x, y, strBS, color='white')

    xlabel("x [m]")
    ylabel("y [m]")
    title("User distribution at time " + str(currentTime))
    imshow(ueValGrid.T, interpolation='nearest', origin='lower')
#
#    subplot(4,1,2)
#    grid = np.zeros((numGridX, numGridY))
#    grid[ue_loc[ue2watch][0], ue_loc[ue2watch][1]] = 1
#
#    for bsId in range(len(bsLoc)):
#        x = bsLoc[bsId][0]
#        y = bsLoc[bsId][1]
#        strBS = "BS " + str(bsId)
#        text(x, y, strBS)
#
#        xlabel("x [m]")
#        ylabel("y [m]")
#        title("UE"+ str(ue2watch) + " current location ")
#        imshow(grid.T, interpolation='nearest', origin='lower')
#
#    #     UE 2 watch SINR from current BS and best BS
#    subplot(4,1,3)
#    xlabel("Time [Steps]")
#    ylabel("SINR [dB]")
#    strLegendcur = "current SINR  (BS" + str(xcurrBS) + ")"
#    strLegendbes = "best SINR (BS" + str(xbestBS) + ")"
#
#    timeAxis = currentTime - np.array(range(len(xcurrSinr)))
##    print len(xcurrSinr)
#    plot(timeAxis, xcurrSinr, label = strLegendcur)
#    hold('on')
#    plot(timeAxis, xbestSinr, label = strLegendbes)
#
#    legend(loc=2)
#    ylim(-100, 100)
#    title("SINR from best BS and current ass BS. UE"+ str(ue2watch))
#
#
#    subplot(4,1,4)
#    xlabel("Time [Steps]")
#    ylabel("Mean Rate [Mbps]")
#    strLegendcur = "UL"
#    strLegendbes = "DL"
#
#    timeAxis = currentTime - np.array(range(len(ulRate))) #here
##    print len(ulRate)
#    plot(timeAxis, ulRate, label = "UL")
#    hold('on')
#    plot(timeAxis, dlRate, label = "DL")
#
#    legend(loc=2)
#    ylim(0, 1)
#    title("Mean rate of all UEs from serving BSs")
#
#
    show()


def plot_sinr_distribution(sinr, x_from, x_to):
    hist(sinr, bins=50, fc='k', ec='k')
    
    xlabel("SINR")
    ylabel("number of UEs")
    xlim(x_from, x_to)
#    savefig("SINR_dist.pdf")
    show()

import numpy as np
import math
from IPython import display
from sinr_visualisation import *
# channel model

WATCH_WINDOW = 200
OUT_THRESH = 0

class LTEChannel:
    def __init__(self, nUE, nBS, boundaries, init_ueLoc, init_bsLoc):
        
        [self.xMin, self.xMax, self.yMin, self.yMax] = boundaries
        self.gridX = self.xMax - self.xMin + 1
        self.gridY = self.yMax - self.yMin + 1

        self.nUE = nUE
        self.nBS = nBS
        self.gridWidth = 5

        # FDD ratio
        self.alpha = 0.5
        # total number of channels
        self.K = 120
        # freq reuse factor
        self.r_freq = 1
        # numebr of channels per BS in DL
        self.K_DL = self.alpha * self.K / self.r_freq
        # numebr of channels per BS in UL
        self.K_UL = (1 - self.alpha) * self.K / self.r_freq
        # UE Tx power in dBm
        self.P_ue_dbm = 23
        # BS Tx power in dBm
        self.P_bs_dbm = 20
        # per channel tx power in DL
        #P_b = P_bs / K_DL
        # per channel Gaussian noise power in dBm
        self.noise_dbm = -121
        
        #path_loss = a + b*log(d) if d>= path_loss_dis in dB
#        self.pathloss_a = 128.1
#        self.pathloss_b = 37.6
#        self.pathloss_dis = 0.035 #in Km  d>= path_loss_dis to have path loss else path loss is 0
        self.pathloss_a = 38
        self.pathloss_b = 30
        self.pathloss_dis = 0
        # Antenna Gain in dB
        self.antenna_gain = 2
        # Equipment/penetrasion loss in dB
        self.eq_loss = 0
        # shadow fading
        self.shadowing_mean = 0 #log normal shadowing N(mean, sd)
        self.shadowing_sd = 2 #6log normal shadowing N(mean, sd)
        
        self.P_ue_watt = 10 **(self.P_ue_dbm / float(10)) * 1e-3 #UE Tx power in W
        self.P_bs_watt = 10 **(self.P_bs_dbm/ float(10)) * 1e-3  #BS Tx power in W
        self.noise_watt = 10 **(self.noise_dbm / float(10)) * 1e-3
        
        self.sc_ofdm = 12 #nbr of data subcarriers/subchannel bandwidth
        self.sy_ofdm = 14 #nbr of ofdm symbols/subframe
        self.t_subframe = 1e-3 #subframe durantion in s
        #mapping from sinr to MCS
        self.sinr_thresholds = [-float('inf'), -6.5, -4, -2.6, -1, 1, 3, 6.6, 10, 11.4, 11.8, 13, 13.8, 15.6, 16.8, 17.6, float('inf')]
        self.sinr_thresholds_watt = [10 ** (s / float(10)) for s in self.sinr_thresholds]
        self.efficiency = [1e-16, 0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.90, 4.52, 5.12, 5.55] #bits/symbol
        self.rate_thresholds = [(self.sc_ofdm * self.sy_ofdm / float(self.t_subframe)) * e * 1e-6 for e in self.efficiency] #Mb/s
        
        self.total_channels = 120
        # dl_channels_init = (alpha * total_channels)/reuse_factor
        self.ul_channels_init = (1-self.alpha) * self.total_channels
        
        # UL gain average over n number of (imaginary) users from the interfering BS
        self.n = 1000
        # radius of BS coverage used to generate (imaginary) users for the interfering BS
        self.dth = 100
        #number of already associated users on each BS
        self.ass_per_bs = np.ones((self.nBS,1))#[1, 1, 1, 1, 1, 1]
    
        self.hoBufDepth = 3 #time to trigger HO
        self.hoThresh_db = 1
        #        self.init_BS = np.zeros((self.nUE)).astype('int32')

        self.interfDL = [range(self.nBS) for bs in range(self.nBS)]
        
        for bs in range(self.nBS):
            self.interfDL[bs].remove(bs)
        
        self.interfUL = self.interfDL
        
        self.current_BS, self.current_BS_sinr = self.GetBestDlBS(init_ueLoc, init_bsLoc) #self.init_BS
        self.bestBS_buf = [self.current_BS]

            
         # rgbColor code for each BS
        self.rgbColours = [[255,0,0],[0,255,0],[0,0,255],[0,255,255],[255,255,0]]
         # needed for animation/plot the rgb colour code for each UE regarding to which BS it connects to
        self.bsColour = np.zeros((nUE,3))
        
         # monitor UE 0
        self.ue2Watch = 0
        self.watch_ue_sinrbest = np.zeros((WATCH_WINDOW))
        self.watch_ue_sinrcurr = np.zeros((WATCH_WINDOW))
    
        #for visualising mean rate ul dl
        self.watch_ul_rate = np.zeros((WATCH_WINDOW))
        self.watch_dl_rate = np.zeros((WATCH_WINDOW))
    
        self.ue_out = np.where(self.current_BS_sinr<= OUT_THRESH)
    
    
    def reset(self, ueLoc, bsLoc):
        self.current_BS, self.current_BS_sinr = self.GetBestDlBS(ueLoc, bsLoc) #self.init_BS
        self.bestBS_buf = [self.current_BS]
        self.ue_out = np.where(self.current_BS_sinr<= OUT_THRESH)


    def GetBestDlBS(self, ueLoc, bsLoc):
        channelGainAll = self.GetChannelGainAll(ueLoc, bsLoc)
        dlSinr = self.GetDLSinrAllDb(channelGainAll)
        bestDlBS = np.argmax(dlSinr, axis=1)
        bestDlSINR = np.max(dlSinr, axis=1)
        return bestDlBS, bestDlSINR
    
    def SetDistanceMultiplier(self, gridWidth):
        self.gridWidth = gridWidth
    
    def SetTxPower (self, txpower):
        self.P_bs_dbm = txpower
        self.P_bs_watt =  10 **(self.P_bs_dbm/ float(10)) * 1e-3
        
    def SetHandoverThreshold (self, ho_thresh):
        self.ho_thresh = ho_thresh
    


    def UpdateDroneNet(self, ueLoc, bsLoc, ifdisplay=False, time_now=0, get_rate=False):
        channelGainAll = self.GetChannelGainAll(ueLoc, bsLoc)
        dlSinr = self.GetDLSinrAllDb(channelGainAll)
        bestDlBS = np.argmax(dlSinr, axis=1)
        bestDlSINR = np.max(dlSinr, axis=1)
#        print time_now, "s ue ", ueLoc[10], " dl sinr ", bestDlSINR[10], " from BS ", bestDlBS[10]

        for ue in xrange(self.nUE):
            self.current_BS_sinr[ue] = dlSinr[ue, self.current_BS[ue]]

        if np.shape(self.bestBS_buf)[0] < self.hoBufDepth:
            self.bestBS_buf = np.append(self.bestBS_buf, [bestDlBS], axis=0)
        else:
            #FIFO buffer bottom-in
            self.bestBS_buf[:-1] = self.bestBS_buf[1:]
            self.bestBS_buf[-1] = bestDlBS
        #     print "bestBS_buf..\n", bestBS_buf
        bestRemain = np.all(self.bestBS_buf == self.bestBS_buf[0,:], axis = 0)
        bestChanged = self.current_BS != self.bestBS_buf[-1,:]

        ifNeedHO = np.logical_and(bestRemain, bestChanged)
        ifNeedHO = np.logical_and(ifNeedHO, bestDlSINR - self.current_BS_sinr > self.hoThresh_db)
#        print "if needHO", ifNeedHO

        if np.any(ifNeedHO):
            ueNeedHO = np.flatnonzero(ifNeedHO)
            for ue in ueNeedHO:
                fromBS = self.current_BS[ue]
                toBS = self.bestBS_buf[-1][ue]
                self.current_BS[ue] = self.bestBS_buf[-1][ue]
        
        #compute number of ue out of coverage
        ue_out = np.array(np.where(self.current_BS_sinr<= OUT_THRESH))
        new_out = ue_out[np.isin(ue_out, self.ue_out, invert=True)]

        self.ue_out = ue_out
        n_outage = np.size(new_out)
#        print " ", new_out, " ", self.ue_out, " ", n_outage

    
        if get_rate or ifdisplay:
            #Get DL/Ul rate updates
            #DL
            dlRatePerChannel = self.GetDLRatePerChannel(dlSinr)
            dlRatePerChannel_from_currentBS = np.zeros((self.nUE))
            #UL
            ulInterfPower = self.GetULInterference(bsLoc)
            # print "UL interference power \n", ulInterfPower
            ulRequiredRate = 1

            ulSinr = []
            ulNumChannelNeeded = []
            ulRateUEBS = []

            for u in range(self.nUE):
                tup = self.GetULRateChannels (u, ulRequiredRate, ulInterfPower, channelGainAll)
                ulSinr.append(tup[0])
                ulNumChannelNeeded.append(tup[1])
                ulRateUEBS.append(tup[2])

            ulSinr = np.array(ulSinr)
            ulNumChannelNeeded = np.array(ulNumChannelNeeded)
            ulRateUEBS = np.array(ulRateUEBS)
        #        print "UL SINR \n", ulSinr
        #        print "UL needed channels \n", ulNumChannelNeeded
        #        print "UL rate UE per BS \n", ulRateUEBS

        #        UL DL rate from current BS
            dlRatePerChannel_from_currentBS = np.zeros((self.nUE))
            ulRatePerChannel_from_currentBS = np.zeros((self.nUE))
            for ue_id in range(self.nUE):
                dlRatePerChannel_from_currentBS[ue_id] = dlRatePerChannel[ue_id][self.current_BS[ue_id]] #can be accelarated
                ulRatePerChannel_from_currentBS[ue_id] = ulRateUEBS[ue_id][self.current_BS[ue_id]] #can be accelarated
            # mean rate of all UEs as received from their current BSs(maybe don't care)
            dl_rate_mean = np.mean(dlRatePerChannel_from_currentBS)
            ul_rate_mean = np.mean(ulRatePerChannel_from_currentBS)
        #        print "mean DL and UL Rate Per Channel \n", dl_rate_mean, ul_rate_mean


        association_map = self.GetCurrentAssociationMap(ueLoc)

#        print self.current_BS_sinr, "\n mean sinr", np.mean(self.current_BS_sinr)
        return association_map, np.mean(self.current_BS_sinr), n_outage


    # compute the euclidean distance between 2 nodes
    def GetDistance(self, coord1, coord2):
        coord1 = coord1[:2]* self.gridWidth
        coord2 = coord2[:2]* self.gridWidth
        dist = np.linalg.norm(coord1-coord2)

        #     dist = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
        return dist
    
    # compute the pass loss value for the given euclidean distance between BS b and UE i
    # based on urban pass loss model as per 3GPP TR 36.814
    def GetPassLoss(self, d):
#        d = d/1000#work with km
        loss = 0
        if d > self.pathloss_dis:
            loss = self.pathloss_a + self.pathloss_b * math.log10(d)
        return loss
    
    def GetChannelGain(self, coord1, coord2):
        d = self.GetDistance(coord1, coord2)
        pathLoss = self.GetPassLoss(d)
        fading = np.random.normal(self.shadowing_mean, self.shadowing_sd)
        #     fading = 10 #static fading >> for calibration only!!!
        #     print fading
        # the channel gain between UE and BS accounts for
        #1) antenna gain 2) pathloss 3) equipment loss 4) shadow fading
        channel_gain_db = self.antenna_gain - pathLoss - fading - self.eq_loss
        channel_gain = 10 ** (channel_gain_db / float(10))
        return channel_gain
    
    def GetChannelGainAll(self, ueLocations, bsLocations):
        n_ue = np.shape(ueLocations)[0]
        n_bs = np.shape(bsLocations)[0]
        channel_gain_all = np.zeros((n_ue, n_bs))
        
        for ue_id in range(n_ue):
            for bs_id in range(n_bs):
                channel_gain_all[ue_id][bs_id] = self.GetChannelGain(ueLocations[ue_id], bsLocations[bs_id])
        return channel_gain_all
    
    def GetDLSinrAllDb(self, channel_gain_all):
        sinr_all = np.zeros((self.nUE,self.nBS))
        for ue_id in range(self.nUE):
            for bs_id in range(self.nBS):
                interf_bs = self.interfDL[bs_id]
                
                P_interf = np.sum(self.P_bs_watt * channel_gain_all[ue_id][interf_bs])
                sinr_dl = self.P_bs_watt * channel_gain_all[ue_id, bs_id] / float(self.noise_watt + P_interf)
                
                sinr_all[ue_id][bs_id] = 10 * math.log10(sinr_dl)
        return sinr_all
    
    #Mapping from SINR to MCS rates
    def GetDLRatePerChannel(self, dl_sinr_db):
        dl_rate_per_channel = np.zeros((self.nUE,self.nBS))
        for ue_id in range(self.nUE):
            for bs_id in range(self.nBS):
                for l in range(len(self.sinr_thresholds) - 1):
                    if (dl_sinr_db[ue_id][bs_id] >= self.sinr_thresholds[l] and dl_sinr_db[ue_id][bs_id] < self.sinr_thresholds[l+1]):
                        dl_rate_per_channel[ue_id][bs_id] = self.rate_thresholds[l]
                        break
        return dl_rate_per_channel
    
    def GetNumberDLChannelNeeded(self, requiredRate, dl_rate_per_channel):
        dl_needed_channels = np.zeros((self.nUE,self.nBS))
        for ue_id in range(self.nUE):
            for bs_id in range(self.nBS):
                dl_needed_channels[ue_id][bs_id] = requiredRate/dl_rate_per_channel[ue_id][bs_id]
        return dl_needed_channels
    
    def GetAverageULChannelGainFromInterfBS(self, bs_id, intf_id, bs_loc, bs_intf_loc):
        channel_gain = np.zeros((self.n))
        
        theta = np.random.uniform(0, 2*math.pi, self.n)
        r = np.random.uniform(0, self.dth, self.n)
        
        #imagine n users attached to bs_intf
        vfunc_sin = np.vectorize(math.sin)
        vfunc_cos = np.vectorize(math.cos)
        
        rand_users_loc = np.array([bs_intf_loc[0] + r* vfunc_sin(theta), bs_intf_loc[1] + r* vfunc_cos(theta)])
        #if simulating 3D model (judging by the number of dimensions bs_loc has)
        if np.size(bs_loc) == 3:
            rand_users_loc = np.append(rand_users_loc, np.ones((1,self.n))*bs_loc[2], axis=0)
        
        rand_users_loc = np.transpose(rand_users_loc)
        
        # save the random user location for calibration
        # bs_id and intf_id can be removed from the function input if not printing this anymore
        #         str_name = "rand_users_bs_" + str(bs_id) + "_intf_" + str(intf_id)
        #         np.save(str_name, rand_users_loc)
        
        for intf_ue_id in range(self.n):
            channel_gain[intf_ue_id] = self.GetChannelGain(bs_loc, rand_users_loc[intf_ue_id])
        
        return np.mean(channel_gain)


    def GetAverageULChannelGain(self, bs_loc):
        avg_channel_gain = np.zeros((self.nBS, self.nBS))
        for bs_id in range(self.nBS):
            for intf_id in range(self.nBS):
                #make avg_channel_gain symmetric
                if (intf_id >= bs_id):
                    if intf_id in self.interfUL[bs_id]:
                        avg_channel_gain[bs_id][intf_id] = self.GetAverageULChannelGainFromInterfBS(bs_id, intf_id, bs_loc[bs_id], bs_loc[intf_id])
                    else:
                        avg_channel_gain[bs_id][intf_id] = avg_channel_gain[intf_id][bs_id]
        
        #     print "UL channel gain", avg_channel_gain
        return avg_channel_gain

    def GetULInterference(self, bs_loc):
        ul_interference = np.zeros((self.nBS))
        ulAvgChannelGain= self.GetAverageULChannelGain(bs_loc)
        
        for bs_id in range(self.nBS):
            for intf_id in self.interfUL[bs_id]:
                ul_interference[bs_id] += self.P_ue_watt * ulAvgChannelGain[bs_id][intf_id] * self.ass_per_bs[intf_id] / self.ul_channels_init
            
        return ul_interference
    
    def GetULRateChannels (self, u, ul_datarate, ul_interference, channel_gain):
        
        """
            #list of the number of needed channels on the UL from each base station to grant the user u
            #the data rate he asks for (when in guaranteed throughput)
            #returns both the number of channels needed by user u from each BS (to get the asked data rate),
            #and the uplink rate between the user u and each BS
            
            :param u: user index
            :param bs: {b0:[x_b0, y_b0], b1:[x_b1, y_b1], ...} #dictionary of BS coordinates
            :param ul_datarate: the value of the requested data rate on the UL
            :param ul_interference: uplink_interference(bs, interference_set, ass_per_bs)
            :param channel_gain: compute_channel_gain (users, bs)
            :return: ul_channels_needed, ul_rate
            """
        
        ul_sinr_db = []
        ul_channels_needed = []
        ul_rate = []
        
        ul_channels_min = [ul_datarate / r for r in self.rate_thresholds]
        
        for b in range(self.nBS):
            ul_channels_threshold = []
            
            sinr_ratio = (self.P_ue_watt * channel_gain[u][b]) / (self.noise_watt + ul_interference[b]) #the SINR ratio without dividing by the number of channels needed, for a given user u
            
            ul_sinr_db.append(10 * math.log10(sinr_ratio))
            
            for l in range(1, len(self.sinr_thresholds_watt) - 1):
                ul_channels_threshold.append(sinr_ratio/self.sinr_thresholds_watt[l])
        
            ul_channels_threshold.insert(0, float('inf'))
            ul_channels_threshold.append(0)
            
            match = []
            for id, val in enumerate(ul_channels_min):
                if val <= ul_channels_threshold[id] and val > ul_channels_threshold[id +1]:
                    match.append(val)

            ul_channels_needed.append(min(match))
            ul_rate.append(ul_datarate/min(match)) #assume it to be rate per channel?


        return ul_sinr_db, ul_channels_needed, ul_rate

    def GetCurrentAssociationMap(self, ueLoc):
        """
            utility func mainly for the class mobile_env
            take ue locations as input,
            and convert current_BS into a (nBS x, GridX, GridY) heatmap for each BS
            NB: current_BS might not be best_bs
            Return the association heatmap
            TODO: check dim gridX_N and gridY_N
        """
#        convert to 2D if given a 3D location
        if np.shape(ueLoc)[1] == 3:
            ueLoc = ueLoc[:,:-1]
        

        association_map = np.zeros((self.nBS, self.gridX, self.gridY))
#        association_sinr_map = np.zeros((self.nBS, gridX, gridY))

        for ue in range(self.nUE):
            bs = self.current_BS[ue]
            association_map[bs][ueLoc[ue][0]][ueLoc[ue][1]] += 1
#            association_sinr_map[bs][ueLoc[ue][0]][ueLoc[ue][1]] = self.current_BS_sinr[ue]

        return association_map#, association_sinr_map

    def GetSinrInArea (self, bsLoc):

        dl_sinr = np.zeros((self.gridX, self.gridY))

        loc_id = 0
        for x_ in range(self.xMin, self.xMax):
            for y_ in range(self.yMin, self.yMax):
                dist = []
                for bs in range(np.shape(bsLoc)[0]):
                    dist.append(self.GetDistance(np.asarray([x_,y_,0]),bsLoc[bs]))
                
                bs_id = np.argmin(dist)
                P_interf = 0
                
                for i in self.interfDL[bs_id]:
                    interf_gain = self.GetChannelGain(np.asarray([x_,y_,0]), bsLoc[i])
                    P_interf += self.P_bs_watt * interf_gain
                
                sinr_dl = self.P_bs_watt * self.GetChannelGain(np.asarray([x_,y_,0]), bsLoc[bs_id]) / float(self.noise_watt + P_interf)

                dl_sinr[x_][y_] = 10 * math.log10(sinr_dl)

        return dl_sinr


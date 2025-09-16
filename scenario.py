# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:49 2020

@author: liangyu

Create the network simulation scenario
"""

import numpy as np
from numpy import pi
from random import random, uniform, choice
from random_factor_mbs import *
from random_factor_sbs import *

x_LOS_list, x_NLOS_list, z_list = [],[],[]
x_list_sbs, z_list_sbs, R_list = [],[],[]
NRB_sbs = 275

def EL_assign(choice):
    if choice == 1:
        requ_rate = 12
    elif choice == 2:
        requ_rate = 10
    elif choice == 3:
        requ_rate = 9
    else:
        raise ValueError("Invalid choice. Choice must be 1, 2, or 3.")
    return requ_rate

def BL_assign(choice):
    if choice == 1:
        requ_rate = 4
    elif choice == 2:
        requ_rate = 2
    elif choice == 3:
        requ_rate = 1
    else:
        raise ValueError("Invalid choice. Choice must be 1, 2, or 3.")
    return requ_rate

class BS:  # Define the base station
    
    def __init__(self, sce, BS_index, BS_type, BS_Loc, BS_Radius, BS_Bw):  # Initialize the base station
        self.sce = sce
        self.id = BS_index
        self.BStype = BS_type
        self.BS_Loc = BS_Loc
        self.BS_Radius = BS_Radius
        self.BS_Bw_Per_Channel = BS_Bw / self.sce.nChannel  # Bandwidth per channel
        self.Noise_dB = 0  # Noise in dB
        if(self.BStype == "MBS"):
            self.Noise_dB = -148.365  # Noise in dB
        elif(self.BStype == "SBS"):
            self.Noise_dB = -204 + 10 * np.log10(self.sce.SBS_BW) + 10  # Noise in dB
        elif(self.BStype == "FBS"):
            self.Noise_dB = -204 + 10 * np.log10(self.sce.FBS_BW) + 10
        
        
    def reset(self):  # Reset the channel status
        self.Ch_State = np.zeros(self.sce.nChannel)    
        
    def Get_Location(self):
        return self.BS_Loc
    
    def Transmit_Power_dBm(self):  # Calculate the transmit power of a BS
        if self.BStype == "MBS":
            Tx_Power_dBm = 46   
        elif self.BStype == "SBS":
            Tx_Power_dBm = 30 
        elif self.BStype == "FBS":
            Tx_Power_dBm = 20 
        return Tx_Power_dBm  # Transmit power in dBm, no consideration of power allocation now

    def loss_sbs_db(self, ue_id, d):  # Path loss of SBS
        alpha = 61.4  # some constant
        beta = 2
        x = x_list_sbs[self.id - self.sce.nMBS][ue_id]  # parameter used in eq
        x_dB = 10 * np.log10(x)
        path_loss = alpha + beta * 10 * np.log10(d) + x_dB  # path loss in dB
        return path_loss

    def loss_mbs_db(self, ue_id, d):  # Path loss of MBS
        h_BS = 25  # height of BS
        h_UT = 2.5  # height of user
        fc = 2.8  # Centre frequency in GHz 
        c = 3e8  # Speed of light
        h_E = 1  # height of common ground
        # (verified and corrected)
        d_BP = 4 * (h_BS - h_E) * (h_UT - h_E) * (fc / c) * 1e9  # Effective distance (Friss free space equation)

        # 3D distance
        d_3D = np.sqrt((d)**2 + (h_BS - h_UT)**2)

        # Path loss for NLOS (verified)
        PL_NLOS1 = 13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(fc) - 0.6 * (h_UT - 1.5)

        # Path loss and total gain for LOS
        if (d) <= d_BP:
            PL_LOS = 28.0 + 22 * np.log10(d_3D) + 20 * np.log10(fc)
            # print("PL_LOS:", PL_LOS, "d_3D:", d_3D, "fc:", fc)  # Debug: Uncomment for troubleshooting
        else:
            PL_LOS = 28.0 + 40 * np.log10(d_3D) + 20 * np.log10(fc) - 9 * np.log10(d_BP**2 + (h_BS - h_UT)**2)
            # Tot_gain_LOS = G_LOS - PL_LOS

        # Path loss and total gain for NLOS
        PL_NLOS = max(PL_LOS, PL_NLOS1)
        # Tot_gain_NLOS = G_NLOS - PL_NLOS

        # if d < 0.6 use LOS, else use NLOS
        if d < self.BS_Radius:
            Tot_path_loss_db = PL_LOS
        else:
            Tot_path_loss_db = PL_NLOS

        return Tot_path_loss_db # Path loss in dB, which is the total gain of the channel

    def tx_gain_sbs_db(self, ue_id):  # Calculate the transmit gain of a certain BS
        f = 28e9  # millimeter wave frequency (Hz)
        lamda = 3e8 / f  # wavelength (meters)
        p_t_RB = 1.0/NRB_sbs
        p_t_dB = 10 * np.log10(p_t_RB)
        L_t = 0.009  # transmitter antenna length (meters)
        L_r = 0.009  # receiver antenna length (meters)
        G_t = 20 * np.log10(np.pi * L_t / lamda)
        G_r = 20 * np.log10(np.pi * L_r / lamda)
        R = R_list[self.id - self.sce.nMBS][ue_id]  # Get the random factor for SBS
        random_gain_dB = 10 * np.log10(np.abs(R))
        tx_gain = G_t + G_r + p_t_dB + random_gain_dB  # Transmit gain in dB
        return tx_gain

    def tx_gain_mbs_db(self, ue_id, d):  # Calculate the transmit gain of a certain BS
        K = 10**(-2)  # Some constant
        # Channel modeling
        x_LOS = x_LOS_list[ue_id]
        x_NLOS = x_NLOS_list[ue_id]
        z = z_list[ue_id]
        G_LOS = 10 * np.log10(K * x_LOS * z)
        G_NLOS = 10 * np.log10(K * x_NLOS * z)

        if d < self.BS_Radius:
            gain_db = G_LOS
        else:
            gain_db = G_NLOS

        G_T= 20*np.log10(10)  # Transmitter gain
        G_R= 20*np.log10(10)  # Receiver gain
        tx_gain_db = G_T + G_R + gain_db  # Transmit gain in dB
        return tx_gain_db

    def tx_power_per_channel_dB(self):  # Calculate the transmit power per channel in dB
        Tx_Power_dBm = self.Transmit_Power_dBm()  # Get the transmit power in dBm
        Tx_power_watt_per_channel = 10**((Tx_Power_dBm - 30)/ 10) / self.sce.nChannel  # Transmit power in W, divided by the number of channels
        Tx_power_per_channel_dB = 10 * np.log10(Tx_power_watt_per_channel)  # Transmit power in dBm per channel
        return Tx_power_per_channel_dB  # Transmit power in dB per channel
    
    def Receive_Power(self, ue_id, d):  # Calculate the received power by transmit power and path loss of a certain BS
        Tx_power_per_channel_dB = self.tx_power_per_channel_dB()  # Transmit power per channel in dBm
        noise_dB = self.Noise_dB
        if self.BStype == "MBS":
            loss_dB = self.loss_mbs_db(ue_id, d)
            tx_gain_dB = self.tx_gain_mbs_db(ue_id, d)  # Transmit gain in dB
        elif self.BStype == "SBS":
            loss_dB = self.loss_sbs_db(ue_id,d)
            tx_gain_dB = self.tx_gain_sbs_db(ue_id)  # Transmit gain in dB
        elif self.BStype == "FBS":
            loss = 37 + 30 * np.log10(d)  

        Rx_power_dB = Tx_power_per_channel_dB + tx_gain_dB - loss_dB # Received power in dBm
        Rx_power = 10**(Rx_power_dB/10)  # Received power in W
        
        return Rx_power
        
        
class Scenario:  # Define the network scenario

    def __init__(self, sce):  # Initialize the scenario we simulate
        self.sce = sce
        self.Loc_MBS, self.Loc_SBS, self.Loc_FBS = self.BS_Location()
        self.BaseStations = self.BS_Init()
        global x_LOS_list, x_NLOS_list, z_list
        global x_list_sbs, z_list_sbs, R_list
        x_LOS_list, x_NLOS_list, z_list = random_factor_mbs(self.sce.nUsers)
        x_list_sbs, z_list_sbs, R_list = random_factor_sbs(self.sce.nUsers, self.sce.nSBS)
        # Initialize UEs
        self.UE_Locations = []
        self.UE_BS_Req_Rates = []
        self.UE_EL_Req_Rates = []
        for i in range(self.sce.nUsers):
            Loc_agent = np.zeros(2)
            if self.sce.nMBS > 0:
                LocM = self.Loc_MBS[0]
            else:
                LocM = np.array([0, 0])
            r = self.sce.rMBS * random()
            theta = uniform(-pi, pi)
            Loc_agent[0] = LocM[0] + r * np.cos(theta)
            Loc_agent[1] = LocM[1] + r * np.sin(theta)
            self.UE_Locations.append(Loc_agent)
            video_choice = np.random.randint(1, 4)
            self.UE_BS_Req_Rates.append(BL_assign(video_choice))
            self.UE_EL_Req_Rates.append(EL_assign(video_choice))
        
    def reset(self):   # Reset the scenario we simulate
        for i in range(len(self.BaseStations)):
            self.BaseStations[i].reset()
            
    def BS_Number(self):
        nBS = self.sce.nMBS + self.sce.nSBS + self.sce.nFBS  # The number of base stations
        return nBS
    
    def BS_Location(self):
        Loc_MBS = np.zeros((self.sce.nMBS,2))  # Initialize the locations of BSs
        Loc_SBS = np.zeros((self.sce.nSBS,2))
        Loc_FBS = np.zeros((self.sce.nFBS,2)) 
        
        # for i in range(self.sce.nMBS):
        #     Loc_MBS[i,0] = 500 + 900*i  # x-coordinate
        #     Loc_MBS[i,1] = 500  # y-coordinate
        
        # Simulation window parameters
        r = 700  # radius of disk
        xx0, yy0 = 0, 0  # centre of disk

        numbPoints1 = self.sce.nSBS  # number of Poisson points

        if numbPoints1 < 3:
            numbPoints1 = 3

        theta = 2 * np.pi * np.random.rand(numbPoints1)  # angular coordinates
        rho = r * np.sqrt(np.random.rand(numbPoints1))  # radial coordinates

        # Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)  # x coordinates of Poisson points
        yy = rho * np.sin(theta)  # y coordinates of Poisson points

        # Shift centre of disk to (xx0, yy0)
        xx = xx + xx0
        yy = yy + yy0
        for i in range(self.sce.nSBS):
            Loc_SBS[i,0] = xx[i]
            Loc_SBS[i,1] = yy[i]
            
        for i in range(self.sce.nFBS):
            LocM = choice(Loc_MBS)
            r = self.sce.rMBS*random()
            theta = uniform(-pi,pi)
            Loc_FBS[i,0] = LocM[0] + r*np.cos(theta)
            Loc_FBS[i,1] = LocM[1] + r*np.sin(theta)

        return Loc_MBS, Loc_SBS, Loc_FBS
    
    def BS_Init(self):   # Initialize all the base stations
        BaseStations = []  # The vector of base stations
        Loc_MBS, Loc_SBS, Loc_FBS = self.Loc_MBS, self.Loc_SBS, self.Loc_FBS
        
        for i in range(self.sce.nMBS):  # Initialize the MBSs
            BS_index = i
            BS_type = "MBS"
            BS_Loc = Loc_MBS[i]
            BS_Radius = self.sce.rMBS            
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, self.sce.MBS_BW))
            
        for i in range(self.sce.nSBS):
            BS_index = self.sce.nMBS + i
            BS_type = "SBS"
            BS_Loc = Loc_SBS[i]
            BS_Radius = self.sce.rSBS
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, self.sce.SBS_BW))
            
        for i in range(self.sce.nFBS):
            BS_index = self.sce.nMBS + self.sce.nSBS + i
            BS_type = "FBS"
            BS_Loc = Loc_FBS[i]
            BS_Radius = self.sce.rFBS
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, self.sce.FBS_BW))
        return BaseStations
            
    def Get_BaseStations(self):
        return self.BaseStations

    def Get_UE_Location(self, ue_id):
        return self.UE_Locations[ue_id]

    def Get_UE_Req_Rates(self, ue_id):
        return self.UE_BS_Req_Rates[ue_id], self.UE_EL_Req_Rates[ue_id]

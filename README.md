# Mobility
Simulation scripts for the mobility management of UAV base stations project mainly built for paper https://dl.acm.org/citation.cfm?id=3308964. 

# Requirements  
* python2.7  
* numpy==1.16.2
* tensorflow  
* IPython  
* matplotlib   

# Files
* main.py
  - main simulation script with A3C (V. Mnih et al. 2016. Asynchronous methods for deep reinforcement learning. In ICML. 1928–1937.) implementation   
  - multi-threading to initialise parallel training of multiple workers in parallel spaces (MobiEnvironments)  
  - each worker creates a MobiEnvironment instance and starts training in this environment   
  - there is a pair of global AC nets and local AC nets for worker. Workers train their own nets individually while push the gradients to the global nets periodically, then the global nets optimise uploaded gradients from all workers and distribute the same optimal gradients to all workers.  
  - choices of CNN and MLP are implimented. Default MLP nets perform as well as CNN in prior work with less training complexity  
  
* mobile_env.py  
  - followed openAI's gym implementation structure for a wireless mobile environment   
  - creates a LTE wireless channel which provides computation of SINR values and handover functionality   
  - step() and step_test() take action from the RL agent and returns updated state, reward, and customisable information to the RL agent. Please be careful here to make the two function consistant. It is not ideal to have two functions one for training and one for testing, but the reason to do this is to enable different user mobility models while keep both training and testing steps computationally cheap (rather than switching between if conditions per step)   
  - during training the user moves following the group reference model  
  - during testing the users move using preloaded trace (ue_trace_10k.npy), which is generated from the group reference model  
  - reward function currently consists of a reward on mean sinr value and a penalty on number of outaged users. which is open for improvement
 
* channel.py 
  - downlink and uplink SINR    
  - In the WAIN work we take only downlink sinr  

* ue_mobility.py 
  - a couple of mobility models for UE's movement  
  - group reference (X. Hong et al. 1999. A group mobility model for ad hoc wireless networks. In ACM MSWiM. 53–60.) model is used in the WAIN paper. please check the WAIN paper for more details    

* main_test.py    
  - load trained model to test (taking input AC model from ./train/Global_A_PARA%.npy where % can be the training step, 2000 by default)  
  - test is done on controlled ue mobility trace by loading a file ./ue_trace_10k.npy  
  - at each test step, the output of nn is argmax-ed to make control decisions of UAV movements   
  - per step reward, SINR, and computation time are recorded for performance evaluation (output to ./test)  

# Build virtual environment  
` virtualenv env  `  
` source env/bin/activate ` 

# Run training  
` mkdir train `   
` python main.py  `  

# Run testing
` mkdir test `    
` python main_test.py `

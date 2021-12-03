#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pickle import TRUE
from re import A
# os.environ["CUDA_VISIBLE_DEVICE"] = "0,1"

from collections import defaultdict
import random
import math
import re
import numpy as np

import argparse
from ns3gym import ns3env
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start simulation script 0/1, Default: 0')

parser.add_argument('--iterations',
                    type=int,
                    default=2,
                    help='Number of iterations, Default: 10')

parser.add_argument('--Agent',
                    type=int,
                    default=1,
                    help='Select Minstrel, Q-Learning Agent or crossentropy, 0/1/2 Default: 1, Q-Learning')

parser.add_argument('--verbose',
                    type=int,
                    default=1,
                    help='Show step state and action, Defult: 1')


parser.add_argument('--Mobility',
                    type=int,
                    default=1,
                    help='Statinary nodes or Dynamic 0/1, Default: 1')

parser.add_argument('--distance',
                    type=int,
                    default=40,
                    help='Set Distance, Default: 40')

parser.add_argument('--stat',
                    type=int,
                    default=0,
                    help='Collect Statistics 0/1, Default: 0')


parser.add_argument('--packetPerSec',
                    type=int,
                    default=60000,
                    help='Number of Packets per second, Default: 5000')

parser.add_argument('--energy',
                    type=float,
                    default=10.0,
                    help='Set vale for Energy Model: 30')

parser.add_argument('--channelWidth',
                    type=int,
                    default=160,
                    help='Channel Width, Default: 160MHz')
                  




args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)
Agent = int(args.Agent)
verbose = int(args.verbose)
distance = int(args.distance)
constant_distance = True if int(args.Mobility) == 1 else False
stat = int(args.stat)
packetPerSec = int(args.packetPerSec)
energy_ = float(args.energy)
channelWidth = int(args.channelWidth)




port = 1212
simTime = 10 # seconds
energy_ = float(10.0)
envStepTime = 0.005  # seconds
seed = 0
simArgs = {"--simTime": simTime,
           "--testArg": 123,
           "--nodeNum": 3,
           "--distance": distance,
           "--minstrel": False,
           "--constant_distance": constant_distance,
           "--packetPerSec": packetPerSec,
           "--energy_": energy_}


debug = False


env = ns3env.Ns3Env(port=port, stepTime=envStepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=True)
s=env.reset()
# env.close()




ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)
state_dim = (len(s)-2,)
print(state_dim)
peraction = 10
total = peraction**2
actions = np.vstack(np.array([[[n, o] for o in range(peraction)] for n in range(peraction)]))
n_actions = len(actions)
print("Number of Actions: ", n_actions)

import dill


def agent_init(Agent):
    if Agent == 1:
        with open("new_model.pk" , "rb") as f:
            agent = dill.load(f)
            agent.epilson = 0
            print(len(agent._qvalues))
            print(agent.epilson)
            # print(agent._qvalues)

              

    # elif Agent == 1:
    #     # with open("/home/predstan/repos/ns-3-allinone/ns-3.29/scratch/red/model.pk" , "rb") as f:
    #     #     agent = dill.load(f)

    #     agent = tf.keras.models.load_model("/home/predstan/repos/ns-3-allinone/ns-3.29/scratch/red/Tlmodel.h5")

    elif Agent == 2:
        with open("lambda0.5_model.pk" , "rb") as f:
            agent = dill.load(f)
            agent.epsilon = 0
            print(len(agent._qvalues))
            print(agent.epsilon)
        


    elif Agent == 3:
        with open("lambda0.2_model.pk" , "rb") as f:
            agent = dill.load(f)
            agent.epsilon = 0
            print(len(agent._qvalues))
            print(agent.epsilon)

    elif Agent == 0:
        agent = False
        

    return agent


def play_data(agents, environments = False, verbose=1):

    time = [0 for i in range(len(environments))]
    total_reward = [0.0 for i in range(len(environments))]
    s_env = [env.reset() for env in environments]
    energy = [0.0 for i in range(len(environments))]
    th = [0.0 for i in range(len(environments))]
    done_flag = [0 for i in range(len(environments))]
    batt = [0 for i in range(len(environments))]
    all_done = False

 
    while not all_done:

        for i in range(len(environments)):
            env = environments[i]
            if done_flag[i]:
                continue

            s = s_env[i]
            s[0] = getlevel((s[0]/5000)*100)

            # if i > 1:
                # s[0] = getlevel((s[0]/5000)*100)
            s[2] = int(s[2]/8)

            th[i] += s.pop()
            s.pop()

            s = [int(obs) for obs in s]

       

            if i > 0:
               a = agents[i].get_best_action(s)
            #    print("A", a)

            else:
                # print("True")
                a = 0

            next_s, r, done, _ = env.step(actions[a])

            if verbose:

                print("---time, obs, action, throughput, energy, done", i,  time[i], s, actions[a],  r, energy[i], done_flag[i])

            time[i]+=5

            s_env[i] = next_s

            energy[i] += r

            batt[i] = next_s[-3]
            
            if done:
                done_flag[i] =1

        for i in range(len(done_flag)):
            all_done = True
            if not done_flag[i]:
                all_done = False
                break

    return th, [np.round(en, 5) for en in energy], batt


def play(env, agent, q_agent, environments = False, verbose=1):
    """
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    time = 0
    total_reward = 0.0
    s = env.reset()
    energy = 0.0
    th=0

    while True:
 
        s[0] = getlevel((s[0]/5000)*100)
        if q_agent >1:
            s[0] = getlevel((s[0]/5000)*100)

        th+=s.pop()
        energy += s.pop()/100000000

        s = [int(obs) for obs in s]

        # if Q agent to pick action given state s.
        if q_agent >= 1:
      
            a = agent.get_best_action(s)
            # print(a)
            # print("here")
            

        # elif q_agent == 2:
        #     a = np.argmax(agent.predict(np.array([s])))
        else:
            a = np.random.choice(range(len(actions)))
        

        if not agent:
            a=0


        next_s, r, done, _ = env.step(actions[a])

        if verbose:

            print("---time, obs, action, throughput, energy, done ", time, s, actions[a],  r, energy, done)

        time+=10

        s = next_s
        total_reward += r
        batt = next_s[-1]
        
        if done:
            break

    return th, np.round(energy, 5), batt

import dill
def main(env=env, Agent=Agent, verbose=verbose, collectdata=stat):
    global port, envStepTime, startSim, seed, debug

    if not collectdata:

        agent=agent_init(Agent)

        reward, energy, batt = play(env=env, agent=agent, q_agent=Agent, verbose=verbose)

        print(reward, energy, batt)


        print("reward = %.3f"% (reward))
        print("Throughput= %.3f", (reward * 1500 * 10) / 10 / (1024 *1024))

        print(f"Energy Expended = {energy}J")

     


    else:
        environments = []
        agents = [agent_init(a) for a in range(4)]

        ports = [3333, 4444, 5555, 6666]

        distances = [50]
        packets = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000 ]
        # packets = [400, 600, 800, 1000]
        dynamic = [1]
        # dyna = {}
        # energy_record = {}
        battery = {}
        for mobility in dynamic:

            constant_distance = True if mobility == 1 else False
            station = "stationary" if mobility == 1 else "Dynamic"
            # dyna[mobility] = {}
            # energy_record[mobility] = {}
            battery[mobility] = {}
            for distance in distances:
               

                dyna[mobility][distance] = {}
                energy_record[mobility][distance] = {}
                battery[mobility][distance] = {}

                for i, agent in enumerate(agents):
                        dyna[mobility][distance][i] = []
                        energy_record[mobility][distance][i] = []
                        battery[mobility][distance][i] = []


                for packetPerSec in packets:
                        # packetPerSec/=10
                    for env in environments: 
                        env.close() 

                    environments = []


                    for i, agent in enumerate(agents):
                        minstrel = True if i == 0 else False

                    
                    

                        print("HERE")
                        port = ports[i]

                        simTime = 10 # seconds

                        energy_ = float(10)

                        envStepTime = 0.005  # seconds
                        seed = 0
                        simArgs = {"--simTime": simTime,
                                "--testArg": 123,
                                "--nodeNum": 3,
                                "--distance": distance,
                                "--minstrel": minstrel,
                                "--constant_distance": constant_distance,
                                "--packetPerSec": packetPerSec,
                                "--energy_": energy_}


                        debug = False
                        
                        env = ns3env.Ns3Env(port=port, stepTime=envStepTime, startSim=1, simSeed=seed, simArgs=simArgs, debug=debug)

                        environments.append(env)
                    


                    print("lot")
                    reward=[]
                    Energy=[]
                    batt=[]


                    for _ in range(5):
                
                        r, e, b = play_data(agents, environments = environments, verbose=0)
                        
                        reward.append(r)
                        Energy.append(e)
                        batt.append(b)

                        if _ == 0:
                            print(packetPerSec, distance, end=": ")
                        print(_, end=", ")


                    reward = np.mean(reward, axis=0)
                    Energy = np.mean(Energy, axis=0)
                    batt = np.mean(batt, axis = 0)
                
                
        
                    throughput = [(r * 1100 * 10) / 10 / (1024 *1024) for r in reward]

                    for i, agent in enumerate(agents):

                        dyna[mobility][distance][i].append(throughput[i])

                        energy_record[mobility][distance][i].append(Energy[i])

                        battery[mobility][distance][i].append(batt[i])
            

                plt.figure(figsize=[15, 5])
                for i in range(1, 3):
                    plt.subplot(1, 2, i)
                    if i == 1:
                        for th in dyna[mobility][distance].keys():
                            plt.plot(packets, dyna[mobility][distance][th] )
                            plt.scatter(packets, dyna[mobility][distance][th],s=10,color='red',zorder=2)
                        #plt.title(f'Throughput for {distance} meters distance with {station} Mobility')
                            plt.ylabel('Throughput (Mbps)')
                            plt.legend(['Minstrel Algorithm','Agent Lambda=0.8', "Agent Lambda=0.5", 'Agent Lambda=0.2'],loc='upper left')
                    else:
                        for th in energy_record[mobility][distance].keys():
                            plt.plot(packets, energy_record[mobility][distance][th] )
                            plt.scatter(packets,energy_record[mobility][distance][th],s=10,color='red',zorder=2)
                        #plt.title(f'Throughput for {distance} meters distance with {station} Mobility')
                            plt.ylabel('Energy Expended (J)')
                            plt.legend(['Minstrel Algorithm','Agent Lambda=0.8', "Agent Lambda=0.5", 'Agent Lambda=0.2'],loc='upper left')

                    plt.xlabel('Packet Per Second')
                    
                    
                # plt.show()
                plt.savefig(f'result/{distance}m--{station}.png')
                plt.clf()
        
                with open("result/Throughput_.pk" , "wb") as f:
                    dill.dump(dyna, f)

                with open("result/Energy_.pk" , "wb") as f:
                    dill.dump(energy_record, f)

def getlevel(value):
    compare =5


    if (value > 90):
      return 100

    elif (value > 80):
      return 90
     
     
    elif (value > 70):
      return 80
     
    elif (value > 60):
      return 70
     
    elif (value > 50):
      return 60
  
     
    elif (value > 40):
      return 50
   
    elif (value > 30):
      return 40
     

    elif (value > 20):
      return 30
     
    elif (value > 10):
      return 20
     
    else:
      return 10


with open("result/Throughput_.pk" , "rb") as f:
        dyna = dill.load(f)

with open("result/Energy_.pk" , "rb") as f:
       energy_record =  dill.load(f)

       
main(collectdata=True)





# Abstract
Increased throughput and energy efficiency of wireless devices are achievable when
factors such as interference reduction, power management, and data transmission rate
selection are considered. This thesis proposes an algorithm for optimizing the performance
of wireless networks using reinforcement learning. The algorithm, a.k.a. the agent, observes
the state of a wireless device’s battery, packet queue, transmission medium, etc. and
establishes the optimal policy for joint control of transmission power and speed. Using
the NS3 network simulation software, we implement agents focusing on three different
reward functions: throughput-critical, energy-critical, and throughput and energy balanced.
We compare their performance to the conventional Minstrel rate adaptation algorithm:
our approach can achieve (i) higher throughput when using the throughput-critical reward
function; (ii) lower energy consumption when using the energy-critical reward function; and
(iii) higher throughput and roughly the same energy when using the throughput and energy
balanced reward function.

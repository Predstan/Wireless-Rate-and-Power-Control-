/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

//    

//   BulkSendApplication                                 BulkSendApplication 
//         SOURCE                 DESTINATION              INTERFERENCE         
//         +-----+                 +--------+                +-----+
// //      | SRC |                  | DST |                  | INT |
// //      +-----+                 +--------+                +-----+
// //     10.1.1.0 ---Distance-->  10.1.1.1  <---Distance---10.1.1.2
// //     --------                 ----------               ----------
// //  WIFI  802.11AX             WIFI 802.11AX             WIFI 802.11AX 
// //     --------                 --------                 ---------
// //       ((*))                    ((*))                    ((*))


/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */


#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/opengym-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/stats-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/node-list.h"
#include <sstream>
#include <iostream>
#include "ns3/wifi-radio-energy-model-helper.h"
#include "ns3/energy-module.h"
#include <math.h> 

#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OpenGym");
float prevstatus = 0.0;
std::vector<std::string> dataRates;
int simu_time = 10.0;
int max_queue = 5000;
uint32_t payloadSize = 1472;
Ptr<FlowMonitor> monitor;
FlowMonitorHelper flowmon;
bool minstrel = true;
bool constant_distance = true;
double lambda_ ;
uint32_t pktPerSec = 60000;
double remain_energy = 10.0;
double Total_energy = 0.0;
uint32_t packet_accept;

/// Trace function for remaining energy at node.
void
RemainingEnergy (double oldValue, double remainingEnergy)
{
	NS_LOG_DEBUG (Simulator::Now ().GetSeconds ()
	                << "s Current remaining energy = " << remainingEnergy << "J");
	remain_energy = remainingEnergy;
}

/// Trace function for total energy consumption at node.
float old;
void
TotalEnergy (double oldValue, double totalEnergy)
{
	 NS_LOG_DEBUG (Simulator::Now ().GetSeconds ()
	                << "s Total energy consumed by radio = " << totalEnergy << "J");
	
	
	Total_energy=totalEnergy-oldValue;
	old = totalEnergy;
}

/// Trace function for total packets Delivered at Destination.
uint64_t g_rxPktNum = 0;
void DestRxPkt (Ptr<const Packet> packet)
{
	NS_LOG_DEBUG ("Client received a packet of " << packet->GetSize () << " bytes");
	g_rxPktNum++;
}


	/*
Define observation space
*/
Ptr<OpenGymSpace>
GetObservationSpace()
{
	// uint32_t nodeNum = NodeList::GetNNodes ();
	float low = 0.0;
	float high = (float)max_queue;
	std::vector<uint32_t> shape = {5};
	std::string dtype = TypeNameGet<uint32_t> ();
	Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
	NS_LOG_DEBUG ("GetObservationSpace: " << space);
	return space;
}


/*
Define action space
*/
Ptr<OpenGymSpace>
GetActionSpace()
{
	uint32_t low = 0;
	uint32_t high = 9;
	std::vector<uint32_t> shape = {2};
	std::string dtype = TypeNameGet<uint32_t> ();
	Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
	NS_LOG_DEBUG ("GetActionSpace: " << space);
	return space;
}

double energy_ = 10.0;
uint32_t energy_left = 0;
uint32_t pkt_drop = 0;
uint32_t pkt_old = 0;
uint32_t getlevel(uint32_t value){	
	if (value > 90){
		return 100;
	}
	
	else if (value > 80){
		return 90;
	}
	
	else if (value > 70){
		return 80;
	}
	
	else if (value > 60){
		return 70;
	}
	
	else if (value > 50){
		return 60;
	}
	
	else if (value > 40){
		return 50;
	}
	
	else if (value > 30){
		return 40;
	}
	
	else if (value > 20){
		return 30;
	}
	
	else if (value > 10){
		return 20;
	}
	
	else {
		return 10;
	}	
}

/*
Define game over condition
*/

bool GetGameOver(void)
{
	bool isGameOver = false;
	NS_LOG_DEBUG ("MyGetGameOver: " << isGameOver);
	Time t = Simulator::Now ();
	energy_left = (uint32_t)remain_energy/energy_*100;
	if (( getlevel(energy_left) <= 10.0) || (pkt_drop >= packet_accept)){
		isGameOver = true;
	}
	return isGameOver;
}

/*
Helper Function for TX Operation
*/
Ptr<QosTxop> GetTxop(Ptr<Node> node)
{
	Ptr<NetDevice> dev = node->GetDevice (0);
	Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
	Ptr<WifiMac> rmac = wifi_dev->GetMac ();
	PointerValue ptr;
	rmac->GetAttribute ("BE_Txop", ptr);
	Ptr<QosTxop> txop = ptr.Get<QosTxop> ();
	
	return txop;
}

uint32_t queue_p;
/*
Collect observations
*/
Ptr<OpenGymDataContainer>
GetObservation()
{
	static float lastValue = 0.0;
	
	uint32_t nodeNum = NodeList::GetNNodes ();
	std::vector<uint32_t> shape = {(5*(nodeNum-2)+1),};
	Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(shape);

	for (uint32_t i=0; i< NodeList::GetNNodes()-2; i++) {
		Ptr<Node> node  = NodeList::GetNode(i);
		Ptr<NetDevice> device = node->GetDevice(0);
		Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice> (device);
		Ptr<SpectrumWifiPhy> phy = DynamicCast<SpectrumWifiPhy>(wifiDevice->GetPhy());	
		Ptr<QosTxop> txop = GetTxop(node);
		Ptr<WifiMacQueue> queue = txop->GetWifiMacQueue ();
		uint32_t packetqueue = queue->GetNPackets();
		queue_p = packetqueue;
		box->AddValue(packetqueue);
		uint32_t slot = txop->GetBackoffSlots();
		uint32_t cw = txop->GetCw();
		if((g_rxPktNum - lastValue) > 0){
			prevstatus =1;
		}
		else{
			prevstatus = 0;
		}
		int th = g_rxPktNum - lastValue;
		lastValue = g_rxPktNum;
		box->AddValue(cw);
		box-> AddValue((int)slot/8);
		box->AddValue(prevstatus);
		energy_left = (uint32_t)remain_energy/energy_*100;
		box->AddValue(getlevel(energy_left));
		box->AddValue(pkt_drop);
		box->AddValue(th);
	}
	NS_LOG_DEBUG ("MyGetObservation: " << box);
	return box;
}
/*
Define reward function
*/
double pks;
double alpha = 0.2;
float GetReward(void)
{
	static float lastValue = 0.0;
	float reward = g_rxPktNum - lastValue;
	Time t = Simulator::Now ();
	lastValue = g_rxPktNum;

	NS_LOG_UNCOND("Packets "<< reward << "Energy " << Total_energy);
	reward = lambda_ * (reward/(pktPerSec *simu_time)) *100  + ((1-lambda_) * (-Total_energy/energy_*100));
	
	if (isnan(reward)){return 0.0;}
	
	return reward;
}


/*
Define extra info. Optional
*/
std::string GetExtraInfo(void)
{
	std::string myInfo = "Tx-to-Rx-Wifi-Simulation";
	myInfo += "|123";
	NS_LOG_DEBUG("MyGetExtraInfo: " << myInfo);
	return myInfo;
}

/*
Helper Function for Changing DataRates
*/		
bool resetRate(std::string data){
	
	if (!minstrel){
		
		Config::Set("NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/DataMode", StringValue (data));
		Config::Set("NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/ControlMode", StringValue (data));
	}
	NS_LOG_DEBUG("MCS set to" << data);
	return true;
}

/*
Helper Function for Changing Transmit Power
*/		
bool resetPower(uint32_t data, Ptr<Node> node){

	if (!minstrel){
		
		Ptr<NetDevice> device = node->GetDevice(0);
		Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice> (device);
		Ptr<SpectrumWifiPhy> phy = DynamicCast<SpectrumWifiPhy>(wifiDevice->GetPhy());
		NS_LOG_DEBUG ("Power" <<phy->GetTxPowerStart());
		phy-> SetTxPowerStart( (data+1)*10.0);
		phy->SetTxPowerEnd( (data+1)*10.0);
		phy-> SetNTxPower(1);
		NS_LOG_DEBUG("Changed Transmission Power to:"<< phy->GetTxPowerStart());
	}
	return true;
}


/*
Execute received actions
*/
bool
ExecuteActions(Ptr<OpenGymDataContainer> action)
{

	NS_LOG_DEBUG ("MyExecuteActions: " << action);
	Ptr<OpenGymBoxContainer<uint32_t> > box = DynamicCast<OpenGymBoxContainer<uint32_t> >(action);
	std::vector<uint32_t> actionVector = box->GetData();	
	Ptr<Node> node = NodeList::GetNode(0);
	uint32_t power = actionVector.at(1);
	uint32_t id = actionVector.at(0);
	for (uint32_t i=0; i< NodeList::GetNNodes()-2; i++) {
		Ptr<Node> node  = NodeList::GetNode(i);
		resetRate(dataRates.at(id));
		resetPower(power, node);
	}
	return true;
}

/*
Schedule Next State
*/
void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGymInterface)
{
	Simulator::Schedule (Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGymInterface);
	openGymInterface->NotifyCurrentState();
}
/*
Get Packet Dropped
*/
static void
RxDrop (Ptr<const Packet> p)
{
	NS_LOG_DEBUG (pkt_drop << " RxDrop at " << Simulator::Now ().GetSeconds ());
	uint32_t pk_drop = pkt_drop;
	pkt_drop++;
	pkt_old = pkt_drop - pk_drop;
}


int
main (int argc, char *argv[])
{
	// Parameters of the environment
	uint32_t simSeed = 0;
	double simulationTime = 10; //seconds
	simu_time = simulationTime;
	double envStepTime = 0.005; //seconds, ns3gym env step time interval
	uint32_t openGymPort = 1212; // Ns3gym Port
	uint32_t testArg = 0;
	
	//Parameters of the scenario
	uint32_t nodeNum = 3;
	double distance = 10.0;
	bool noErrors = false;
	std::string errorModelType = "ns3::NistErrorRateModel";
	bool enableFading = true;
	
	uint32_t channelWidth = 160;
	
	// define datarates
	
	dataRates.push_back("VhtMcs0");
	dataRates.push_back("VhtMcs1");
	dataRates.push_back("VhtMcs2");
	dataRates.push_back("VhtMcs3");
	dataRates.push_back("VhtMcs4");
	dataRates.push_back("VhtMcs5");
	dataRates.push_back("VhtMcs6");
	dataRates.push_back("VhtMcs7");
	dataRates.push_back("VhtMcs8");
	dataRates.push_back("VhtMcs9");
	uint32_t dataRateId = 0;
	
	
	CommandLine cmd;
	// required parameters for OpenGym interface
	cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 1111", openGymPort);
	cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
	// optional parameters
	cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
	cmd.AddValue ("envStepTime", "Environment Step Time", envStepTime);
	cmd.AddValue ("nodeNum", "Number of nodes. Default: 2", nodeNum);
	cmd.AddValue ("distance", "Inter node distance. Default: 10m", distance);
	cmd.AddValue ("packetPerSec", "Inter node distance. Default: 10000", pktPerSec);
	cmd.AddValue ("channelWidth", "ChannelWidth. Default:40", channelWidth);
	cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
	cmd.AddValue ("minstrel", "Enable Minstrel Algorithm. Default: False", minstrel);
	cmd.AddValue ("constant_distance", "Enable Constant Mobility. Default: True", constant_distance);
	cmd.AddValue ("energy_", "Energy Model", energy_);
	cmd.AddValue ("lambda_", "Energy Model", lambda_);
	cmd.Parse (argc, argv);
	
	
	pks = pktPerSec;
	NS_LOG_UNCOND("Ns3Env parameters:");
	NS_LOG_UNCOND("--simulationTime: " << simulationTime);
	NS_LOG_UNCOND("--openGymPort: " << openGymPort);
	NS_LOG_UNCOND("--envStepTime: " << envStepTime);
	NS_LOG_UNCOND("--seed: " << simSeed);
	NS_LOG_UNCOND("--distance: " << distance);
	NS_LOG_UNCOND("--testArg: " << testArg);
// Packet Accetance Percentage
	packet_accept = (uint32_t) pktPerSec * simulationTime * 0.005;
	remain_energy = energy_;
	
	if (noErrors){
		errorModelType = "ns3::NoErrorRateModel";
	}
	
	
	
	RngSeedManager::SetSeed (1);
	RngSeedManager::SetRun (simSeed);
	
	// Configuration of the scenario
	// Create Nodes
	NodeContainer nodes;
	nodes.Create (nodeNum);
	
	// WiFi device
	WifiHelper wifi;
	wifi.SetStandard (WIFI_PHY_STANDARD_80211ac);
	
	// Channel
	SpectrumWifiPhyHelper spectrumPhy = SpectrumWifiPhyHelper::Default ();
	Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
	
	spectrumPhy.SetChannel (spectrumChannel);
	spectrumPhy.SetErrorRateModel (errorModelType);
	spectrumPhy.Set ("Frequency", UintegerValue (5200));
	spectrumPhy.Set ("ChannelWidth", UintegerValue (channelWidth));
	spectrumPhy.Set ("ShortGuardEnabled", BooleanValue (true));
	
	Config::SetDefault ("ns3::WifiPhy::CcaMode1Threshold", DoubleValue (-82.0));
	Config::SetDefault ("ns3::WifiPhy::Frequency", UintegerValue (5200));
	Config::SetDefault ("ns3::WifiPhy::ChannelWidth", UintegerValue (channelWidth));
	
	// Channel
	Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
	Ptr<NakagamiPropagationLossModel> fadingModel = CreateObject<NakagamiPropagationLossModel> ();
	
	if (enableFading) {
		lossModel->SetNext (fadingModel);
	}
	spectrumChannel->AddPropagationLossModel (lossModel);
	Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
	spectrumChannel->SetPropagationDelayModel (delayModel);
	
	// Add MAC and set DataRate
	WifiMacHelper wifiMac;
	
	if (minstrel) {
// Set Minstrel Algroithm for Minstrel Agent
		wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");
	} 
	else {
// Set Constant Rate for Reinforcement Agent
		std::string dataRateStr = dataRates.at(dataRateId);
		NS_LOG_DEBUG("dataRateStr: " << dataRateStr);
		wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager");
		
	};
	
	double Prss = -80;  // -dBm
	double Irss = -95;  // -dBm
	double offset = 91; // -dBm
	
	// Set it to adhoc mode
	wifiMac.SetType ("ns3::AdhocWifiMac",
	"QosSupported", BooleanValue (false));
	
	// Install wifi device
	NetDeviceContainer devices = wifi.Install (spectrumPhy, wifiMac, nodes.Get(0));
	
	spectrumPhy.Set ("TxGain", DoubleValue (offset + Prss) );
	devices.Add (wifi.Install (spectrumPhy, wifiMac, nodes.Get (1)));
	
	spectrumPhy.Set ("TxGain", DoubleValue (offset + Irss) );
	devices.Add (wifi.Install (spectrumPhy, wifiMac, nodes.Get (2)));
	
	// Mobility model
	MobilityHelper mobility;
	mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
	"MinX", DoubleValue (0.0),
	"MinY", DoubleValue (0.0),
	"DeltaX", DoubleValue (distance),
	"DeltaY", DoubleValue (distance),
	"GridWidth", UintegerValue (nodeNum),  // will create linear topology
	"LayoutType", StringValue ("RowFirst"));
	
	
	if (constant_distance){

		mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
		mobility.Install (nodes);
	}
	
	else {
		
		mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
		mobility.Install (nodes.Get(0));
		
		mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
		"Bounds", RectangleValue (Rectangle (-50, 50, -50, 50)));
		mobility.Install (nodes.Get(1));
	}
	// IP stack and routing
	InternetStackHelper internet;
	internet.Install (nodes);
	
	// Assign IP addresses to devices
	Ipv4AddressHelper ipv4;
	NS_LOG_INFO ("Assign IP Addresses");
	ipv4.SetBase ("10.1.1.0", "255.255.255.0");
	Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);
	
	// //Configure static multihop routing
	for (uint32_t i = 0; i < nodes.GetN()-1; i++)
	{
		  Ptr<Node> src = nodes.Get(i);
		  Ptr<Node> nextHop = nodes.Get(i+1);
		  Ptr<Ipv4> destIpv4 = nextHop->GetObject<Ipv4> ();
		  Ipv4InterfaceAddress dest_ipv4_int_addr = destIpv4->GetAddress (1, 0);
		  Ipv4Address dest_ip_addr = dest_ipv4_int_addr.GetLocal ();
		
		  Ptr<Ipv4StaticRouting>  staticRouting = Ipv4RoutingHelper::GetRouting <Ipv4StaticRouting> (src->GetObject<Ipv4> ()->GetRoutingProtocol ());
		  staticRouting->RemoveRoute(1);
		  staticRouting->SetDefaultRoute(dest_ip_addr, 1, 0);
		}
	
	// Traffic
	// Create a BulkSendApplication and install it on node 0
	Ptr<UniformRandomVariable> startTimeRng = CreateObject<UniformRandomVariable> ();
	startTimeRng->SetAttribute ("Min", DoubleValue (0.0));
	startTimeRng->SetAttribute ("Max", DoubleValue (1.0));
	
	uint16_t port = 1000;
	uint32_t srcNodeId = 0;
	uint32_t destNodeId = nodes.GetN() - 1;
	Ptr<Node> srcNode = nodes.Get(srcNodeId);
	Ptr<Node> dstNode = nodes.Get(destNodeId);
	
	Ptr<Ipv4> destIpv4 = dstNode->GetObject<Ipv4> ();
	Ipv4InterfaceAddress dest_ipv4_int_addr = destIpv4->GetAddress (1, 0);
	Ipv4Address dest_ip_addr = dest_ipv4_int_addr.GetLocal ();
	
	InetSocketAddress destAddress (dest_ip_addr, port);
	destAddress.SetTos (0x70); //AC_BE
	UdpClientHelper source (destAddress);
	source.SetAttribute ("MaxPackets", UintegerValue (pktPerSec * simulationTime));
	source.SetAttribute ("PacketSize", UintegerValue (payloadSize));
	Time interPacketInterval = Seconds (1.0/pktPerSec);
	source.SetAttribute ("Interval", TimeValue (interPacketInterval)); //packets/s
	
	for (uint32_t i=0; i< NodeList::GetNNodes()-2; i++) {
		
		Ptr<Node> node  = NodeList::GetNode(i);
		Ptr<NetDevice> dev = node->GetDevice (0);
		Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
		Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
		Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
		
		PointerValue ptr;
		wifi_mac->GetAttribute ("BE_Txop", ptr);
		Ptr<QosTxop> txop = ptr.Get<QosTxop> ();
		Ptr<WifiMacQueue> queue = txop->GetWifiMacQueue ();
		std::string queuelen = std::to_string(max_queue)+"p";
		QueueSize size(queuelen);
		queue->SetMaxSize(size);
		queue->SetMaxDelay(Seconds(3));
		
	}
	
	BasicEnergySourceHelper basicSourceHelper;
	// configure energy source
	basicSourceHelper.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (energy_));
	// install source
	EnergySourceContainer sources = basicSourceHelper.Install (nodes.Get(0));
	/* device energy model */
	WifiRadioEnergyModelHelper radioEnergyHelper;
	// configure radio energy model
	radioEnergyHelper.Set ("TxCurrentA", DoubleValue (0.25));
	// install device model
	DeviceEnergyModelContainer deviceModels = radioEnergyHelper.Install (devices.Get(0), sources);
	monitor = flowmon.InstallAll();
	
	
	TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
	Ptr<Socket> interferer = Socket::CreateSocket (nodes.Get (2), tid);
	InetSocketAddress interferingAddr = InetSocketAddress (Ipv4Address ("255.255.255.255"), 49000);
	interferer->SetAllowBroadcast (true);
	interferer->Connect (interferingAddr);
	
	ApplicationContainer sourceApps = source.Install (srcNode);
	sourceApps.Start (Seconds (0.0));
	sourceApps.Stop (Seconds (simulationTime));
	
	// Create a packet sink to receive these packets
	UdpServerHelper sink (port);
	ApplicationContainer sinkApps = sink.Install (dstNode);
	sinkApps.Start (Seconds (0.0));
	sinkApps.Stop (Seconds (simulationTime));
	
	Ptr<UdpServer> udpServer = DynamicCast<UdpServer>(sinkApps.Get(0));
	udpServer->TraceConnectWithoutContext ("Rx", MakeCallback (&DestRxPkt));
	
	Ptr<BasicEnergySource> basicSourcePtr = DynamicCast<BasicEnergySource> (sources.Get (0));
	basicSourcePtr->TraceConnectWithoutContext ("RemainingEnergy", MakeCallback (&RemainingEnergy));
	// device energy model
	Ptr<DeviceEnergyModel> basicRadioModelPtr =
	basicSourcePtr->FindDeviceEnergyModels ("ns3::WifiRadioEnergyModel").Get (0);
	NS_ASSERT (basicRadioModelPtr != NULL);
	basicRadioModelPtr->TraceConnectWithoutContext ("TotalEnergyConsumption", MakeCallback (&TotalEnergy));

	// Print node positions
	NS_LOG_UNCOND ("Node Positions:");
	for (uint32_t i = 0; i < nodes.GetN(); i++)
	{
		Ptr<Node> node = nodes.Get(i);
		Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
		NS_LOG_UNCOND ("---Device ID: " << node->GetId() << " Positions: " << mobility->GetPosition());
	}

	Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxDrop", MakeCallback(&RxDrop));
	Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxDrop", MakeCallback(&RxDrop));
	Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRxDrop", MakeCallback(&RxDrop));
	
	
	// OpenGym Env
	Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
	openGymInterface->SetGetActionSpaceCb( MakeCallback (&GetActionSpace) );
	openGymInterface->SetGetObservationSpaceCb( MakeCallback (&GetObservationSpace) );
	openGymInterface->SetGetGameOverCb( MakeCallback (&GetGameOver) );
	openGymInterface->SetGetObservationCb( MakeCallback (&GetObservation) );
	openGymInterface->SetGetRewardCb( MakeCallback (&GetReward) );
	openGymInterface->SetGetExtraInfoCb( MakeCallback (&GetExtraInfo) );
	openGymInterface->SetExecuteActionsCb( MakeCallback (&ExecuteActions) );
	
	Simulator::Schedule (Seconds(0.0), &ScheduleNextStateRead, envStepTime, openGymInterface);
	
	NS_LOG_UNCOND ("Simulation start");
	Simulator::Stop (Seconds (simulationTime));
	
	Simulator::Run ();
	
	monitor->CheckForLostPackets (); 
	
	Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
	std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
	
	for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator iter = stats.begin (); iter != stats.end (); ++iter)
	{
		Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (iter->first);
		
		NS_LOG_UNCOND("Flow ID: " << iter->first << " Src Addr " << t.sourceAddress << " Dst Addr " << t.destinationAddress);
		NS_LOG_UNCOND("Tx Packets = " << iter->second.txPackets);
		NS_LOG_UNCOND("Rx Packets = " << iter->second.rxPackets);
		NS_LOG_UNCOND("Lost Packets = " << iter->second.lostPackets);
		NS_LOG_UNCOND("Jitter Sum= " << iter->second.delaySum);
		NS_LOG_UNCOND("Throughput: " << iter->second.rxBytes << " " << iter->second.rxBytes * 8.0 / (iter->second.timeLastRxPacket.GetSeconds()-iter->second.timeFirstTxPacket.GetSeconds()) / (1024*1024)  << " Mbps");
	}

	NS_LOG_UNCOND (packet_accept << "Simulation stop");
	
	monitor->SerializeToXmlFile("NameOfFile.xml", true, true);
	
	openGymInterface->NotifySimulationEnd();
	

	Simulator::Destroy ();
	
	
}

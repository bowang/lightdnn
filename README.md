#LightDNN
LightDNN is a distributed deep learning framework for large-scale neural network training and testing.

**Table of Contents** 

- [1. Design Principles](#1-design-principles)
- [2. System Architecture](#2-system-architecture)
  - [2.1. Client](#21-client)
  - [2.2. Scheduler](#22-scheduler)
  - [2.3. Worker](#23-worker)
  - [2.4. Parameter Server](#24-parameter-server)
- [3. Fault Tolerance](#3-fault-tolerance)
  - [3.1 Scheduler](#31-scheduler)
  - [3.2 Worker](#32-worker)
  - [3.3 Parameter server](#33-parameter-server)
- [4. Implementation](#4-implementation)
  - [4.1 Communication](#41-communication)
  - [4.2 Graph partition](#42-graph-partition)
  - [4.3 Fault tolerance](#43-fault-tolerance)
- [5. Implementation Plan](#5-implementation-plan)
- [References](#references)

### 1. Design Principles
 - **Distributed**: LightDNN is designed with a distributed architecture in mind at the very first line of code. Both data parallelism and model parallelism are exploited on heterogeneous clusters.
 - **High Performance**: LightDNN unleashes the performance by utilizing multicore CPUs and GPUs. Remote and local I/O are handled via asynchronous communication to minimize the overhead.
 - **Intelligent**: LightDNN models the performance of servers and network to make the optimal partition on workloads. Workloads can be dynamically migrated to minimize the effects of slow outliers.
 - **Fault Tolerant**: LightDNN is resistive to failures in distributed settings. It leverages replication and re-computation to minimize the cost of server breakdown.
 - **Extensible**: LightDNN has a modular design. It is easy to add new models or modify existing ones.

### 2. System Architecture
This section describes different components of LightDNN. The system is logically partitioned into two tiers: the upper tier handles the data parallelism; the lower tier handles the model parallelism.

#### 2.1. Client
The `client` is a light-weight process running on the user's local machine. The major responsibilities of a client are as follows.

 - Parse the job config file
 - Submit the job to the `scheduler`
 - Print the progress reported by the `scheduler`

The `client` process terminates when the training/testing job is done. If the `client` is killed before the job is done, the job is still running on the servers. The results can be checked on a web interface.

#### 2.2. Scheduler
The `scheduler` is responsible to:

 - Partition a neural net into multiple subnets
 - Assign one or more subnets to each `worker`
 - Decompose the training/testing job into a DAG of tasks based on the network partition
 - Schedule tasks on each `worker` based on subnet assignment
 - Keep track of the task execution and report progress to the `client`

**Model Parallelism**

*Inter-layer partition*:
Once a job is submitted to the `scheduler`, the specified neural network is partitioned into *n* sub-networks of similar sizes (or proportional to the computing capabilities of the `worker` to be assigned to) while minimizing the amount of inter-machine communication. Each sub-network consists of one or more connected layers. A sub-network is the minimum unit to assign to a `worker`.

*Intra-layer partition*:
Besides partitioning a network across layers, some large layers (e.g. fully-connected layers) may also be partitioned and assigned to multiple `worker` nodes if the sizes exceed the memory limit. As shown in [1], convolutional layers typically consist of about 5% of the parameters of a convolutional neural network while taking 90-95% of the computation. In contrast, fully-connected layers contain about 95% of the parameters but only 5-10% of the computation. Thus, a feasible strategy is to partition these two types of layers in different manners - model parallelism for fully-connected layers and data parallelism for convolutional layers.

**Network Partition**

Partitioning the neural network can be formalized as a *weighted graph partitioning* problem [2].

 - Each layer (or parts of a layer) can be viewed as a vertex.
 - The data flow of the network forms the edges.
 - The weight of a vertex is the computational workload, which can be characterized by the size (height, width, #channels) and transformation (e.g. fully-connected, convolutional, sigmoid, etc.) of a layer.
 - The weight of an edge is the communication workload, which can be characterized by the amount of data (e.g. activations, deltas) to send or receive from other layers.
 - The ideal model parallelism is achieved by partitioning the weighted graph (almost) equally while minimizing sum of weights of cross-partition edges.

The workload modeling can reference some prior works on MapReduce [3, 4]. However, the modeling in LightDNN should be more precise since both forward and backward operations are essentially dense matrix multiplication, which could be accurately characterized by the input size.

**Server Performance Characterization**
A cluster can be heterogeneous due to various reasons: different configurations (e.g. with/without GPUs), different generations of hardware, different network bandwidths, etc. To make the best scheduling decision, it is desirable to have a profile of servers. A server can be characterized by its CPU speed, number of cores, size of memory, I/O performance, etc. A per-server performance model is updated online according to the runtime of different sizes of tasks.

**Task Scheduling**
A training/testing job is decomposed into a DAG of tasks according to the network partition. The `scheduler` schedules tasks onto `worker` nodes as a workflow execution. A task includes the following information:

 - *Subnet*: the network partition to perform the task on
 - *Operation*: feed-forward or back-propagate on the subnet
 - *Inputs*: identifiers of the input data
 - *Outputs*: where to send the data generated by the operation

In a feed-forward operation, a mini-batch of *N* activations (or input data samples) are fed into a subnet. After propagating through all layers of the subnet, the output activations are sent to the `worker` responsible for the subsequent subnets via asynchronous network channels. Once a `worker` node notifies the `master` of the task completion, the `master` schedules another task (e.g. the next mini-batch) on this node. Back-propagation is scheduled in a similar fashion, except for updating parameters with the `parameter server`.

#### 2.3. Worker
A LightDNN cluster consists of a large number of `worker` nodes. Worker nodes are scheduled by the `scheduler` to conduct most of the training/testing computation. The major responsibilities of a `worker` are as follows.

 - Construct the subnets assigned to this worker
 - Process the task scheduled by the `scheduler`
 - Report task execution status to `scheduler`

When a `worker` receives a task from the `scheduler`, it firstly checks whether the specified input data are readily available. The input data are sent by the upstream `worker`s node. Once all inputs are received, it conducts the forward or backward propagation. The generated outputs are asynchronously sent to the downstream `worker` nodes. When all inputs of a task are processed, it notifies the `scheduler` via RPC and waits for the next task.

#### 2.4. Parameter Server
Parameter servers synchronize parameter updates from replicated models being trained concurrently, i.e. to exploit *data parallelism*.

**Data parallelism**

Data parallelism is achieved by training multiple model replicas on different partitions of the data sets. Within each model replica, the workload can be divided into multiple subnets to exploit model parallelism as described above.

Model replicas are scheduled by the `scheduler` as multiple independent training jobs. When a mini-batch is trained on a replica, the parameter updates are sent to the corresponding `parameter server` nodes.

The `parameter server` is modeled as a distributed key-value store. Parameters are grouped by the weight matrices they belong to. Each weight matrix has a unique id and is distributed to  different `parameter server` nodes using consistent hashing [5].

The size of parameters could be very large in a deep neural network. To minimize the network traffic, we differentiate the weight update protocol according to the type of the weight matrix [17]. For small weight matrices (e.g. convolutional layers), the accumulated weight updates over a mini-batch are sent to and applied on the `parameter server`. For large weight matrices (e.g. fully-connected layers), rather than sending the weight updates, the activation and delta vectors are sent over the network. Then the `parameter server` perform the matrix multiplication locally to generate the weight updates.

**Asynchronous communication**

To overlap the network communication overhead, training on a mini-batch is conducted on an outdated version of parameters. There are two sets of parameters on any given `worker` at a time: one set used for training, the other set used for synchronization. These two sets are handled as a ping-pong buffer. When a mini-batch is being trained on the first set of parameters, the `worker` fetches the latest version of parameters from the `parameter server` and saves them to the second set. Once a mini-batch finishes, the training on the next mini-batch begins on the set of parameters just fetched in the background. A background thread is responsible for sending weight updates to the `parameter server` and then fetching the new version of parameters. 

### 3. Fault Tolerance
Fault tolerance is indispensable for large-scale training in a distributed system. For the four components described in Section 2, we only need to make `scheduler`, `worker` and `parameter server` fault-tolerant. The `client` can crash or be killed after the job submission, which will not affect the job execution on the cluster. Among the three components to be fault-tolerant, only the `scheduler` needs strong consistency, while `worker`s and `parameter server`s can afford losing a few updates without affecting the final convergence due to statistical robustness. This characteristic facilitates the cost of fault tolerance.

#### 3.1 Scheduler
The `scheduler` keeps track of the job execution and cluster resources. Thus, it is very important to make its states robust in case of failure. To achieve this goal, we adopt the Paxos model [6] which replicates states in a quorum of *k* servers. A new change is committed when a majority of servers agree on it. On a server crash, its state can be reconstructed by the rest of the quorum.

Besides fault tolerance, using multiple servers for `scheduler` also has performance benefit. Different servers can be in charge of different jobs, which increases the responsiveness of the scheduling.

#### 3.2 Worker
Each `worker` snapshots its portion of the parameters to local disk periodically in the background and sends heartbeats to the `scheduler` regularly. When a heartbeat timeout occurs, the `scheduler` marks the corresponding node as offline and tries to restart the `worker` process via ssh. If the node is inaccessible or the restart fails for a given number of times, the `scheduler` reassigns the subnet of the crashed node to another node.

At a restart, the `worker` checks whether it has a local snapshot of the parameters to be populated and whether the local snapshot is up-to-date against the global version on the `parameter server`. If true, it will restore the parameters locally. Otherwise, it pulls the parameters from the `parameter server`. Because inputs and outputs are not persistent, the lost inputs need to be regenerated by the upstream layers.

#### 3.3 Parameter server
The fault tolerance of `parameter server` is achieved by *chain replication* [7]. Each set of parameters have one master server and are replicated to *k - 1* slave servers in pipeline. The replication is done asynchronously, i.e. the worker nodes do not need to wait till the finish of replication to begin training on the next mini-batch. There is a chance that the parameter server may crash before the replication is done. In this case, the updates to that parameter server are lost. Though this kind of lost is unacceptable in a general distributed file system, it is ok for parameter servers, thanks to the statistical robustness of the model. 

### 4. Implementation
The system is implemented in C++11 and CUDA. This section describes some choices we made in the implementation.

#### 4.1 Communication
The RPC channel is built on top of Google's Protocol Buffers [8]. Protocol Buffers offer a nice serialization/deserialization mechanism as well as an elegant format for writing config files. However, unlike Thrift [9], Protocol Buffers do not have built-in RPC. We can use 3rd party libraries for this functionality. In comparison, Thrift has built-in RPC support but we didn't find a nice format for config files as Protocol Buffers.

The majority of data transmission is based on MPI. MPI provides synchronous and asynchronous interfaces, which is necessary to overlap communication with computation. We also surveyed other network libraries like asio [10]. In comparison, MPI exposes more optimization opportunities like unbuffered send/receive.

Another reason behind the choice of MPI is its good integration with CUDA. CUDA-aware MPI [11] is available in major MPI implementations. Furthermore, Remote Direct Memory Access (RDMA) [12] can be leveraged to reduce the unnecessary memory copy. RDMA requires the underlying network being Infiniband, though some versions of ethernet also support it (e.g. Converged Ethernet [13]).

To relieve the pressure on network, the parameters are compressed by Snappy [14] before sending onto the network. Snappy can achieve a compression speed at 250MB/s and a decompression speed at 500MB/s on a Core i7.

#### 4.2 Graph partition
Optimal graph partitioning is a NP-hard problem. Solutions to this problem are typically derived under heuristics and approximation. There exist a few open-source implementations. The one we chose is METIS﻿ [15], which supports partitioning graphs with weighted vertex and edges and fits our requirement.

#### 4.3 Fault tolerance
Replication of the `scheduler` is taken cared by Zookeeper [16], which adopts a modified Paxos protocol for fault tolerance. Zookeeper presents a filesystem-like interface and can handle tens of thousands transactions per second, which is sufficient for our workloads.

### 5. Implementation Plan
The implementation can be decomposed into the following steps.

 1. Implement a single-machine version supporting both CPU and GPU.
 2. Implement a non-fault-tolerant distributed version based on static partitioning and no data-parallelism.
 3. Add dynamic graph partitioning.
 4. Add parameter servers and support data parallelism.
 5. Make it fault tolerant.

### References

 1. Alex Krizhevsky, One weird trick for parallelizing convolutional neural networks, arXiv:1404.5997, 2014
 2. http://en.wikipedia.org/wiki/Graph_partition
 3. Hailong Yang, et. al. MapReduce Workload Modeling with Statistical Approach, J Grid Computing 2012
 4. Yanpei Chen, et. al. A Methodology for Understanding MapReducePerformance Under Diverse Workloads, UCB Tech Report, 2010
 5. I. Stoica, et. al. Chord: A scalable peer-to-peer lookup service for internet applications. ACM SIGCOMM, 2001
 6. Lamport, Leslie (2001). Paxos Made Simple ACM SIGACT News (Distributed Computing Column) 32, 4
 7. Robbert van Renesse, et. al. Chain Replication for Supporting High Throughput and Availability., OSDI 2004
 8. Protocol Buffer, http://code.google.com/apis/protocolbuffers/
 9. Apache Thrift, https://thrift.apache.org/
 10. asio C++ library, http://think-async.com/
 11. CUDA-aware MPI, http://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/
 12. Remote Direct Memory Access, http://en.wikipedia.org/wiki/Remote_direct_memory_access
 13. RDMA over Converged Ethernet, http://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet
 14. Google Snappy, https://code.google.com/p/snappy/
 15. METIS, http://glaros.dtc.umn.edu/gkhome/views/metis
 16. Apache Zookeeper, http://zookeeper.apache.org/
 17. Trishul Chilimbi, Yutaka Suzue, Johnson Apacible, and Karthik Kalyanaraman, Project Adam: Building an Efficient and Scalable Deep Learning Training System, OSDI 2014

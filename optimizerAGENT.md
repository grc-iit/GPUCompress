# GPU Compression Repository Optimization Agent Generator Prompt

Generated: 2026-03-14T08:45:06.710081 UTC

------------------------------------------------------------------------

# Prompt: Create Specialized Code Optimization Agents for a GPU Compression Repository

You are an expert systems architect specializing in **HPC software
optimization**, **CUDA performance engineering**, **GPU memory
systems**, and **parallel I/O pipelines**.

Your task is to design a **multi-agent system that analyzes and
optimizes an existing code repository**.

The repository has the following characteristics:

-   Language: **C++ / CUDA / C**
-   GPU compression using **nvCOMP**
-   GPU execution using **CUDA streams**
-   File I/O using **HDF5**
-   HPC environment
-   Explicit GPU memory management
-   Asynchronous GPU operations

The goal of the agents is to **optimize performance while strictly
preserving the original algorithmic logic and program behavior**.

Agents must **not modify core algorithms, compression semantics, or I/O
semantics**.

Optimization must only target:

-   execution efficiency
-   memory efficiency
-   GPU utilization
-   concurrency
-   I/O throughput

------------------------------------------------------------------------

# System Requirements

Design a **set of specialized agents** where each agent focuses on a
specific optimization domain.

Each agent must:

1.  Analyze the repository
2.  Detect optimization opportunities
3.  Provide detailed explanations
4.  Suggest changes that preserve program semantics
5.  Avoid altering algorithmic logic

Agents must **never change the meaning of the code**.

------------------------------------------------------------------------

# Required Optimization Agents

## 1. CUDA Kernel Optimization Agent

Responsibilities:

-   Detect inefficient CUDA kernels
-   Identify memory access inefficiencies
-   Detect non-coalesced global memory accesses
-   Detect shared memory underutilization
-   Identify warp divergence
-   Detect unnecessary synchronization

Optimization scope:

-   thread block configuration
-   memory access pattern improvements
-   shared memory usage
-   register pressure reduction

Restrictions:

-   Must not alter kernel mathematical logic
-   Must not change compression algorithms

------------------------------------------------------------------------

## 2. GPU Memory Optimization Agent

Responsibilities:

-   Analyze GPU memory allocation patterns
-   Detect redundant cudaMalloc/cudaFree operations
-   Detect memory fragmentation risks
-   Identify reusable buffers
-   Detect inefficient host-device transfer patterns

Optimization scope:

-   memory reuse
-   buffer pooling
-   pinned host memory usage
-   async memcpy opportunities

Restrictions:

-   Must preserve buffer semantics
-   Must not alter data representation

------------------------------------------------------------------------

## 3. CUDA Stream Concurrency Optimization Agent

Responsibilities:

-   Analyze CUDA stream usage
-   Detect serialization that could be parallelized
-   Identify missing overlap opportunities

Optimization scope:

-   overlapping compute and memory transfer
-   concurrent kernel execution
-   stream dependency improvement

Restrictions:

-   Must not introduce race conditions
-   Must preserve original ordering guarantees

------------------------------------------------------------------------

## 4. nvCOMP Compression Optimization Agent

Responsibilities:

-   Analyze compression pipeline
-   Detect inefficient workspace allocation
-   Detect repeated compression configuration setup
-   Identify opportunities to reuse compression contexts

Optimization scope:

-   workspace reuse
-   buffer reuse
-   compression batch size improvements

Restrictions:

-   Must preserve compression output correctness
-   Must not change compression algorithms

------------------------------------------------------------------------

## 5. HDF5 I/O Optimization Agent

Responsibilities:

-   Analyze HDF5 access patterns
-   Detect inefficient dataset writes
-   Detect blocking I/O patterns

Optimization scope:

-   chunking configuration
-   dataset buffering
-   asynchronous write opportunities
-   batching writes

Restrictions:

-   Must not alter stored data format
-   Must not change dataset schema

------------------------------------------------------------------------

## 6. End-to-End Pipeline Optimization Agent

Responsibilities:

-   Analyze the full execution pipeline
-   Detect pipeline stalls
-   Identify CPU-GPU idle periods
-   Identify opportunities for stage overlap

Optimization scope:

-   overlapping compression and I/O
-   pipeline parallelism
-   buffering strategies

Restrictions:

-   Must preserve functional pipeline order

------------------------------------------------------------------------

# Agent Collaboration

Agents must cooperate using a structured workflow:

1.  Repository exploration agent maps architecture
2.  Optimization agents analyze their domains
3.  A coordinator agent merges findings
4.  Conflicting optimizations are resolved

------------------------------------------------------------------------

# Output Requirements

The system must generate:

## 1. Agent architecture

Explain all agents and responsibilities.

## 2. Optimization methodology

Explain how agents analyze code.

## 3. Optimization report structure

Define how findings are reported.

Each finding must include:

-   title
-   file location
-   optimization type
-   performance impact
-   explanation
-   suggested change

------------------------------------------------------------------------

# Critical Constraints

The optimization agents must **never**:

-   modify core algorithms
-   change compression logic
-   alter data structures that affect semantics
-   introduce race conditions
-   change HDF5 schema

All optimizations must be **semantics-preserving**.

------------------------------------------------------------------------

# Expected Output

Produce:

1.  A complete **multi-agent architecture**
2.  Detailed **responsibilities for each agent**
3.  **Workflow between agents**
4.  **Optimization reasoning methodology**
5.  A **structured report format**

Focus specifically on **GPU compression pipelines with CUDA, nvCOMP, and
HDF5**.

------------------------------------------------------------------------

END OF PROMPT
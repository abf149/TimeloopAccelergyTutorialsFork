Apply Timeloop+Accelergy to model dense GEMM on MatRaptor (Srivastava, et. al.)
==========

[MatRaptor (Srivastava, et. al.) paper](https://www.csl.cornell.edu/~albonesi/research/papers/micro20-2.pdf)

To try this model run:

```
./matraptor-model.sh
```

MatRaptor is an spGEMM accelerator that exploits the Gustavson/row-wise product dataflow. Here Timeloop+Accelergy was used to model MatRaptor performing dense GEMM.

## Summary: compare Timeloop+Accelergy to published simulation results in MatRaptor (Srivastava, et. al.)

The table below summarizes the results of modeling MatRaptor in Timeloop+Accelergy. The total area & energy of each component type is compared between the Timeloop+Accelergy model and the published data in MatRaptor (Srivastava, et. al.):

|  Component | Individual component area<br>(Accelergy ART)  | # Component instances  | Area, all component instances  | Energy, all component instances<br>(Timeloop stats)*  | Published area, all instances<br>From MatRaptor<br>(Srivastava, et. al.) | Published energy, all instances<br>From MatRaptor<br>(Srivastava, et. al.) |
|---|---|---|---|---|---|---|
|  MAC | 1,239.5 um^2  | 8  | 0.00992 mm^2 (0.4%)  | 1.13 nJ/GEMM  | 0.080 mm^2 (4%) | 43.08 mW (3%)  |
|  DRAM | -  | -  | -  | -  | - |  - |
|  4KB merge queues<br>(set of 10) | 277,669.0 um^2  | 8  | 2.22 mm^2 (98%)  | 0 nJ  | 1.901 mm^2 (84%) |  1007.49 mW (75%) |
|  spAL | 2,319.32 um^2  | 8  | 0.0186 mm^2  (0.8%) | 0.39654 nJ/GEMM  | 0.129 mm^2 (6%) | 144.15 mW (11%)  |
|  spBL | 2,319.32 um^2  | 8  | 0.0186 mm^2  (0.8%)| 0.79802 nJ/GEMM  | 0.129 mm^2 (6%) | 144.15 mW (11%)  |
|  - | -  | -  | -  | -  | -  | -  |
|  Total | -  | -  | 2.27 mm^2   | 70.2 nJ/GEMM   | 2.257 mm^2 (100%)  | 1344.95 mW (100%)  |

*TODO: address issues in `timeloop-model` configuration that affect the energy estimates

### Design parameters plugged into Accelergy:

| Parameter | Value
|--|--|
| Clock frequency | 1GHz |
| Process node | 45nm* |

*TODO: 28nm

### MatRaptor (Srivastava, et. al.) design parameters:

| Parameter | Value
|--|--|
| Clock frequency | 1GHz |
| Process node | 28nm |

### Modeling workflow used in MatRaptor (Srivastava, et. al.)

| Parameter | Value
|--|--|
| Architectural  design | PEs & crossbar implemented in pyMTL |
| HDL | pyMTL export to Verilog RTL |
| Cycle-accurate performance simulation | RTL simulation, gem5 model |
| Synthesis | Synopsys Design Compiler |
| P&R | Cadence Innovus => TSMC 28nm |
| Power & area modeling | Output of Innovus (?);<br>CACTI 7.0 for spAL/spBL/queues + 10% control logic overhead for spAL/spBL;<br>HBM energy numbers from public docs |


## Dense GEMM problem formulation

```
problem:
  instance:
    M: 8
    K: 8
    N: 8
  shape:
    data-spaces:
    - name: A
      projection:
      - - - M
      - - - K
    - name: B
      projection:
      - - - K
      - - - N
    - name: C
      projection:
      - - - M
      - - - N
      read-write: true
    dimensions:
    - M
    - K
    - N
    name: GEMM
```

## MatRaptor architecture

```
architecture:
  # ============================================================
  # Architecture Description
  # ============================================================
  version: 0.3
  subtree:
    - name: system
      local:
        - name: DRAM
          class: DRAM
          attributes:
            type: HBM2
            width: 64
            block-size: 4
            word-bits: 16
      subtree:
        - name: matraptor
          attributes:
            technology: 45nm
          local:
            - name: spAL[0..7]
              class: smartbuffer_SRAM
              attributes:
                memory_depth: 64
                memory_width: 16
                n_banks: 1
                block-size: 1
                word-bits: 16
                read_bandwidth: 16
                write_bandwidth: 16
                meshX: 8                
            - name: spBL[0..7]
              class: smartbuffer_SRAM
              attributes:
                memory_depth: 64
                memory_width: 16
                n_banks: 1
                block-size: 1
                word-bits: 16
                read_bandwidth: 16
                write_bandwidth: 16
                meshX: 8              
          subtree:
          - name: PE[0..7]
            local:
              # Model 10x 32-bit-wide merge queues as 1x 320-bit-wide merge queue
              - name: merge_queue
                class: smartbuffer_SRAM
                attributes:
                  memory_depth: 1024
                  memory_width: 320
                  block-size: 1
                  word-bits: 16
                  meshX: 8
                  read_bandwidth: 16
                  write_bandwidth: 16
              - name: mac
                class: intmac
                attributes:
                  datawidth: 16
                  meshX: 8
```

## MatRaptor map

### Simplified description:

The MatRaptor control logic assigns each row `m` of matrix `A` to a PE, in "round-robin" fashion (`spatial-for m`). Examining the behavior of a single PE - the PE walks along row `m` of `A` (increasing the `k` coordinate at each step; `for k`); for each non-zero element `a_mk` in row `m` of `A`, the PE vector-reads the entire compressed row `k` of `B`. 

The PE then proceeds to walk along the `n` coordinate of row `k` (`for n`) and multiply (but *not* accumulate) every non-zero element `b_kn` that it finds by `a_mk` yielding a product `c^k_mn`. The product of `a_mk` with all `b_kn` in row `k` of `B` results in up to `n` products `c^k_mn`, and the PE immediately pushes these products onto local "merge queues" to be accumulated later. 

The PE then loads the next `a_mk` from row `m` of `A` (i.e. increasing the `k` coordinate) and repeats, pushing additional `c^k_mn` products onto the merge queues. Accumulation is delayed until the entire row `m` of `A` has been multiplied by `B`; therefore the merge queues require considerable space to hold many `c^k_mn` products. 

When the PE finishes multiplying row `m` of `A` by `B`, at this point the "merge" step starts. The PE pops all the `c^k_mn` off the merge queues and accumulates them, yielding up to `n` full sums `c_mn=sum(c^k_mn)`.

### High-level:

```
for k
  for n
    spatial-for m
      c^k_mn = a_mk * b_kn
      push(c^k_mn)
merge()
stream-to-DRAM()
```


### Detailed:

```
spatial-for m # each PE picks a row of matrix A. Representing this as an outer loop for convenience
  a_m = vector-DRAM-read(A,m)
  for k # PE walks along the k-coordinate of matrix A row m
    a_mk = a_m[k]
    b_k = vector-DRAM-read(B,k)
    for n # PE walks along the n-coordinate of matrix B row k
      b_kn = b_k[n]
      c^k_mn = a_mk * b_kn # NO accumulation
      push(c^k_mn) # push c^k_mn onto one of ten PE-local merge queues
  merge() # for each spatial split along m, pop all c^k_mn products from PE-local queues and merge into full-sums: c_mn = sum{c^k_mn}
  stream-to-DRAM() # stream c_mn back to DRAM
```

## Detailed modeling result


### Area breakdown (ART)

```
ART:
  version: 0.3
  tables:
  - name: system.matraptor.PE[0..7].mac
    area: 1239.5
  - name: system.DRAM
    area: 0
  - name: system.matraptor.PE[0..7].merge_queue
    area: 277669.0
  - name: system.matraptor.spAL[0..7]
    area: 2319.32
  - name: system.matraptor.spBL[0..7]
    area: 2319.32
```

### Power breakdown (ERT)

```
ERT:
  version: 0.3
  tables:
  - name: system.matraptor.PE[0..7].mac
    actions:
    - name: mac_random
      arguments: null
      energy: 2.20035
    - name: mac_reused
      arguments: null
      energy: 1.87673
    - name: mac_gated
      arguments: null
      energy: 0.10285
    - name: idle
      arguments: null
      energy: 0.06595
  - name: system.DRAM
    actions:
    - name: read
      arguments: null
      energy: 249.6
    - name: write
      arguments: null
      energy: 249.6
    - name: idle
      arguments: null
      energy: 0
  - name: system.matraptor.PE[0..7].merge_queue
    actions:
    - name: write
      arguments:
        address_delta: 0
        data_delta: 0
      energy: 0.05093
    - name: write
      arguments:
        address_delta: 0
        data_delta: 1
      energy: 62.50393
    - name: write
      arguments:
        address_delta: 1
        data_delta: 0
      energy: 26.8165
    - name: write
      arguments:
        address_delta: 1
        data_delta: 1
      energy: 89.2695
    - name: read
      arguments:
        address_delta: 0
        data_delta: 0
      energy: 0.05093
    - name: read
      arguments:
        address_delta: 0
        data_delta: 1
      energy: 34.82271
    - name: read
      arguments:
        address_delta: 1
        data_delta: 0
      energy: 14.95312
    - name: read
      arguments:
        address_delta: 1
        data_delta: 1
      energy: 49.7249
    - name: idle
      arguments: null
      energy: 0.01403
  - name: system.matraptor.spAL[0..7]
    actions:
    - name: write
      arguments:
        address_delta: 0
        data_delta: 0
      energy: 0.04881
    - name: write
      arguments:
        address_delta: 0
        data_delta: 1
      energy: 0.64197
    - name: write
      arguments:
        address_delta: 1
        data_delta: 0
      energy: 0.30302
    - name: write
      arguments:
        address_delta: 1
        data_delta: 1
      energy: 0.89618
    - name: read
      arguments:
        address_delta: 0
        data_delta: 0
      energy: 0.04881
    - name: read
      arguments:
        address_delta: 0
        data_delta: 1
      energy: 0.47837
    - name: read
      arguments:
        address_delta: 1
        data_delta: 0
      energy: 0.2329
    - name: read
      arguments:
        address_delta: 1
        data_delta: 1
      energy: 0.66246
    - name: idle
      arguments: null
      energy: 0.01191
  - name: system.matraptor.spBL[0..7]
    actions:
    - name: write
      arguments:
        address_delta: 0
        data_delta: 0
      energy: 0.04881
    - name: write
      arguments:
        address_delta: 0
        data_delta: 1
      energy: 0.64197
    - name: write
      arguments:
        address_delta: 1
        data_delta: 0
      energy: 0.30302
    - name: write
      arguments:
        address_delta: 1
        data_delta: 1
      energy: 0.89618
    - name: read
      arguments:
        address_delta: 0
        data_delta: 0
      energy: 0.04881
    - name: read
      arguments:
        address_delta: 0
        data_delta: 1
      energy: 0.47837
    - name: read
      arguments:
        address_delta: 1
        data_delta: 0
      energy: 0.2329
    - name: read
      arguments:
        address_delta: 1
        data_delta: 1
      energy: 0.66246
    - name: idle
      arguments: null
      energy: 0.01191

```

### Timeloop statistics

```
Buffer and Arithmetic Levels
----------------------------
Level 0
-------
=== mac ===

    SPECS
    -----
    Word bits             : 16
    Instances             : 8 (8*1)
    Compute energy        : 2.20 pJ

    STATS
    -----
    Utilized instances           : 8
    Cycles                       : 64
    Algorithmic Computes (total) : 512
    Actual Computes (total)      : 512
    Gated Computes (total)       : 0
    Skipped Computes (total)     : 0
    Energy (total)               : 1126.58 pJ
    Area (total)                 : 9916.00 um^2

Level 1
-------
=== merge_queue ===

    SPECS
    -----
        Technology                   : SRAM
        Data storage size            : 1024
        Data word bits               : 16
        Data block size              : 1
        Metadata storage size        : 0
        Metadata word bits           : 0
        Metadata block size          : 1
        Cluster size                 : 20
        Instances                    : 8 (8*1)
        Read bandwidth               : 16.00
        Write bandwidth              : 16.00
        Multiple buffering           : 1.00
        Effective data storage size  : 1024
        Min utilization              : 0.00
        Vector read energy           : 49.72 pJ
        Vector write energy          : 89.27 pJ
        Vector metadata read energy  : 0.01 pJ
        Vector metadata write energy : 0.01 pJ
        (De)compression energy       : 0.00 pJ
        Area                         : 13883.45 um^2

    MAPPING
    -------
    Loop nest:

    STATS
    -----
    Cycles               : 64
    Bandwidth throttling : 1.00
    C:
        Partition size                                              : 8
        Tile density distribution                                   : fixed-structured
        Data tile shape                                             : 1
        Max utilized data storage capacity                          : 1
        Metadata format                                             : none
        Max utilized metadata storage capacity                      : 0
        Utilized instances (max)                                    : 8
        Utilized clusters (max)                                     : 0
        Algorithmic scalar reads (per-instance)                     : 56
        Actual scalar reads (per-instance)                          : 56
        Gated scalar reads (per-instance)                           : 0
        Skipped scalar reads (per-instance)                         : 0
        Algorithmic scalar fills (per-instance)                     : 64
        Actual scalar fills (per-instance)                          : 64
        Gated scalar fills (per-instance)                           : 0
        Skipped scalar fills (per-instance)                         : 0
        Algorithmic scalar updates (per-instance)                   : 64
        Actual scalar updates (per-instance)                        : 64
        Gated scalar updates (per-instance)                         : 0
        Skipped scalar updates (per-instance)                       : 0
        Actual scalar metadata reads (per-instance)                 : 0
        Gated scalar metadata reads (per-instance)                  : 0
        Skipped scalar metadata reads (per-instance)                : 0
        Actual scalar metadata fills (per-instance)                 : 0
        Gated scalar metadata fills (per-instance)                  : 0
        Skipped metadata fills (per-instance)                       : 0
        Actual scalar metadata updates (per-instance)               : 0
        Gated scalar metadata gated updates (per-instance)          : 0
        Skipped scalar metadata updates (per-instance)              : 0
        Scalar decompression counts (per-cluster)                   : 0
        Scalar compression counts (per-cluster)                     : 0
        Temporal reductions (per-instance)                          : 56
        Address generations (per-cluster)                           : 128
        Energy (per-scalar-access)                                  : 0.00 pJ
        Energy (per-instance)                                       : 0.00 pJ
        Energy (total)                                              : 0.00 pJ
        Temporal Reduction Energy (per-instance)                    : 0.00 pJ
        Temporal Reduction Energy (total)                           : 0.00 pJ
        Address Generation Energy (per-cluster)                     : 0.00 pJ
        Address Generation Energy (total)                           : 0.00 pJ
        Read Bandwidth (per-instance)                               : 0.88 words/cycle
        Read Bandwidth (total)                                      : 7.00 words/cycle
        Write Bandwidth (per-instance)                              : 2.00 words/cycle
        Write Bandwidth (total)                                     : 16.00 words/cycle

Level 2
-------
=== spBL ===

    SPECS
    -----
        Technology                   : SRAM
        Data storage size            : 64
        Data word bits               : 16
        Data block size              : 1
        Metadata storage size        : 0
        Metadata word bits           : 0
        Metadata block size          : 1
        Cluster size                 : 1
        Instances                    : 8 (8*1)
        Read bandwidth               : 16.00
        Write bandwidth              : 16.00
        Multiple buffering           : 1.00
        Effective data storage size  : 64
        Min utilization              : 0.00
        Vector read energy           : 0.66 pJ
        Vector write energy          : 0.90 pJ
        Vector metadata read energy  : 0.01 pJ
        Vector metadata write energy : 0.01 pJ
        (De)compression energy       : 0.00 pJ
        Area                         : 2319.32 um^2

    MAPPING
    -------
    Loop nest:
      for N in [0:8)

    STATS
    -----
    Cycles               : 64
    Bandwidth throttling : 1.00
    B:
        Partition size                                              : 64
        Tile density distribution                                   : fixed-structured
        Data tile shape                                             : 8
        Max utilized data storage capacity                          : 8
        Metadata format                                             : none
        Max utilized metadata storage capacity                      : 0
        Utilized instances (max)                                    : 8
        Utilized clusters (max)                                     : 8
        Algorithmic scalar reads (per-instance)                     : 64
        Actual scalar reads (per-instance)                          : 64
        Gated scalar reads (per-instance)                           : 0
        Skipped scalar reads (per-instance)                         : 0
        Algorithmic scalar fills (per-instance)                     : 64
        Actual scalar fills (per-instance)                          : 64
        Gated scalar fills (per-instance)                           : 0
        Skipped scalar fills (per-instance)                         : 0
        Algorithmic scalar updates (per-instance)                   : 0
        Actual scalar updates (per-instance)                        : 0
        Gated scalar updates (per-instance)                         : 0
        Skipped scalar updates (per-instance)                       : 0
        Actual scalar metadata reads (per-instance)                 : 0
        Gated scalar metadata reads (per-instance)                  : 0
        Skipped scalar metadata reads (per-instance)                : 0
        Actual scalar metadata fills (per-instance)                 : 0
        Gated scalar metadata fills (per-instance)                  : 0
        Skipped metadata fills (per-instance)                       : 0
        Actual scalar metadata updates (per-instance)               : 0
        Gated scalar metadata gated updates (per-instance)          : 0
        Skipped scalar metadata updates (per-instance)              : 0
        Scalar decompression counts (per-cluster)                   : 0
        Scalar compression counts (per-cluster)                     : 0
        Temporal reductions (per-instance)                          : 0
        Address generations (per-cluster)                           : 128
        Energy (per-scalar-access)                                  : 0.78 pJ
        Energy (per-instance)                                       : 99.75 pJ
        Energy (total)                                              : 798.02 pJ
        Temporal Reduction Energy (per-instance)                    : 0.00 pJ
        Temporal Reduction Energy (total)                           : 0.00 pJ
        Address Generation Energy (per-cluster)                     : 0.00 pJ
        Address Generation Energy (total)                           : 0.00 pJ
        Read Bandwidth (per-instance)                               : 1.00 words/cycle
        Read Bandwidth (total)                                      : 8.00 words/cycle
        Write Bandwidth (per-instance)                              : 1.00 words/cycle
        Write Bandwidth (total)                                     : 8.00 words/cycle

Level 3
-------
=== spAL ===

    SPECS
    -----
        Technology                   : SRAM
        Data storage size            : 64
        Data word bits               : 16
        Data block size              : 1
        Metadata storage size        : 0
        Metadata word bits           : 0
        Metadata block size          : 1
        Cluster size                 : 1
        Instances                    : 8 (8*1)
        Read bandwidth               : 16.00
        Write bandwidth              : 16.00
        Multiple buffering           : 1.00
        Effective data storage size  : 64
        Min utilization              : 0.00
        Vector read energy           : 0.66 pJ
        Vector write energy          : 0.90 pJ
        Vector metadata read energy  : 0.01 pJ
        Vector metadata write energy : 0.01 pJ
        (De)compression energy       : 0.00 pJ
        Area                         : 2319.32 um^2

    MAPPING
    -------
    Loop nest:

    STATS
    -----
    Cycles               : 64
    Bandwidth throttling : 1.00
    A:
        Partition size                                              : 8
        Tile density distribution                                   : fixed-structured
        Data tile shape                                             : 1
        Max utilized data storage capacity                          : 1
        Metadata format                                             : none
        Max utilized metadata storage capacity                      : 0
        Utilized instances (max)                                    : 8
        Utilized clusters (max)                                     : 8
        Algorithmic scalar reads (per-instance)                     : 64
        Actual scalar reads (per-instance)                          : 64
        Gated scalar reads (per-instance)                           : 0
        Skipped scalar reads (per-instance)                         : 0
        Algorithmic scalar fills (per-instance)                     : 8
        Actual scalar fills (per-instance)                          : 8
        Gated scalar fills (per-instance)                           : 0
        Skipped scalar fills (per-instance)                         : 0
        Algorithmic scalar updates (per-instance)                   : 0
        Actual scalar updates (per-instance)                        : 0
        Gated scalar updates (per-instance)                         : 0
        Skipped scalar updates (per-instance)                       : 0
        Actual scalar metadata reads (per-instance)                 : 0
        Gated scalar metadata reads (per-instance)                  : 0
        Skipped scalar metadata reads (per-instance)                : 0
        Actual scalar metadata fills (per-instance)                 : 0
        Gated scalar metadata fills (per-instance)                  : 0
        Skipped metadata fills (per-instance)                       : 0
        Actual scalar metadata updates (per-instance)               : 0
        Gated scalar metadata gated updates (per-instance)          : 0
        Skipped scalar metadata updates (per-instance)              : 0
        Scalar decompression counts (per-cluster)                   : 0
        Scalar compression counts (per-cluster)                     : 0
        Temporal reductions (per-instance)                          : 0
        Address generations (per-cluster)                           : 72
        Energy (per-scalar-access)                                  : 0.69 pJ
        Energy (per-instance)                                       : 49.57 pJ
        Energy (total)                                              : 396.54 pJ
        Temporal Reduction Energy (per-instance)                    : 0.00 pJ
        Temporal Reduction Energy (total)                           : 0.00 pJ
        Address Generation Energy (per-cluster)                     : 0.00 pJ
        Address Generation Energy (total)                           : 0.00 pJ
        Read Bandwidth (per-instance)                               : 1.00 words/cycle
        Read Bandwidth (total)                                      : 8.00 words/cycle
        Write Bandwidth (per-instance)                              : 0.12 words/cycle
        Write Bandwidth (total)                                     : 1.00 words/cycle

Level 4
-------
=== DRAM ===

    SPECS
    -----
        Technology                   : DRAM
        Data storage size            : -
        Data word bits               : 16
        Data block size              : 4
        Metadata storage size        : 0
        Metadata word bits           : 0
        Metadata block size          : 1
        Cluster size                 : 1
        Instances                    : 1 (1*1)
        Read bandwidth               : -
        Write bandwidth              : -
        Multiple buffering           : 1.00
        Effective data storage size  : -
        Min utilization              : 0.00
        Vector read energy           : 249.60 pJ
        Vector write energy          : 249.60 pJ
        Vector metadata read energy  : 0.00 pJ
        Vector metadata write energy : 0.00 pJ
        (De)compression energy       : 0.00 pJ
        Area                         : 0.00 um^2

    MAPPING
    -------
    Loop nest:
      for K in [0:8)
        for M in [0:8) (Spatial-X)

    STATS
    -----
    Cycles               : 64
    Bandwidth throttling : 1.00
    A:
        Partition size                                              : 64
        Tile density distribution                                   : fixed-structured
        Data tile shape                                             : 64
        Max utilized data storage capacity                          : 64
        Metadata format                                             : none
        Max utilized metadata storage capacity                      : 0
        Utilized instances (max)                                    : 1
        Utilized clusters (max)                                     : 1
        Algorithmic scalar reads (per-instance)                     : 64
        Actual scalar reads (per-instance)                          : 64
        Gated scalar reads (per-instance)                           : 0
        Skipped scalar reads (per-instance)                         : 0
        Algorithmic scalar fills (per-instance)                     : 0
        Actual scalar fills (per-instance)                          : 0
        Gated scalar fills (per-instance)                           : 0
        Skipped scalar fills (per-instance)                         : 0
        Algorithmic scalar updates (per-instance)                   : 0
        Actual scalar updates (per-instance)                        : 0
        Gated scalar updates (per-instance)                         : 0
        Skipped scalar updates (per-instance)                       : 0
        Actual scalar metadata reads (per-instance)                 : 0
        Gated scalar metadata reads (per-instance)                  : 0
        Skipped scalar metadata reads (per-instance)                : 0
        Actual scalar metadata fills (per-instance)                 : 0
        Gated scalar metadata fills (per-instance)                  : 0
        Skipped metadata fills (per-instance)                       : 0
        Actual scalar metadata updates (per-instance)               : 0
        Gated scalar metadata gated updates (per-instance)          : 0
        Skipped scalar metadata updates (per-instance)              : 0
        Scalar decompression counts (per-cluster)                   : 0
        Scalar compression counts (per-cluster)                     : 0
        Temporal reductions (per-instance)                          : 0
        Address generations (per-cluster)                           : 64
        Energy (per-scalar-access)                                  : 62.40 pJ
        Energy (per-instance)                                       : 3993.60 pJ
        Energy (total)                                              : 3993.60 pJ
        Temporal Reduction Energy (per-instance)                    : 0.00 pJ
        Temporal Reduction Energy (total)                           : 0.00 pJ
        Address Generation Energy (per-cluster)                     : 0.00 pJ
        Address Generation Energy (total)                           : 0.00 pJ
        Read Bandwidth (per-instance)                               : 1.00 words/cycle
        Read Bandwidth (total)                                      : 1.00 words/cycle
        Write Bandwidth (per-instance)                              : 0.00 words/cycle
        Write Bandwidth (total)                                     : 0.00 words/cycle
    B:
        Partition size                                              : 64
        Tile density distribution                                   : fixed-structured
        Data tile shape                                             : 64
        Max utilized data storage capacity                          : 64
        Metadata format                                             : none
        Max utilized metadata storage capacity                      : 0
        Utilized instances (max)                                    : 1
        Utilized clusters (max)                                     : 1
        Algorithmic scalar reads (per-instance)                     : 64
        Actual scalar reads (per-instance)                          : 64
        Gated scalar reads (per-instance)                           : 0
        Skipped scalar reads (per-instance)                         : 0
        Algorithmic scalar fills (per-instance)                     : 0
        Actual scalar fills (per-instance)                          : 0
        Gated scalar fills (per-instance)                           : 0
        Skipped scalar fills (per-instance)                         : 0
        Algorithmic scalar updates (per-instance)                   : 0
        Actual scalar updates (per-instance)                        : 0
        Gated scalar updates (per-instance)                         : 0
        Skipped scalar updates (per-instance)                       : 0
        Actual scalar metadata reads (per-instance)                 : 0
        Gated scalar metadata reads (per-instance)                  : 0
        Skipped scalar metadata reads (per-instance)                : 0
        Actual scalar metadata fills (per-instance)                 : 0
        Gated scalar metadata fills (per-instance)                  : 0
        Skipped metadata fills (per-instance)                       : 0
        Actual scalar metadata updates (per-instance)               : 0
        Gated scalar metadata gated updates (per-instance)          : 0
        Skipped scalar metadata updates (per-instance)              : 0
        Scalar decompression counts (per-cluster)                   : 0
        Scalar compression counts (per-cluster)                     : 0
        Temporal reductions (per-instance)                          : 0
        Address generations (per-cluster)                           : 64
        Energy (per-scalar-access)                                  : 62.40 pJ
        Energy (per-instance)                                       : 3993.60 pJ
        Energy (total)                                              : 3993.60 pJ
        Temporal Reduction Energy (per-instance)                    : 0.00 pJ
        Temporal Reduction Energy (total)                           : 0.00 pJ
        Address Generation Energy (per-cluster)                     : 0.00 pJ
        Address Generation Energy (total)                           : 0.00 pJ
        Read Bandwidth (per-instance)                               : 1.00 words/cycle
        Read Bandwidth (total)                                      : 1.00 words/cycle
        Write Bandwidth (per-instance)                              : 0.00 words/cycle
        Write Bandwidth (total)                                     : 0.00 words/cycle
    C:
        Partition size                                              : 64
        Tile density distribution                                   : fixed-structured
        Data tile shape                                             : 64
        Max utilized data storage capacity                          : 64
        Metadata format                                             : none
        Max utilized metadata storage capacity                      : 0
        Utilized instances (max)                                    : 1
        Utilized clusters (max)                                     : 1
        Algorithmic scalar reads (per-instance)                     : 448
        Actual scalar reads (per-instance)                          : 448
        Gated scalar reads (per-instance)                           : 0
        Skipped scalar reads (per-instance)                         : 0
        Algorithmic scalar fills (per-instance)                     : 0
        Actual scalar fills (per-instance)                          : 0
        Gated scalar fills (per-instance)                           : 0
        Skipped scalar fills (per-instance)                         : 0
        Algorithmic scalar updates (per-instance)                   : 512
        Actual scalar updates (per-instance)                        : 512
        Gated scalar updates (per-instance)                         : 0
        Skipped scalar updates (per-instance)                       : 0
        Actual scalar metadata reads (per-instance)                 : 0
        Gated scalar metadata reads (per-instance)                  : 0
        Skipped scalar metadata reads (per-instance)                : 0
        Actual scalar metadata fills (per-instance)                 : 0
        Gated scalar metadata fills (per-instance)                  : 0
        Skipped metadata fills (per-instance)                       : 0
        Actual scalar metadata updates (per-instance)               : 0
        Gated scalar metadata gated updates (per-instance)          : 0
        Skipped scalar metadata updates (per-instance)              : 0
        Scalar decompression counts (per-cluster)                   : 0
        Scalar compression counts (per-cluster)                     : 0
        Temporal reductions (per-instance)                          : 448
        Address generations (per-cluster)                           : 512
        Energy (per-scalar-access)                                  : 62.40 pJ
        Energy (per-instance)                                       : 59904.00 pJ
        Energy (total)                                              : 59904.00 pJ
        Temporal Reduction Energy (per-instance)                    : 0.00 pJ
        Temporal Reduction Energy (total)                           : 0.00 pJ
        Address Generation Energy (per-cluster)                     : 0.00 pJ
        Address Generation Energy (total)                           : 0.00 pJ
        Read Bandwidth (per-instance)                               : 7.00 words/cycle
        Read Bandwidth (total)                                      : 7.00 words/cycle
        Write Bandwidth (per-instance)                              : 8.00 words/cycle
        Write Bandwidth (total)                                     : 8.00 words/cycle

Networks
--------
Network 0
---------
DRAM <==> spAL

    SPECS
    -----
        Type            : Legacy
        Legacy sub-type : 
        ConnectionType  : 3
        Word bits       : 16
        Router energy   : - pJ
        Wire energy     : - pJ/b/mm

    STATS
    -----
    A:
        Fanout                                  : 8
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 1.41
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    B:
        Fanout                                  : 8
        Fanout (distributed)                    : 0
        Multicast factor                        : 8
        Ingresses                               : 64
            @multicast 8: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 7.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    C:
        Fanout                                  : 8
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 960
            @multicast 1: 960
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 1.41
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ

Network 1
---------
merge_queue <==> mac

    SPECS
    -----
        Type            : Legacy
        Legacy sub-type : 
        ConnectionType  : 3
        Word bits       : 16
        Router energy   : - pJ
        Wire energy     : - pJ/b/mm

    STATS
    -----
    A:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    B:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    C:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ

Network 2
---------
spAL <==> spBL

    SPECS
    -----
        Type            : Legacy
        Legacy sub-type : 
        ConnectionType  : 3
        Word bits       : 16
        Router energy   : - pJ
        Wire energy     : - pJ/b/mm

    STATS
    -----
    A:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    B:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    C:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ

Network 3
---------
spBL <==> merge_queue

    SPECS
    -----
        Type            : Legacy
        Legacy sub-type : 
        ConnectionType  : 3
        Word bits       : 16
        Router energy   : - pJ
        Wire energy     : - pJ/b/mm

    STATS
    -----
    A:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    B:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ
    C:
        Fanout                                  : 1
        Fanout (distributed)                    : 0
        Multicast factor                        : 1
        Ingresses                               : 64
            @multicast 1: 64
        Link transfers                          : 0
        Spatial reductions                      : 0
        Average number of hops                  : 0.50
        Energy (per-hop)                        : 0.00 fJ
        Energy (per-instance)                   : 0.00 pJ
        Energy (total)                          : 0.00 pJ
        Link transfer energy (per-instance)     : 0.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (per-instance) : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ

Total topology energy: 70212.34 pJ
Total topology area: 158092.72 um^2
Max topology cycles: 64

Summary Stats
-------------
Utilization: 1.00
Cycles: 64
Energy: 0.07 uJ
Area: 0.16 mm^2

Algorithmic Computes = 512
pJ/Algorithmic-Compute
    mac                   = 2.20
    merge_queue           = 0.00
    spBL                  = 1.56
    spAL                  = 0.77
    DRAM                  = 132.60
    DRAM <==> spAL        = 0.00
    merge_queue <==> mac  = 0.00
    spAL <==> spBL        = 0.00
    spBL <==> merge_queue = 0.00
    Total                 = 137.13

Actual Computes = 512
pJ/Compute
    mac                   = 2.20
    merge_queue           = 0.00
    spBL                  = 1.56
    spAL                  = 0.77
    DRAM                  = 132.60
    DRAM <==> spAL        = 0.00
    merge_queue <==> mac  = 0.00
    spAL <==> spBL        = 0.00
    spBL <==> merge_queue = 0.00
    Total                 = 137.13
```
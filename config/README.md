# Configuration 

This directory includes the hardware configuration needed by the simulator. 
It includes multiple configuration files for different hardware components in the accelerator.
They are detailed as follows. 

## Simulator

There are several parameters to control the mode of simulation.
The configuration file ```simulation_config.json``` includes these parameters and they are shown as the following table.

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```display_simulation_progress```| false | bool | Whether the simulation progress will be displayed.| 
|```bypass_ld_st_global_backend```| false | bool | Whether the backend of ld.global and st.global is bypassed. |
|```bypass_lsu_remote_dcache```| false | bool | Whether the read-only dacache in the remote load-store unit is bypassed |
|```bypass_niu_dcache```| false | bool | Whether the read-only dacache in the network interface unit is bypassed |
|```bypass_router```| false | bool | Whether bypass routing of the packets in the network. |

## Hardware 
An MPU accelerator is composed of several processors. 
The configuration file ```hardware_config.json``` includes parameters at the level of the whole accelerator. 
The detailed configuration parameters are shown as the following table.

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```num_processor_x```| 4 | | The number of processors along the x-axis in a 2D layout|
|```num_processor_y```| 2 | | The number of processors along the y-axis in a 2D layout|

## Processor 
A processor inside the MPU is composed of several cores. 
The configuration file ```processor_config.json``` includes parameters at the level of the processor. 
The detailed configuration parameters are shown as the following table.

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```num_core_x```| 4 | | The number of control cores inside a processor along the x-axis|
|```num_core_y```| 4 | | The number of control cores inside a processor along the y-axis|
|```processor_shared_bus_io_width```| 128 | byte | The I/O width of the shared bus between the subcores and the processing groups per processor |

## Core
A core is the unit composed of several subcores, processing groups (PG), and processing elements (PE) to execute programs. 
The configuration file ```core_config.json``` includes parameters at the level of the core. 
The detailed configuration parameters are shown as the following table. 

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```num_pg```| 4 | | The number of processing groups (PG) inside a core|
|```num_pe```| 8 | | The number of processing elements (PE) inside a PG |
|```num_subcore```| 4 | | The number of subcores to execute thread warps |
|```max_num_warp_per_subcore```| 4 | | The maximum number of concurrent warps on each subcore |
|```data_path_unit_size```| 4 | byte | The size of the data path unit size (32b), which is used to enforce data alignment |
|```max_bar_id_per_block```| 8 | | The maximum number of barrier IDs per thread block |
|```max_simt_stack_depth```| 32 | | The maximum depth of SIMT stack for each warp |
|```icache_size```| 131072 | byte | The icache size |
|```num_port_icache```| 4 | | The number of ports of an icache |
|```icache_read_latency```| 1 | cycle | The icache read latency |
|```icache_write_latency```| 1 | cycle | The icache write latency |
|```icache_read_energy```| 0.107244 | nJ | The icache read energy |
|```icache_write_energy```| 0.230658 | nJ | The icache write energy |
|```decode_buffer_size```| 4 | | The maximum number of instructions a decode buffer can hold | 
|```instr_fetch_decode_energy```| 0.00087 | nJ | The fetch and decode energy for an instruction |
|```instr_wb_commit_energy```| 0.00030 | nJ | The writeback and commit energy for an instruction |
|```dep_table_read_energy```| 0.00002750 | nJ | scoreboad read energy |
|```dep_table_write_energy```| 0.00002750 | nJ | scoreboad write energy |
|```subcore_execute_buffer_size```| 4 | | The maximum number of instructions a subcore execute buffer can hold |
|```subcore_writeback_buffer_size```| 4 | | The maximum number of instructions a subcore writeback buffer can hold |
|```subcore_commit_buffer_size```| 4 | | The maximum number of instructions a subcore commit buffer can hold |
|```subcore_bus_receive_buffer_size```| 4 | | The maximum number of packets a subcore receive bus buffer can hold |
|```pg_execute_buffer_size```| 4 | | The maximum number of instructions a PG execute buffer can hold |
|```pg_writeback_buffer_size```| 4 | | The maximum number of instructions a PG writeback buffer can hold |
|```pg_commit_buffer_size```| 4 | | The maximum number of instructions a PG commit buffer can hold |
|```pg_bus_receive_buffer_size```| 4 | | The maximum number of packets a pg receive bus buffer can hold |
|```core_shared_bus_energy_bit```| 0.00453 | nJ | The energy per bit access for shared subcore-pg bus |
|```opc_size```| 4 | byte | operand collector size |
|```opc_read_energy```| 0.04149 | nJ | operand collector read energy |
|```opc_write_energy```| 0.04991 | nJ | operand collector write energy |
|```lsu_prt_size```| 4096 | byte | load-store unit pending request table size |
|```lsu_prt_read_energy```| 0.03967 | nJ | load-store unit pending request table read energy |
|```lsu_prt_write_energy```| 0.04090 | nJ | load-store unit pending request table write energy |
|```lsu_remote_prt_size```| 4096 | byte | remote load-store unit pending request table size |
|```lsu_remote_prt_read_energy```| 0.03967 | nJ | remote load-store unit pending request table read energy |
|```lsu_remote_prt_write_energy```| 0.04090 | nJ | remote load-store unit pending request table write energy |
|```lsu_remote_readonly_dcache_size```| 32768 | byte | read only cache size |
|```lsu_remote_readonly_dcache_latency```| 1 | cycle |  read only cache latency |
|```lsu_remote_readonly_dcache_eviction_policy```| lru | |  read only cache eviction policy |
|```lsu_extension_prt_size```| 4096 | byte | load-store unit extension pending request table size |
|```lsu_extension_prt_read_energy```| 0.03967 | nJ | load-store unit extension pending request table read energy |
|```lsu_extension_prt_write_energy```| 0.04090 | nJ | load-store unit extension pending request table write energy |
|```lsu_shared_issue_port```| 4 | | number of issue ports for shared memory request in load-store unit |
|```lsu_extensiuon_shared_issue_port```| 4 | | number of issue ports for shared memory request in load-store unit extension |
|```num_opc_lsu```| 4 | | The number of operand collector units for load-store units of each subcore |
|```num_opc_lsu_extension```| 4 | | The number of operand collector units for load-store unit extension of each PG |
|```num_opc_fb_alu```| 4 | | The number of operand collector units for far-bank ALUs of each subcore |
|```num_opc_sfu```| 1 | | The number of operand collector units for SFUs of each subcore |
|```num_opc_cfu```| 4 | | The number of operand collector units for control flow unit of each subcore |
|```num_opc_syncu```| 1 | | The number of operand collector units for synchronization unit of each subcore |
|```num_lsu```| 1 | | The number of load-store units for each subcore |
|```num_lsu_extension```| 1 | | The number of load-store unit extensions for each processing group |
|```num_fb_alu```| 4 | | The number of far-bank ALUs for each subcore |
|```num_sfu```| 1 | | The number of SFUs for each subcore |
|```num_cfu```| 1 | | The number of control flow units for each subcore |
|```num_syncu```| 1 | | The number of synchronization units for each subcore |
|```num_opc_nb_alu```| 4 | | The number of operand collector units for near-bank ALUs of each PG |
|```num_nb_alu```| 1 | | The number of near-bank ALUs for each PG |
|```num_offload_engine```| 4 | | The number of instruction offloading engines for each subcore |
|```num_issue_port_per_warp```| 2 | | The number of issue ports for each subcore per warp |
|```num_fb_wb_port```| 4 | | The number of writeback ports for each subcore |
|```num_nb_wb_port```| 4 | | The number of writeback ports for each PG |
|```num_fb_commit_port```| 4 | | The number of commit ports for each subcore |
|```num_nb_commit_port```| 4 | | The number of commit ports for each PG |
|```num_fb_reg_move_engine```| 1 | | The number of register movement engine for each subcore |
|```num_nb_reg_move_engine```| 1 | | The number of register movement engine for each PG |
|```core_clock_freq```| 1000 |  MHz | The clock frequency of a core |
|```subcore_pg_bus_clock_freq```| 2000 |  MHz | The clock frequency of the shared bus between PGs and subcores |
|```default_specialreg_loc_is_near_bank```| false | | The default location of special registers|
|```default_param_loc_is_near_bank```| false | | The default location of parameters |
|```default_imm_loc_is_near_bank```| false | | The default location of immediate values |
|```default_smem_loc_is_near_bank```| true | | The default location of shared memory |

## Register File

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```subcore_reg_file_size```| 32768 | byte | The size of register file per subcore |
|```num_subcore_reg_file_bank```| 4 | | The number of banks per register file for each subcore |
|```subcore_reg_file_bank_queue_size```| 4 | | The queue size (read request or write request) for each subcore register file bank |
|```subcore_reg_file_read_latency```| 1 |  cycle | subcore register file read latency |
|```subcore_reg_file_write_latency```| 1 |  cycle | subcore register file write latency |
|```subcore_special_reg_read_latency```| 1 |  cycle | subcore special register read latency |
|```subcore_param_reg_read_latency```| 1 |  cycle | subcore parameter register read latency |
|```subcore_reg_file_bank_read_energy```| 0.0402723 | nJ | subcore register file bank read energy |
|```subcore_reg_file_bank_write_energy```| 0.0438918 | nJ | subcore register file bank write energy |
|```pg_reg_file_bank_read_energy```| 0.0402723 | nJ | PG register file bank read energy |
|```pg_reg_file_bank_write_energy```| 0.0438918 | nJ | PG register file bank write energy |
|```num_pg_reg_file_bank```| 4 | | The number of banks per register file for each PG |
|```pg_reg_file_bank_queue_size```| 4 | | The queue size (read request or write request) for each PG register file bank |
|```pg_reg_file_read_latency```| 1 |  cycle | PG register file read latency |
|```pg_reg_file_write_latency```| 1 |  cycle | PG register file write latency |
|```reg_track_table_size```| 512 | byte | register track table size |
|```reg_track_table_read_energy```| 0.000219836 | nJ | register track table read energy |
|```reg_track_table_write_energy```| 0.00018576 | nJ | register track table write energy |

### Address mapping
| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```reg_file_addr_map```| reg_file_addr_map_1 | | address mapping scheme (both far-bank and near-bank) from register file address to device address |

reg_file_addr_map_1:<br/>
| REG_FILE_BANK_INTERNAL_ADDR | REG_FILE_BANK_INDEX | THREAD_INDEX | DATA_PATH_UNIT_ALIGNMENT |

For subcore register file:
* Minimum addressable unit is a data path unit (default: 4 bytes) which enforces data alignment. Sub-word data type (less than 4 bytes) only utilizes part of the register. For example, .b16 element only utilizes the lower 2 bytes of the 4 bytes register.
* **THREAD_INDEX | DATA_PATH_UNIT_ALIGNMENT** part will be discarded before indexing into the regsiter file. This provides a vector register file abstraction, as a single access results in ```num_threads_per_warp``` elements of size ```data_path_unit_size``` (default: 128 bytes).
* **REG_FILE_BANK_INTERNAL_ADDR | REG_FILE_BANK_INDEX** addresses a warp's private portion of the register file. Note that a warp's register file section does not have to be aligned to ```num_subcore_reg_file_bank``` * ```num_threads_per_warp``` * ```data_path_unit_size```.

Please see the following patent for register alignment:
> "Conflict-free register allocation using a multi-bank register file with input operand alignment" (US8555035B1)

## Shared Memory
| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```smem_size```| 65536 | byte | The size of shared memory per core |
|```num_smem_bank```| 32 | | The number of banks of the shared memory (this should be equal to the number of threads per warp) |
|```smem_req_queue_size```| 1 | | The number of buffer slots in the shared memory request queue |
|```smem_bank_req_queue_size```| 32 | | The number of buffer slots for each shared memory bank|
|```smem_read_latency```| 1 | cycle | shared memory read latency |
|```smem_write_latency```| 1 | cycle | shared memory write latency |
|```smem_bank_read_energy```| 0.000694566 | nJ | shared memory bank read energy |
|```smem_bank_write_energy```| 0.000753546 | nJ | shared memory bank write energy |

### Address mapping
| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```smem_addr_map```| smem_addr_map_1 | | address mapping scheme for shared memory |

smem_addr_map_1:<br/>
| BANK_INTERNAL_ADDR | BANK_INDEX | BANK_INTERFACE |

Please refer to the following page for shared memory:
> "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory"

## Execution Unit
Execution units are the main components of the backend part of the core, including arithmetic-logic-unit (ALU), special-functional-unit (SFU), and load-store-unit (LSU).

### latency configuration

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```alu_t_int_add```| 1 | cycle | integer add latency |
|```alu_t_int_sub```| 1 | cycle | integer sub latency |
|```alu_t_int_min```| 1 | cycle | integer min latency |
|```alu_t_int_max```| 1 | cycle | integer max latency |
|```alu_t_int_mul```| 1 | cycle | integer multiply latency |
|```alu_t_int_mad```| 1 | cycle | integer multiply-add latency |
|```alu_t_int_div_s```| 124 | cycle | integer divide latency (signed) |
|```alu_t_int_div_u```| 119 | cycle | integer divide latency (unsigned) |
|```alu_t_int_rem_s```| 122 | cycle | integer remainder of division latency (signed) |
|```alu_t_int_rem_u```| 114 | cycle | integer remainder of division latency (unsigned) |
|```alu_t_int_abs```| 5 | cycle | integer absolute latency |
|```alu_t_logic_and```| 1 | cycle | logic and latency |
|```alu_t_logic_or```| 1 | cycle | logic or latency |
|```alu_t_logic_not```| 1 | cycle | logic not latency |
|```alu_t_logic_xor```| 1 | cycle | logic xor latency |
|```alu_t_logic_cnot```| 5 | cycle | logic cnot latency |
|```alu_t_logic_shl```| 1 | cycle | logic shl latency |
|```alu_t_logic_shr```| 1 | cycle | logic shr latency |
|```alu_t_fp16_add```| 3 | cycle | FP16 add latency |
|```alu_t_fp16_sub```| 3 | cycle | FP16 sub latency |
|```alu_t_fp16_mul```| 3 | cycle | FP16 mul latency |
|```alu_t_fp16_fma```| 3 | cycle | FP16 fma latency |
|```alu_t_fp32_add```| 1 | cycle | FP32 add latency |
|```alu_t_fp32_sub```| 1 | cycle | FP32 sub latency |
|```alu_t_fp32_min```| 1 | cycle | FP32 min latency |
|```alu_t_fp32_max```| 1 | cycle | FP32 max latency |
|```alu_t_fp32_mul```| 1 | cycle | FP32 multiply latency |
|```alu_t_fp32_mad```| 1 | cycle | FP32 multiply-add latency |
|```alu_t_fp32_fma```| 1 | cycle | FP32 fused multiply-add latency |
|```alu_t_fp32_div```| 198 | cycle | FP32 divide latency |
|```alu_t_fp64_add```| 5 | cycle | FP64 add latency |
|```alu_t_fp64_sub```| 5 | cycle | FP64 sub latency |
|```alu_t_fp64_min```| 5 | cycle | FP64 min latency |
|```alu_t_fp64_max```| 5 | cycle | FP64 max latency |
|```alu_t_fp64_mul```| 5 | cycle | FP64 mul latency |
|```alu_t_fp64_mad```| 5 | cycle | FP64 mad latency |
|```alu_t_fp64_fma```| 5 | cycle | FP64 fma latency |
|```alu_t_fp64_div```| 156 | cycle | FP64 divide latency |
|```alu_t_int_setp```| 1 | cycle | integer setp latency |
|```alu_t_fp32_setp```| 1 | cycle | FP32 setp latency |
|```alu_t_int_selp```| 1 | cycle | integer selp latency |
|```alu_t_fp32_selp```| 1 | cycle | FP32 selp latency |
|```alu_t_mov```| 1 | cycle | mov latency |
|```alu_t_cvt```| 1 | cycle | cvt latency |
|```sfu_t_rcp```| 57 | cycle | special function unit reciprocal latency |
|```sfu_t_sqrt```| 57 | cycle | special function unit square root latency |
|```sfu_t_rsqrt```| 28 | cycle | special function unit reciprocal square root latency |
|```sfu_t_sin_cos```| 8 | cycle | special function unit sin / cos latency |
|```sfu_t_lg2```| 28 | cycle | special function unit log2 latency |
|```sfu_t_ex2```| 19 | cycle | special function unit exp2 latency |

### energy configuration

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```alu_e_int_add```| 0.0064 | nJ | integer add energy |
|```alu_e_int_sub```| 0.0064 | nJ | integer sub energy |
|```alu_e_int_min```| 0.0064 | nJ | integer min energy |
|```alu_e_int_max```| 0.0064 | nJ | integer max energy |
|```alu_e_int_mul```| 0.0092 | nJ | integer multiply energy |
|```alu_e_int_mad```| 0.0092 | nJ | integer multiply-add energy |
|```alu_e_int_div_s```| 4.2489 | nJ | integer divide energy (signed) |
|```alu_e_int_div_u```| 3.9254 | nJ | integer divide energy (unsigned) |
|```alu_e_int_rem_s```| 4.2100 | nJ | integer remainder of division energy (signed) |
|```alu_e_int_rem_u```| 3.9587 | nJ | integer remainder of division energy (unsigned) |
|```alu_e_int_abs```| 0.0647 | nJ | integer absolute energy |
|```alu_e_logic_and```| 0.0064 | nJ | logic and energy |
|```alu_e_logic_or```| 0.0064 | nJ | logic or energy |
|```alu_e_logic_not```| 0.0064 | nJ | logic not energy |
|```alu_e_logic_xor```| 0.0064 | nJ | logic xor energy |
|```alu_e_logic_cnot```| 0.0071 | nJ | logic cnot energy |
|```alu_e_logic_shl```| 0.0064 | nJ | logic shl energy |
|```alu_e_logic_shr```| 0.0064 | nJ | logic shr energy |
|```alu_e_fp16_add```| 0.0924 | nJ | FP16 add energy |
|```alu_e_fp16_sub```| 0.0924 | nJ | FP16 sub energy |
|```alu_e_fp16_mul```| 0.0924 | nJ | FP16 mul energy |
|```alu_e_fp32_add```| 0.0064 | nJ | FP32 add energy |
|```alu_e_fp32_sub```| 0.0064 | nJ | FP32 sub energy |
|```alu_e_fp32_min```| 0.0064 | nJ | FP32 min energy |
|```alu_e_fp32_max```| 0.0064 | nJ | FP32 max energy |
|```alu_e_fp32_mul```| 0.0021 | nJ | FP32 multiply energy |
|```alu_e_fp32_mad```| 0.0021 | nJ | FP32 multiply-add energy |
|```alu_e_fp32_fma```| 0.0021 | nJ | FP32 fused multiply-add energy |
|```alu_e_fp32_div```| 5.1096 | nJ | FP32 divide energy |
|```alu_e_fp64_add```| 0.3608 | nJ | FP64 add energy |
|```alu_e_fp64_sub```| 0.3608 | nJ | FP64 sub energy |
|```alu_e_fp64_min```| 0.3608 | nJ | FP64 min energy |
|```alu_e_fp64_max```| 0.3608 | nJ | FP64 max energy |
|```alu_e_fp64_div```| 3.7249 | nJ | FP64 divide energy |
|```sfu_e_rcp```| 2.4265 | nJ | special function unit reciprocal energy |
|```sfu_e_sqrt```| 2.4349 | nJ | special function unit square root energy |
|```sfu_e_rsqrt```| 1.2488 | nJ | special function unit reciprocal square root energy |
|```sfu_e_sin_cos```| 0.5887 | nJ | special function unit sin / cos energy |
|```sfu_e_lg2```| 1.2357 | nJ | special function unit log2 energy |
|```sfu_e_ex2```| 0.4798 | nJ | special function unit exp2 energy |

For the latency number and energy consumption number:
* The numbers are based on NVIDIA Volta TITAN V (V100) GPU, using optimized kernels and Performance Application Programming Interface (PAPI).

Please see the following papers for latency and enenrgy consumption number:
> "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs" <br/>
> "Verified Instruction-Level Energy Consumption Measurement for NVIDIA" GPUs



## DRAM 
The configuration file ```dram_config.json``` includes parameters for the DRAM, especially parameters for the DRAM bank.
It also contains the physical address mapping scheme.
The detailed configuration parameters are shown as the following table. 

### Address mapping

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```dram_addr_map```| dram_addr_map_1 | | address mapping scheme from global physical address to dram device address |

dram_addr_map_1:<br/>
| BANK_ROW_ADDR | BANK_COL_ADDR | PROCESSOR_Y | PROCESSOR_X | CORE_Y | CORE_X | PG_ID | BANK_ID | BANK_INTERFACE | 

For dram:
* Minimum addressable unit is a byte.
* For addresses generated by each instruction, **BANK_INTERFACE** field indicate byte index into CAS returned data (default: 16 bytes).
* For addresses sent from load/store unit, memory coalescing is already performed, so **BANK_INTERFACE** field should all be zero.

### DRAM parameters

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```dram_trans_queue_size```| 4 | | The size of queue for DRAM transactions |
|```dram_bank_io_width```| 16 | bytes | The data width of DRAM bank in bytes |
|```dram_bank_row```| 16384 | | The number of rows of a DRAM bank (default 16K rows) |
|```dram_bank_col```| 64 | | The number of columns of a DRAM bank (default 64 columns, column width = 16Byte) |
|```dram_clock_freq```| 1000 | MHz | The clock frequency of DRAM | 
|```dram_page_policy```| close_page | string | The page policy of DRAM bank (open_page or close_page) |
|```dram_schedule_policy```| FCFS | string | The DRAM transaction scheduling policy (FCFS or FRFCFS) |
|```dram_controller```| ideal | string | The DRAM memory controller design (ideal/simple) |
|```dram_bank_num_row_buf```| 4 | | Number of row buffers per DRAM bank |
|```dram_ideal_load_latency```| 10 | dram cycle | dram load latency in ideal case |
|```dram_ideal_store_latency```| 10 | dram cycle | dram store latency in ideal case |
|```dram_tRCDR```| 14 | dram cycle | RAS to CAS latency for a read command |
|```dram_tRCDW```| 10 | dram cycle | RAS to CAS latency for a write command |
|```dram_CL```| 2 | dram cycle | Bank level CAS read latency |
|```dram_CWL```| 2 | dram cycle | Bank level CAS write latency |
|```dram_tRAS```| 33 | dram cycle | Minimum row active period. [ACT]<->[PRE] |
|```dram_tRP```| 14 | dram cycle | Minimum row precharge period. [PRE]<->[ACT] |
|```dram_tRRDL```| 6 | dram cycle | Row cycle time (Same PG). [ACT]<->[ACT] |
|```dram_tRRDS```| 4 | dram cycle | Row cycle time (Differnt PGs). [ACT]<->[ACT] |
|```dram_tCCDL```| 4 | dram cycle | Minimum CAS to CAS delay (Same PG). [CAS]<->[CAS] |
|```dram_tCCDS```| 2 | dram cycle | Minimum CAS to CAS delay (Different PGs). [CAS]<->[CAS] |
|```dram_tRTPL```| 5 | dram cycle | Minimum read to precharge delay (Same PG). [RD]<->[PRE] |
|```dram_tRTPS```| 4 | dram cycle | Minimum read to precharge delay (Different PGs). [RD]<->[PRE] |
|```dram_tWTRL```| 8 | dram cycle | Minimum write to precharge delay (Same PG). [WR]<->[PRE] |
|```dram_tWTRS```| 3 | dram cycle | Minimum write to precharge delay (Different PGs). [WR]<->[PRE] |
|```dram_tWR```| 15 | dram cycle | Write Recovery Time. [data is written to a bank]<->[] |
|```dram_tFAW```| 16 | dram cycle | Four activation window |
|```dram_tRFC```| 350 | dram cycle | DRAM all banks refresh cycles |
|```dram_tREFI```| 3900 | dram cycle | DRAM all banks refresh interval |
|```dram_tRFCSB```| 160 | dram cycle | DRAM single bank refresh cycles |
|```dram_tREF1B```| 244 | dram cycle | DRAM single bank refresh interval |
|```dram_bank_read_energy```| 0.15302 | nJ | DRAM bank read energy |
|```dram_bank_write_energy```| 0.15304 | nJ | DRAM bank write energy |
|```dram_bank_refresh_energy```| 1.12872 | nJ | DRAM bank refresh energy |
|```dram_bank_act_energy```| 0.27837 | nJ | DRAM bank activation energy |
|```dram_bank_pre_energy```| 0.25604 | nJ | DRAM bank precharge energy |

## Load Store Unit

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```max_num_fb_local_prt_entry```| 8 | | maximum number of far-bank pending request table entries for local request |
|```max_num_fb_remote_prt_entry```| 8 | | maximum number of far-bank pending request table entries for remote request |
|```max_num_nb_prt_entry```| 8 | | maximum number of near-bank pending request table entries |

## Interconnect Network

### Interface Architecture

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```niu_req_queue_size```| 4 | | number of buffer slots in the network interface unit request queue |
|```niu_in_req_msg_queue_size```| 16 | | number of buffer slots in the network interface unit input request message queue |
|```niu_in_resp_msg_queue_size```| 16 | | number of buffer slots in the network interface unit input response message queue |
|```niu_out_msg_queue_size```| 16 | | number of buffer slots in the network interface unit outcoming message queue |
|```niu_readonly_dcache_size```| 32768 | byte | read only cache size |
|```niu_readonly_dcache_latency```| 1 | cycle | read only cache latency  |
|```niu_readonly_dcache_eviction_policy```| random | | read only cache eviction policy |
|```max_num_niu_track_req_table_entry```| 32 | | maximum number of network tracking request table entries |

### Topology

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```network_topology```| 2D_mesh | | Network Topology |

### Routing

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```network_routing```| deterministic | | Network Routing Algorithm |

### Flow Control

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```network_flow_control```| buffered | | Network FLow Control Algorithm |

### Router Architecture

| Parameter Name   |      Default Value      |  Unit | Description |
|:----------------:|------------------------:|------:|:------------|
|```network_router_clock_freq```| 2000 |  MHz | The clock frequency of a router |
|```network_onchip_bus_clock_freq```| 2000 |  MHz | The clock frequency of the onchip bus |
|```network_offchip_bus_clock_freq```| 2000 |  MHz | The clock frequency of the offchip bus |
|```network_onchip_bus_width```| 32 | byte | The width of the on-chip router-to-router data bus |
|```network_offchip_bus_width```| 16 | byte | The width of the off-chip router-to-router data bus |
|```network_router_in_channel_size```| 32 | | The number of packet buffer slots for a router's input channel |
|```network_router_out_channel_size```| 32 | | The number of packet buffer slots for a router's output channel |
|```network_packet_size```| 64 | byte | The size of a packet |
|```network_flit_size```| 16 | byte | The size of a flit |
|```network_metal_pitch```| 0.000080 | mm | metal pitch |
|```network_cw_cpl```| 0.000000000000267339 | F/mm | Wire left/right coupling capacitance |
|```network_cw_gnd```| 0.000000000000267339 | F/mm | Wire up/down ground capacitance |
|```network_cg```| 0.000000000000534 | F/mm | Device parameters for delay |
|```network_cgdl```| 0.0000000000001068 | F/mm | Device parameters for delay |
|```network_cd```| 0.000000000000267 | F/mm | Device parameters for delay |
|```network_cg_pwr```| 0.000000000000534 | F/mm | Device parameters for power |
|```network_cd_pwr```| 0.000000000000267 | F/mm | Device parameters for power |
|```network_vdd```| 0.9 | V | voltage |
|```network_onchip_bus_energy_per_bit```| TBD | nJ | on-chip bus energy per bit |
|```network_offchip_bus_energy_per_bit```| TBD | nJ | off-chip bus energy per bit |
|```network_chan_buf_access_energy```| TBD | nJ | network channel buffer access energy |
|```H_DFQD1```| 8 |  | Device parameters |

import json 
import os
import numpy as np 
import math


def is_power_of_two(num):
    return math.log(num, 2).is_integer()


def load_hardware_config(overwrite_config_file_path=None): 
    """Loads and infers hardware configurations. 
    
    This function loads hardware config and infer hardware config not specified
        explictly in the configuration file. 

    Args:
        overwrite_config_file_path: the file path of hardware config used to 
            overwrite the default configuration.  

    Returns:
        A dictionary of hardware config.  
    """

    # Load default configuration first
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    config_path_list = ["hardware_config.json", "processor_config.json", 
                        "core_config.json", "execution_unit_config.json", 
                        "register_file_config.json", "dram_config.json",
                        "shared_memory_config.json",
                        "load_store_unit_config.json",
                        "interconnect_network_config.json",
                        "simulation_config.json"]

    config_dict = {}
    for each_config_path in config_path_list:
        full_config_path = os.path.join(curr_dir, each_config_path)

        with open(full_config_path, "r") as f:
            tmp_config_dict = json.load(f)

            for config_name, config_value in tmp_config_dict.items():
                config_dict[config_name] = config_value 

    # Load and overwrite config 
    if overwrite_config_file_path is not None:
        with open(overwrite_config_file_path, "r") as f:
            overwrite_config = json.load(f)

        for config_name, config_value in overwrite_config.items():
            config_dict[config_name] = config_value 

    # Add inference here for missing configuration 
    
    # Infer the clock frequency for the simulation across multiple 
    # clock domains
    clock_freq_list = []
    for config_name, config_value in config_dict.items():
        if config_name.find("clock_freq") >= 0:
            clock_freq_list.append(config_value) 

    assert len(clock_freq_list) >= 1, "Configuration files need to " \
        "contain at least one clock source!"  

    config_dict["sim_clock_freq"] = int(np.lcm.reduce(clock_freq_list))

    # Infer total number of cores
    config_dict["total_num_cores"] = (
        config_dict["num_processor_x"] * config_dict["num_processor_y"]
        * config_dict["num_core_x"] * config_dict["num_core_y"]
    )

    # Infer shared bus width per core
    num_core_per_proc = config_dict["num_core_x"] * config_dict["num_core_y"]
    assert config_dict["processor_shared_bus_io_width"] \
        % num_core_per_proc == 0
    config_dict["core_shared_bus_io_width"] = \
        config_dict["processor_shared_bus_io_width"] \
        // num_core_per_proc
    
    # Infer total memory capacity
    config_dict["dram_capacity"] = (
        config_dict["num_processor_x"] * config_dict["num_processor_y"] 
        * config_dict["num_core_x"] * config_dict["num_core_y"] 
        * config_dict["num_pg"] * config_dict["num_pe"] 
        * config_dict["dram_bank_row"] * config_dict["dram_bank_col"] 
        * config_dict["dram_bank_io_width"]
    )

    # Infer the number of threads per warp 
    pg_width = config_dict["num_pe"] * config_dict["dram_bank_io_width"]
    assert (pg_width % config_dict["data_path_unit_size"])\
        == 0, "The PG I/O width should be a multiple of " \
        "the data path unit size"
    config_dict["num_threads_per_warp"] = pg_width // 4 
    assert (config_dict["num_threads_per_warp"] == 32), "The default value " \
        "should be 32. Please remove this assertion if a different warp " \
        "size is needed"

    # Shared memory
    assert config_dict["num_smem_bank"] == \
        config_dict["num_threads_per_warp"]
    # The io width of a shared memory bank in bytes
    config_dict["smem_io_width"] = config_dict["data_path_unit_size"] 

    # Check register file banks
    assert is_power_of_two(config_dict["num_subcore_reg_file_bank"]),\
        "num_subcore_reg_file_bank must be power of 2!"
    assert is_power_of_two(config_dict["num_pg_reg_file_bank"]),\
        "num_pg_reg_file_bank must be power of 2!"
    
    # Infer near-bank register file capacity
    core_reg_file_size = config_dict["subcore_reg_file_size"] \
        * config_dict["num_subcore"]
    assert core_reg_file_size % config_dict["num_pg"] == 0
    config_dict["pg_reg_file_size"] = core_reg_file_size \
        // config_dict["num_pg"]
    
    # Infer the total number of write ports
    # For far-bank register file
    config_dict["num_subcore_reg_file_write_port"] = \
        config_dict["num_fb_wb_port"] \
        + config_dict["num_fb_reg_move_engine"]
    # For near-bank register file
    config_dict["num_pg_reg_file_write_port"] = \
        config_dict["num_nb_wb_port"] \
        + config_dict["num_nb_reg_move_engine"]
    
    # Infer the total number of read ports 
    # For far-bank register file
    config_dict["num_read_port_fb_rmw"] = \
        config_dict["num_subcore_reg_file_write_port"]
    config_dict["num_subcore_reg_file_read_port"] = \
        config_dict["num_opc_lsu"] \
        + config_dict["num_opc_fb_alu"] \
        + config_dict["num_opc_sfu"] \
        + config_dict["num_opc_cfu"] \
        + config_dict["num_opc_syncu"] \
        + config_dict["num_fb_reg_move_engine"] \
        + config_dict["num_read_port_fb_rmw"]
    # For near-bank register file
    config_dict["num_read_port_nb_rmw"] = \
        config_dict["num_pg_reg_file_write_port"]
    config_dict["num_pg_reg_file_read_port"] = \
        config_dict["num_opc_nb_alu"] \
        + config_dict["num_opc_lsu_extension"] \
        + config_dict["num_nb_reg_move_engine"] \
        + config_dict["num_read_port_nb_rmw"]

    # Infer base read port id
    # For far-bank register file
    config_dict["base_regfile_read_port_id_cfu"] = 0
    config_dict["base_regfile_read_port_id_syncu"] = \
        config_dict["base_regfile_read_port_id_cfu"] \
        + config_dict["num_opc_cfu"]
    config_dict["base_regfile_read_port_id_lsu"] = \
        config_dict["base_regfile_read_port_id_syncu"] \
        + config_dict["num_opc_syncu"]
    config_dict["base_regfile_read_port_id_fb_alu"] = \
        config_dict["base_regfile_read_port_id_lsu"] \
        + config_dict["num_opc_lsu"]
    config_dict["base_regfile_read_port_id_sfu"] = \
        config_dict["base_regfile_read_port_id_fb_alu"] \
        + config_dict["num_opc_fb_alu"]
    config_dict["base_regfile_read_port_id_fb_reg_move_engine"] = \
        config_dict["base_regfile_read_port_id_sfu"] \
        + config_dict["num_opc_sfu"]
    config_dict["base_regfile_read_port_id_fb_rmw"] = \
        config_dict["base_regfile_read_port_id_fb_reg_move_engine"] \
        + config_dict["num_fb_reg_move_engine"]
    # For near-bank register file
    config_dict["base_regfile_read_port_id_nb_alu"] = 0
    config_dict["base_regfile_read_port_id_lsu_extension"] = \
        config_dict["base_regfile_read_port_id_nb_alu"] \
        + config_dict["num_opc_nb_alu"]
    config_dict["base_regfile_read_port_id_nb_reg_move_engine"] = \
        config_dict["base_regfile_read_port_id_lsu_extension"] \
        + config_dict["num_opc_lsu_extension"]
    config_dict["base_regfile_read_port_id_nb_rmw"] = \
        config_dict["base_regfile_read_port_id_nb_reg_move_engine"] \
        + config_dict["num_nb_reg_move_engine"]

    # Infer base write port id
    # For far-bank register file
    config_dict["base_regfile_write_port_id_fb_reg_move_engine"] = 0
    config_dict["base_regfile_write_port_id_fb_commit"] = \
        config_dict["base_regfile_write_port_id_fb_reg_move_engine"] \
        + config_dict["num_fb_reg_move_engine"]
    # For near-bank register file
    config_dict["base_regfile_write_port_id_nb_reg_move_engine"] = 0
    config_dict["base_regfile_write_port_id_nb_commit"] = \
        config_dict["base_regfile_write_port_id_nb_reg_move_engine"] \
        + config_dict["num_nb_reg_move_engine"]

    # Configure address space size for various components
    # Register file
    reg_file_alignment = \
        config_dict["data_path_unit_size"] \
        * config_dict["num_threads_per_warp"]
    assert config_dict["subcore_reg_file_size"] \
        % reg_file_alignment == 0
    num_reg_file_entry = config_dict["subcore_reg_file_size"] \
        // reg_file_alignment
    config_dict["subcore_reg_file_addr_len"] = \
        math.ceil(
            math.ceil(math.log2(num_reg_file_entry)) / 8
    )
    # DRAM bank
    num_dram_bank_entry = config_dict["dram_bank_row"] \
        * config_dict["dram_bank_col"]
    config_dict["dram_bank_addr_len"] = \
        math.ceil(
            math.ceil(math.log2(num_dram_bank_entry)) / 8
    )
    
    # Configure supported instruction
    config_dict["alu_instr"] =\
        {"add", "sub", "min", "max", "mul", "mad", "fma", "div", "rem", 
         "abs", "and", "or", "not", "xor", "cnot", "shl", "shr", "setp",
         "mov", "cvt", "selp"}
    config_dict["sfu_instr"] = \
        {"rcp", "sqrt", "rsqrt", "cos", "sin", "lg2", "ex2"}
    config_dict["lsu_instr"] = {"ld", "st", "cvta", "atom"}
    config_dict["cfu_instr"] = {"bra", "ret"}
    config_dict["syncu_instr"] = {"bar", "barrier"}
    
    config_dict["far_bank_op_set"] = config_dict["sfu_instr"] \
        | config_dict["cfu_instr"] \
        | config_dict["syncu_instr"] \
        | config_dict["lsu_instr"]

    # Configure Interconnect Network
    config_dict["network_msg_type"] = \
        {"ld_req", "ld_resp", "st_req", "st_resp"}
    assert config_dict["network_packet_size"] \
        % config_dict["network_onchip_bus_width"] == 0
    assert config_dict["network_packet_size"] \
        % config_dict["network_offchip_bus_width"] == 0
    config_dict["network_onchip_bus_packet_delay"] = \
        config_dict["network_packet_size"] \
        // config_dict["network_onchip_bus_width"]
    config_dict["network_offchip_bus_packet_delay"] = \
        config_dict["network_packet_size"] \
        // config_dict["network_offchip_bus_width"]
    assert config_dict["network_packet_size"] \
        % config_dict["network_flit_size"] == 0
    config_dict["network_router_crossbar_delay"] = \
        config_dict["network_packet_size"] \
        // config_dict["network_flit_size"]
    if config_dict["network_topology"] == "2D_mesh":
        # N, S, W, E
        config_dict["network_router_num_channel"] = 4
        config_dict["network_channel_index"] = {
            "N": 0, "S": 1, "W": 2, "E": 3
        }
    else:
        raise NotImplementedError(
            "Unknown network topology:{}"
            .format(config_dict["network_topology"])
        )

    return config_dict


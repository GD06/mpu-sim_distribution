import simpy 
import logging 
import json 
from copy import deepcopy 

from simulator.processor import Processor
from simulator.memory import Memory 
from simulator.network_config import config_2d_mesh


class Hardware:

    def __init__(self, config, env, log, filter_func):
        self.config = config 
        self.env = env 
        self.log = log 
        self.filter_func = filter_func 

        self.traceEvents = []
        self.processor_array = {}
        for i in range(config["num_processor_x"]):
            for j in range(config["num_processor_y"]):
                proc = Processor(
                    proc_id=(i, j),
                    env=env,
                    config=config,
                    log=log,
                    hardware=self,
                )
                self.processor_array[(i, j)] = proc 

        mem_alignment = (config["num_processor_x"] 
                         * config["num_processor_y"]
                         * config["num_core_x"]
                         * config["num_core_y"]
                         * config["num_pg"] * config["num_pe"]
                         * config["dram_bank_io_width"])
        self.mem = Memory(alignment=mem_alignment, size=config["dram_capacity"])

        # configure network connection
        # index into bus array: 
        # bus_array[(p_id_x_1, p_id_y_1), (c_id_x_1, c_id_y_1),
        #           (p_id_x_2, p_id_y_2), (c_id_x_2, c_id_y_2)]
        self.bus_array = {}
        self._config_network()
        return 

    def _config_network(self):
        if self.config["network_topology"] == "2D_mesh":
            config_2d_mesh(self)
        else:
            raise NotImplementedError(
                "Unknown network topology:{}"
                .format(self.config["network_topology"])
            )

    def addr_hashing(self, addr):
        """This function returns the detailed memory hierarchy mapping given 
        the global physical address.
        
        Args:
            addr: global physical address 

        Returns:
            loc: a location tuple formatted as 
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id, 
                bank_addr, bank_interface_offset)
        """
        proc_id_y = -1
        proc_id_x = -1
        core_id_y = -1
        core_id_x = -1
        pg_id = -1
        pe_id = -1
        bank_addr = -1
        bank_interface_offset = -1

        if (self.config["dram_addr_map"] == "dram_addr_map_1"):
            bank_interface_offset = addr % self.config["dram_bank_io_width"]
            addr = addr // self.config["dram_bank_io_width"]
            pe_id = addr % self.config["num_pe"]
            addr = addr // self.config["num_pe"]
            pg_id = addr % self.config["num_pg"]
            addr = addr // self.config["num_pg"]
            core_id_x = addr % self.config["num_core_x"]
            addr = addr // self.config["num_core_x"]
            core_id_y = addr % self.config["num_core_y"]
            addr = addr // self.config["num_core_y"]
            proc_id_x = addr % self.config["num_processor_x"]
            addr = addr // self.config["num_processor_x"]
            proc_id_y = addr % self.config["num_processor_y"]
            addr = addr // self.config["num_processor_y"]
            bank_addr = addr % (self.config["dram_bank_row"] 
                                * self.config["dram_bank_col"])
        else:
            raise NotImplementedError(
                "Unrecognized address mapping scheme: {}".format(
                    self.config["dram_addr_map"])
            )
        return (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id, 
                bank_addr, bank_interface_offset)

    def re_addr_hashing(self, loc):
        """This function returns the global physical address given the detailed
        memory hierarchy mapping.
        
        Args: loc: a location tuple formatted as
            (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id, 
            bank_addr, bank_interface_offset)

        Return:
            addr: global physical address
        """

        proc_id_y = loc[0]
        proc_id_x = loc[1]
        core_id_y = loc[2]
        core_id_x = loc[3]
        pg_id = loc[4]
        pe_id = loc[5]
        bank_addr = loc[6]
        bank_interface_offset = loc[7]
        addr = 0
        if(self.config["dram_addr_map"] == "dram_addr_map_1"):
            addr = addr + bank_addr
            addr = addr * self.config["num_processor_y"] + proc_id_y
            addr = addr * self.config["num_processor_x"] + proc_id_x
            addr = addr * self.config["num_core_y"] + core_id_y
            addr = addr * self.config["num_core_x"] + core_id_x
            addr = addr * self.config["num_pg"] + pg_id
            addr = addr * self.config["num_pe"] + pe_id
            addr = (addr * self.config["dram_bank_io_width"] 
                    + bank_interface_offset)
        else:
            raise NotImplementedError(
                "Unrecognized address mapping scheme: {}".format(
                    self.config["dram_addr_map"])
            )
        return addr

    def translate_bank_addr(self, bank_addr):
        """translate dram bank address to row and column address
        Args:
            bank_addr: the local bank address (aligned to bank interface width)
        Return:
            row_addr: bank row address
            col_addr: bank column address
        """
        row_addr = -1
        col_addr = -1
        if (self.config["dram_addr_map"] == "dram_addr_map_1"):
            col_addr = bank_addr % (self.config["dram_bank_col"])
            row_addr = ((bank_addr // (self.config["dram_bank_col"])) 
                        % self.config["dram_bank_row"])
        else:
            raise NotImplementedError(
                "Unrecognized address mapping scheme: {}".format(
                    self.config["dram_addr_map"])
            )
        return row_addr, col_addr

    def run_simulation(
        self, kernel, kernel_args, grid_dim, block_dim, block_schedule
    ):
        """This function invokes the simulation providing the executable 
        kernel, the value of kernel arguments, the dimension of grid, the 
        dimension of thread block, and the schedule of thread blocks. 

        Args:
            kernel: the executable kernel of MPU hardware. 
            kernel_args: a list of values for kernel arguments. 
            grid_dim: the dimension of block grid. 
            block_dim: the dimension of a thread block. 
            block_schedule: a dictionary mapping from thread block ID to the
                core ID. 

        Returns:
            (dur, sim_clock_freq): a tuple of the duration of simulation (the 
                total number of cycles) and the clock frequency of simulation 
                with the unit detailed in the configuration. 
        """

        # Step 1: construct the dictionary for the arguments of kernel and 
        # compute the register and shared memory usage for the kernel 
        # Also, initialize the latency of instructions in this kernel
        param_dict = {}
        assert len(kernel.arg_list) == len(kernel_args), "The length of " \
            "provided argument values does not equal to the length of " \
            "kernel argument list"

        for i in range(len(kernel.arg_list)):
            param_name = kernel.arg_list[i]
            param_value = kernel_args[i]
            param_dict[param_name] = param_value

        reg_usage_per_warp, shared_memory_usage_per_block = (
            kernel.compute_resource_usage(
                data_path_unit_size=self.config["data_path_unit_size"],
                num_threads_per_warp=self.config["num_threads_per_warp"],
                num_pe=self.config["num_pe"],
            ) 
        )

        kernel.init_instr_latency(self.config)

        # Step 2: construct a dictionary mapping from core ID to a list of 
        # thread blocks running on this core. 
        block_task_dict = {}

        for block_id_z in range(grid_dim[0]):
            for block_id_y in range(grid_dim[1]):
                for block_id_x in range(grid_dim[2]):
                    block_id = block_id_z 
                    block_id = block_id * grid_dim[1] + block_id_y 
                    block_id = block_id * grid_dim[2] + block_id_x 

                    global_core_id = block_schedule[block_id] 
                    loc = self.addr_hashing(
                        global_core_id 
                        * self.config["num_pg"]
                        * self.config["num_pe"]
                        * self.config["dram_bank_io_width"]
                    )
                    core_loc = loc[:4]

                    if core_loc not in block_task_dict:
                        block_task_dict[core_loc] = []
                    block_task_dict[core_loc].append(
                        (block_id_z, block_id_y, block_id_x)
                    )

        # Step 3: spawn a process for each task list 
        for each_core_id, each_task in block_task_dict.items():
            proc_id_y = each_core_id[0]
            proc_id_x = each_core_id[1]
            core_id_y = each_core_id[2]
            core_id_x = each_core_id[3]

            self.env.process(
                self.processor_array[(proc_id_x, proc_id_y)].core_array[
                    (core_id_x, core_id_y)].run_simulation(
                        reg_usage_per_warp, shared_memory_usage_per_block,  
                        kernel, param_dict, grid_dim, block_dim, each_task 
                )
            )

        # Step 4: invoke the simulation 
        start_cycle = self.env.now
        self.env.run() 
        end_cycle = self.env.now 

        dur = end_cycle - start_cycle 
        sim_clock_freq = self.config["sim_clock_freq"]

        return (dur, sim_clock_freq) 

    def get_perf_metrics(self):
        """Gets a dictionary of performance metrics. """
        perf_metrics = {}
        # Collect the performance metrics of this hardware module. 
        perf_metrics["total_num_cycles"] = self.env.now
        hw_config = deepcopy(self.config)
        for config_key, config_val in hw_config.items():
            if isinstance(config_val, set):
                hw_config[config_key] = list(config_val)
        perf_metrics["hardware_config"] = hw_config 

        # Collect the performance metrics of all hardware sub-moddules.
        for proc_id in self.processor_array.keys():
            proc_metrics = self.processor_array[proc_id].get_perf_metrics() 
            assert len(proc_metrics) == 1
            perf_metrics.update(proc_metrics) 

        all_bus_metrics = {}
        for bus_id in self.bus_array.keys():
            bus_metrics = self.bus_array[bus_id].get_perf_metrics()
            assert len(bus_metrics) == 1
            all_bus_metrics.update(bus_metrics)
        perf_metrics["network_bus"] = all_bus_metrics

        return {"hardware": perf_metrics} 

    def get_trace_events(self):
        """Get a list of trace events of the whole hardware."""
        # Collect the trace events of this hardware module 
        _traceEvents = deepcopy(self.traceEvents) 

        # Collect the trace events of all hardware sub-modules 
        for proc_id in self.processor_array.keys():
            _traceEvents += self.processor_array[proc_id].get_trace_events() 
        return _traceEvents 

    def dump_perf_metrics(self, output_file):
        """Saves the performance metrics of the simulator to the output file.

        Args:
            output_file: the file path of output file to store the performance
                metrics of the simulator. 

        Returns:
            None 
        """
        perf_metrics = self.get_perf_metrics() 
        with open(output_file, "w") as f:
            json.dump(perf_metrics, f, indent=2)
        return

    def dump_timeline(self, output_file):
        """Save the trace of events into the output file. 

        Args: 
            output_file: the file path of output file to store the timeline. 

        Returns:
            None 
        """
        events = self.get_trace_events() 
        
        timeline = {}
        timeline["traceEvents"] = events 
        timeline["displayTimeUnit"] = "ns"

        with open(output_file, "w") as f:
            json.dump(timeline, f, indent=4)

        return 


def _return_false(event):
    return False 


def init_hardware(hw_config_dict, logger=None, filter_func=None):
    """Initiate a hardware instance according to config 

    Args:
        hw_config_dict: the input hardware config 
        logger: (optional) the logger used to record logs 
        filter_func: (optional) the function pointer to a function filtering 
            trace events 

    Returns:
        hardware: the initalized hardware 
    """

    if logger is None:
        logging_level = logging.ERROR 

        # Uncomment the next line to enable a debug log output 
        # logging_level = logging.DEBUG 

        logger = logging.getLogger(__name__)
        logger.setLevel(logging_level)
        ch = logging.StreamHandler()
        ch.setLevel(logging_level)
        logger.addHandler(ch)

    if filter_func is None:
        filter_func = _return_false 

    env = simpy.Environment() 
    hardware = Hardware(
        env=env, config=hw_config_dict, log=logger, filter_func=filter_func
    )
    
    return hardware 

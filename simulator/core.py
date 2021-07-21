from copy import deepcopy 

from simulator.processing_group import ProcessingGroup
from simulator.subcore import Subcore 
from simulator.instr_cache import InstrCache
from simulator.subcore_table import WarpInfoTableEntry
from simulator.shared_memory import SharedMemory 
from simulator.subcore_pg_bus import SubcorePGBusArbiter
from simulator.network import NetworkInterfaceUnit
from simulator.load_store_unit_remote import LoadStoreUnitRemote


class Core:

    def __init__(self, core_id, env, config, log, processor):
        self.core_id = core_id
        self.env = env
        self.config = config 
        self.log = log 
        self.processor = processor 

        self.filter_func = processor.filter_func 
        self.traceEvents = [] 

        assert config["sim_clock_freq"] % config["core_clock_freq"] == 0, (
            "Undividable simulation clock frequency")
        self.clock_unit = config["sim_clock_freq"] // config["core_clock_freq"]
        self._loc_str = "Processor ID: {proc_id}, Core ID: {core_id}".format(
            proc_id=self.processor.proc_id,
            core_id=self.core_id
        )

        self.subcore_pg_bus_arbiter = SubcorePGBusArbiter(
            env=self.env,
            log=self.log,
            config=self.config,
            core=self
        )

        self.pg_array = []
        for i in range(config["num_pg"]):
            pg = ProcessingGroup(
                pg_id=i,
                env=env,
                config=config, 
                log=log,
                core=self, 
            )
            self.pg_array.append(pg)

        self.subcore_array = []
        for i in range(config["num_subcore"]):
            subcore = Subcore(
                subcore_id=i,
                env=env, 
                config=config, 
                log=log,
                core=self, 
            )
            self.subcore_array.append(subcore)

        self.icache = InstrCache(env=env, config=config, log=log, core=self) 
        self.smem = SharedMemory(
            env=env, log=log, config=config,
            clock_unit=self.clock_unit
        )

        self.niu = NetworkInterfaceUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            core=self
        )

        self.lsu_remote = LoadStoreUnitRemote(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.clock_unit,
            core=self
        )

        # block_id --> a list of barrier count, init to 0
        # list len = max_bar_id_per_block
        self.bar_count = {}
        # block_id --> a list of release signal, init to False
        # list len = max_bar_id_per_block
        self.bar_release = {}

        self.shared_memory_ptr = 0
        self.warp_id = 0 
        self.current_kernel = None
        self.param_dict = None
        self.grid_dim = None 
        self.block_dim = None 

        return

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}
        
        # Collect the performance metrics of all hardware sub-module
        # subcores
        for subcore_id in range(self.config["num_subcore"]):
            subcore_metrics = self.subcore_array[subcore_id].get_perf_metrics()
            assert len(subcore_metrics) == 1
            perf_metrics.update(subcore_metrics) 
        # processing groups
        for pg_id in range(self.config["num_pg"]):
            pg_metrics = self.pg_array[pg_id].get_perf_metrics()
            assert len(pg_metrics) == 1
            perf_metrics.update(pg_metrics)
        # icache
        icache_metrics = self.icache.get_perf_metrics()
        assert len(icache_metrics) == 1
        perf_metrics.update(icache_metrics)
        # subcore_pg_bus
        subcore_pg_bus_metrics = self.subcore_pg_bus_arbiter.get_perf_metrics()
        assert len(subcore_pg_bus_metrics) == 1
        perf_metrics.update(subcore_pg_bus_metrics)
        # shared memory
        smem_metrics = self.smem.get_perf_metrics()
        assert len(smem_metrics) == 1
        perf_metrics.update(smem_metrics)
        # remote load-store unit
        lsu_remote_metrics = self.lsu_remote.get_perf_metrics()
        assert len(lsu_remote_metrics) == 1
        perf_metrics.update(lsu_remote_metrics)
        # network interface unit
        niu_metrics = self.niu.get_perf_metrics()
        assert len(niu_metrics) == 1
        perf_metrics.update(niu_metrics)
        
        return {"core_{}".format(self.core_id): perf_metrics}

    def get_trace_events(self):
        """Get a list of trace events"""
        # Collect the trace evenets of this core
        _trace_Events = deepcopy(self.traceEvents)
        
        # Collect the trace events of all sub-modules 
        _trace_Events += self.subcore_pg_bus_arbiter.get_trace_events()
        for i in range(self.config["num_subcore"]):
            _trace_Events += self.subcore_array[i].get_trace_events()
        for i in range(self.config["num_pg"]):
            _trace_Events += self.pg_array[i].get_trace_events()
        return _trace_Events 

    def _append_trace_event(self, name, ts, dur, cat="", args={}):
        new_event = {}
        new_event["pid"] = "proc_{}_core_{}".format(
            self.processor.proc_id, self.core_id)
        new_event["tid"] = "main"
        new_event["ph"] = "X"
        new_event["name"] = name
        new_event["ts"] = ts / 1000
        new_event["dur"] = dur / 1000
        if len(cat) > 0: 
            new_event["cat"] = cat
        if len(args) > 0:
            new_event["args"] = args 

        if self.filter_func(new_event):
            self.traceEvents.append(new_event) 
        return 

    def _check_shared_memory_usage(self, shared_memory_usage_per_block):
        """Check whether the current available shared memory on the core is 
        sufficient to accomodate a new thread block 
        """
        new_ptr = self.shared_memory_ptr + shared_memory_usage_per_block 
        if new_ptr > self.config["smem_size"]:
            return False 
        return True

    def _check_available_resources(
        self, reg_usage_per_warp, shared_memory_usage_per_block, block_dim
    ):
        """Check whether the current available hardware resources on the 
        core to accommodate a new thread block.
        
        Args: 
            reg_usage_per_warp: the number of bytes of register file needed by
                a thread warp. 
            shred_memory_usage_per_block: the number of bytes of shared memory
                needed by a thread block.  
            block_dim: the dimension of a thread block. 

        Returns:
            A boolean variable to indicate whether the core is able to 
                schedule a new thread block. 
        """
        # Check whether there is sufficient shared memory 
        if not self._check_shared_memory_usage(shared_memory_usage_per_block):
            return False 

        warp_usage_subcores = [] 
        reg_usage_subcores = []
        reg_usage_pgs = []

        for i in range(self.config["num_subcore"]):
            warp_usage_subcores.append(0)
            reg_usage_subcores.append(0)

        for i in range(self.config["num_pg"]):
            reg_usage_pgs.append(0)

        warp_id = deepcopy(self.warp_id) 
        for tidz in range(block_dim[0]):
            for tidy in range(block_dim[1]):
                for tidx in range(
                    0, block_dim[2], self.config["num_threads_per_warp"]
                ):

                    subcore_id = warp_id % self.config["num_subcore"]
                    pg_id = warp_id % self.config["num_pg"]

                    warp_usage_subcores[subcore_id] += 1
                    reg_usage_subcores[subcore_id] += reg_usage_per_warp 
                    reg_usage_pgs[pg_id] += reg_usage_per_warp

                    warp_id = warp_id + 1

        for i in range(self.config["num_subcore"]):
            if not self.subcore_array[i].check_reg_usage(
                reg_usage_subcores[i]
            ):
                return False

            if not self.subcore_array[i].check_warp_usage(
                warp_usage_subcores[i]
            ):
                return False 

        for i in range(self.config["num_pg"]):
            if not self.pg_array[i].check_reg_usage(reg_usage_pgs[i]):
                return False 

        return True

    def _schedule_thread_block(
        self, reg_usage_per_warp, shared_memory_usage_per_block, 
        block_dim, block_id
    ):
        """Allocate harddware resources to schedule a thread block into 
        subcores, and initiate entries needed in the warp table. 

        Args: 
            reg_usage_per_warp: the number of bytes of register file needed by
                a thread warp. 
            shared_memory_usage_per_block: the number of bytes of shared memory 
                needed by a thread block.  
            block_dim: the dimension of a thread block. 
            block_id: a tuple including the ID of a thread block formated as 
                (block_id_z, block_id_y, block_id_x)
        """
        # Step 1: fill in warp table accordingly 
        for tidz in range(block_dim[0]):
            for tidy in range(block_dim[1]):
                for tidx in range(
                    0, block_dim[2], self.config["num_threads_per_warp"]
                ):

                    subcore_id = self.warp_id % self.config["num_subcore"]
                    pg_id = self.warp_id % self.config["num_pg"]

                    subcore_reg_base_addr = deepcopy(
                        self.subcore_array[subcore_id].reg_base_ptr
                    )
                    pg_reg_base_addr = deepcopy(
                        self.pg_array[pg_id].reg_base_ptr 
                    )
                    smem_base_addr = deepcopy(self.shared_memory_ptr) 

                    new_warp_info_table_entry = WarpInfoTableEntry(
                        thread_id=(tidz, tidy, tidx),
                        block_id=block_id,
                        pg_id=pg_id, 
                        subcore_reg_base_addr=subcore_reg_base_addr,
                        pg_reg_base_addr=pg_reg_base_addr, 
                        smem_base_addr=smem_base_addr,
                        prog_reg_offset=self.current_kernel.reg_offset,
                        prog_reg_size=self.current_kernel.reg_size,
                        prog_smem_offset=self.current_kernel.smem_offset,
                        prog_length=len(self.current_kernel.instr_list) 
                    )

                    entry_id = deepcopy(
                        self.subcore_array[subcore_id].num_active_warps
                    ) 
                    self.subcore_array[subcore_id].warp_info_table.entry[
                        entry_id] = new_warp_info_table_entry 

                    self.subcore_array[subcore_id].num_active_warps += 1
                    self.subcore_array[subcore_id].reg_base_ptr += (
                        reg_usage_per_warp 
                    )
                    self.subcore_array[subcore_id]\
                        .num_issued_not_commit_instr[block_id]\
                        = [0] * self.config["max_num_warp_per_subcore"]
                    self.pg_array[pg_id].reg_base_ptr += (
                        reg_usage_per_warp
                    )

                    self.warp_id = self.warp_id + 1

        # Step 2: increase the shared memory pointer accordingly 
        self.shared_memory_ptr += shared_memory_usage_per_block 

        # Step 3: update synchronization info accordingly
        self.bar_count[block_id] = [0] \
            * self.config["max_bar_id_per_block"]
        self.bar_release[block_id] = [False] \
            * self.config["max_bar_id_per_block"]

        return 

    def _invoke_subcore_simulation(self):
        """This function invokes the simulation of subcores and wait until 
        the completion. After that, this function reset the subcore status 
        into its original. 
        """
        for i in range(self.config["num_subcore"]):
            yield self.subcore_array[i].start_exec_cmd.put("start")

        for i in range(self.config["num_subcore"]):
            resp = yield self.subcore_array[i].finish_exec_resp.get()
            assert resp == "success", "The response {} does not " \
                "indicate a successful execution".format(resp)

        # Reset the hardware status for every components 
        for i in range(self.config["num_subcore"]):
            self.subcore_array[i].reset_status() 

        for i in range(self.config["num_pg"]):
            self.pg_array[i].reset_status() 

        # Release the shared memory 
        self.shared_memory_ptr = 0 

        # Release the synchronization barrier info
        self.bar_count = {}
        self.bar_release = {}

        # Reset the warp ID counter 
        self.warp_id = 0

        return 

    def run_simulation(self, reg_usage_per_warp, shared_memory_usage_per_block,
                       kernel, param_dict, grid_dim, block_dim, task_list):
        """This function schedules thread blocks into subcores when resources
        are available. 

        Args: 
            reg_usage_per_warp: the usage of registers per thread warp. 
            shared_memory_usage_per_block: the usage of shared memory per 
                thread block. 
            kernel: the kernel function to execute. 
            param_dict: the dictionary mapping from parameter names to values. 
            grid_dim: the dimension of a block grid, which is a tuple 
                formatted as (block_id_z, block_id_y, block_id_x)
            block_dim: the dimension of a thread block, which is a tuple 
                formatted as (thread_id_z, thread_id_y, thread_id_x). 
            task_list: a list of thread blocks scheduled to this core. 
        """ 
        self.current_kernel = kernel 
        self.param_dict = param_dict 
        self.grid_dim = grid_dim 
        self.block_dim = block_dim 

        task_list_length = len(task_list)
        curr_task_id = 0

        start_time = self.env.now 
        while curr_task_id < task_list_length:
            if self._check_available_resources(
                reg_usage_per_warp, 
                shared_memory_usage_per_block, 
                block_dim
            ):

                self._schedule_thread_block(
                    reg_usage_per_warp, shared_memory_usage_per_block, 
                    block_dim, task_list[curr_task_id]
                )
                curr_task_id = curr_task_id + 1

            else:
                # Invoke the subcore simulation until the completion of 
                # scheduled blocks so that resources can be reclaimed to 
                # continue scheduling the next thread block 
                yield self.env.process(self._invoke_subcore_simulation())

                # Record the simulation events  
                end_time = self.env.now 
                self._append_trace_event(
                    name="running", ts=start_time, dur=(end_time - start_time),
                    args={
                        "num_blocks_executed": curr_task_id, 
                        "total_num_blocks": task_list_length
                    }
                )
                start_time = end_time 

                # Display the simulatin progress 
                if self.config["display_simulation_progress"]:
                    if self.core_id == (0, 0):
                        print(
                            "Processor {} finished the simulation of {} blocks"
                            " out of {} blocks".format(
                                self.processor.proc_id, 
                                curr_task_id, 
                                task_list_length
                            )
                        )

        yield self.env.process(self._invoke_subcore_simulation())  

        # Record the simulation events  
        end_time = self.env.now 
        self._append_trace_event(
            name="running", ts=start_time, dur=(end_time - start_time),
            args={
                "num_blocks_executed": curr_task_id, 
                "total_num_blocks": task_list_length
            }
        )

        # Display the simulation progress 
        if self.config["display_simulation_progress"]:
            if self.core_id == (0, 0):
                print(
                    "Processor {} finished the simulation of {} blocks"
                    " out of {} blocks".format(
                        self.processor.proc_id, 
                        curr_task_id, 
                        task_list_length
                    )
                )

        self.current_kernel = None 
        self.param_dict = None 
        self.grid_dim = None 
        self.block_dim = None 

        return 

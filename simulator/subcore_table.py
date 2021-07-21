from copy import deepcopy 


class WarpInfoTableEntry:

    def __init__(
        self, 
        thread_id=None, 
        block_id=None, 
        pg_id=None,
        subcore_reg_base_addr=None,
        pg_reg_base_addr=None, 
        smem_base_addr=None,
        prog_reg_offset=None, 
        prog_reg_size=None, 
        prog_smem_offset=None, 
        prog_length=None
    ):

        self.thread_id = thread_id 
        self.block_id = block_id 
        self.pg_id = pg_id 
        self.subcore_reg_base_addr = subcore_reg_base_addr 
        self.pg_reg_base_addr = pg_reg_base_addr 
        self.smem_base_addr = smem_base_addr 
        self.prog_reg_offset = prog_reg_offset
        self.prog_reg_size = prog_reg_size 
        self.prog_smem_offset = prog_smem_offset 
        self.prog_length = prog_length
        self.warp_finished = False 

        return 


class WarpInfoTable:

    def __init__(self, config, log):
        self.config = config 
        self.log = log 

        self.entry = []
        self.reset() 
        return 

    def reset(self):
        """Delete all existing entries in the warp info table, and reset it into 
        its original status. 
        """
        self.entry = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.entry.append(WarpInfoTableEntry())

        return

    def check_all_finished(self, num_active_warps):
        for i in range(num_active_warps):
            if not self.entry[i].warp_finished:
                return False 
        return True 


class WarpPipelineTableEntry:

    def __init__(self, pc=0):
        self.pc = pc 
        self.next_pc = pc + 1

        self.warp_finished = False 
        self.valid = False
        self.instr = None 

        self.issued_to_icache = False
        self.issued_to_decode = False 
        self.skip_resource_contention = True 
        self.skip_exec_stall = True
        self.executed = False 
        return 


class WarpPipelineTable:

    def __init__(self, config, log):
        self.config = config
        self.log = log 

        self.entry = []
        self.reset()
        return 

    def reset(self):
        """Delete all existing entries in the fetch table, and reset it into 
        its original status. 
        """
        self.entry = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.entry.append(WarpPipelineTableEntry())

        return

    def check_all_finished(self, num_active_warps):
        for i in range(num_active_warps):
            if not self.entry[i].warp_finished:
                return False 
        return True


class SIMTStack:

    def __init__(self, max_depth, num_threads_per_warp):
        self.max_depth = max_depth 
        self._top_ptr = 0 
        self._stack_context = []

        simt_mask_str = "1" * num_threads_per_warp 
        self._stack_context.append((-1, 0, int(simt_mask_str, 2)))
        return 

    def top(self):
        return self._stack_context[self._top_ptr] 

    def pop(self):
        old_top_ptr = deepcopy(self._top_ptr) 
        self._top_ptr = self._top_ptr - 1
        return self._stack_context[old_top_ptr] 

    def push(self, new_entry):
        self._top_ptr = self._top_ptr + 1
        if self._top_ptr == len(self._stack_context):
            self._stack_context.append(new_entry) 
            assert len(self._stack_context) <= self.max_depth, "Fail to push " \
                "a new entry to the full stack"
        else:
            assert self._top_ptr < len(self._stack_context)
            self._stack_context[self._top_ptr] = new_entry 

        return 

    def check_converge(self, current_pc):
        top_entry = self._stack_context[self._top_ptr]
        if top_entry[0] == current_pc:
            return True
        else:
            return False 

    def get_simt_mask(self):
        top_entry = self._stack_context[self._top_ptr]
        return top_entry[2]


class StackTable:

    def __init__(self, config, log):
        self.config = config 
        self.log = log 

        self.entry = []
        self.reset()
        return 

    def reset(self):
        """Delete all existing entries in the stack table, and reset it into 
        its original status. 
        """
        self.entry = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.entry.append(
                SIMTStack(
                    self.config["max_simt_stack_depth"],
                    self.config["num_threads_per_warp"]
                )
            )

        return


class DepTableEntry:

    def __init__(self, table):
        self.table = table
        self.read_dict = {}
        self.write_dict = {}
        self.data_path_unit_size = table.config["data_path_unit_size"]
        return 

    def _get_op_name(self, raw_op_name):
        if not raw_op_name.startswith("%p"):
            return raw_op_name 
        new_index = int(raw_op_name[2:]) // self.data_path_unit_size 
        return "%p{}".format(new_index)

    def check_read(self, raw_op_name):
        """Check whether a read from the opeartor will be blocked 
        """
        op_name = self._get_op_name(raw_op_name)
        if not (op_name in self.write_dict):
            self.write_dict[op_name] = 0

        if self.write_dict[op_name] > 0:
            return False 
        return True  

    def check_write(self, raw_op_name):
        """Check whether a write to the opeartor will be blocked
        """
        op_name = self._get_op_name(raw_op_name)
        if not (op_name in self.read_dict):
            self.read_dict[op_name] = 0
        if not (op_name in self.write_dict):
            self.write_dict[op_name] = 0

        if self.read_dict[op_name] > 0:
            return False 
        if self.write_dict[op_name] > 0:
            return False 
        return True 

    def increase_read(self, raw_op_name):
        """Record an ongoing read to the register 
        """
        op_name = self._get_op_name(raw_op_name) 
        if not (op_name in self.read_dict):
            self.read_dict[op_name] = 0
        self.read_dict[op_name] += 1
        # update performance counter
        self.table.num_write += 1
        return 

    def increase_write(self, raw_op_name):
        """Record an ongoing write to the register 
        """
        op_name = self._get_op_name(raw_op_name)
        if not (op_name in self.write_dict):
            self.write_dict[op_name] = 0
        self.write_dict[op_name] += 1
        # update performance counter
        self.table.num_write += 1
        return 

    def decrease_read(self, raw_op_name):
        op_name = self._get_op_name(raw_op_name)
        self.read_dict[op_name] -= 1
        assert self.read_dict[op_name] >= 0
        # update performance counter
        self.table.num_write += 1
        return 

    def decrease_write(self, raw_op_name):
        op_name = self._get_op_name(raw_op_name)
        self.write_dict[op_name] -= 1
        assert self.write_dict[op_name] >= 0
        # update performance counter
        self.table.num_write += 1
        return 


class DepTable:

    def __init__(self, config, log):
        self.config = config 
        self.log = log 

        self.entry = []
        self.reset() 
        # performance counter
        self.num_read = 0
        self.num_write = 0
        return 

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_read"] = self.num_read
        perf_metrics["num_write"] = self.num_write
        return {"dep_table": perf_metrics}
    
    def reset(self):
        """Delete all entries in the dependency table and reset them into 
        original status. 
        """
        self.entry = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.entry.append(DepTableEntry(self))

        return


class RegTrackTableEntry:

    def __init__(self, config, log, reg_file, table):
        """This table entry keeps track of the valid information of far-bank
        register file entry and near-bank register file entry per warp. 
        Note that each register file entry is aligned to bank interface.
        """
        self.config = config
        self.log = log
        self.reg_file = reg_file
        self.table = table
        self.alignment = reg_file.alignment
        # register entry starting addr --> valid info
        # valid: True
        # invalid: False
        self.fb_valid_dict = {}
        self.nb_valid_dict = {}

    def write_update(self, op_str, reg_file_type):
        """When writing to a register file location, update register
        tracking table
        """
        # update performance counter
        self.table.num_write += 1
        if reg_file_type == "near-bank":
            self.nb_valid_dict[op_str] = True
            self.fb_valid_dict[op_str] = False
        elif reg_file_type == "far-bank":
            self.fb_valid_dict[op_str] = True
            self.nb_valid_dict[op_str] = False
        else:
            raise NotImplementedError(
                "Unknown register file type:{}".format(reg_file_type)
            )

    def mov_update(self, op_str, reg_file_type):
        """When moving data to a register file location, update register
        tracking table
        """
        # update performance counter
        self.table.num_write += 1
        if reg_file_type == "near-bank":
            self.nb_valid_dict[op_str] = True
        elif reg_file_type == "far-bank":
            self.fb_valid_dict[op_str] = True
        else:
            raise NotImplementedError(
                "Unknown register file type:{}".format(reg_file_type)
            )

    def check_ready(self, op_str, is_near_bank):
        """Check if the given register file locations contain valid entries
        Args:
            op_str: operand name in string
            is_near_bank: True for near-bank; False for far-bank
        """
        # update performance counter
        self.table.num_read += 1
        # perform checking
        valid_dict = self.nb_valid_dict if is_near_bank\
            else self.fb_valid_dict
        if op_str not in valid_dict:
            assert False, "{} does not exist in reg-file"\
                .format(op_str)
        if valid_dict[op_str] is not True:
            return False
        else:
            return True


class RegTrackTable:

    def __init__(self, config, log, reg_file):
        """This table contains each entry per warp to track its regiser 
        location validity information.
        """
        self.config = config
        self.log = log
        self.reg_file = reg_file
        self.alignment = reg_file.alignment
        self.entry = []
        self.reset()
        # performance counter
        self.num_read = 0
        self.num_write = 0
        return

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_read"] = self.num_read
        perf_metrics["num_write"] = self.num_write
        return {"reg_track_table": perf_metrics}

    def reset(self):
        """Delete all entries in the register tracking table and reset them 
        into original status.
        """
        self.entry = []
        for i in range(self.config["max_num_warp_per_subcore"]):
            self.entry.append(
                RegTrackTableEntry(
                    config=self.config,
                    log=self.log,
                    reg_file=self.reg_file,
                    table=self
                )
            )

        return

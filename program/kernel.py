from copy import deepcopy 
from program.instruction import Instr 


def _get_num_bits_by_type(type_str):
    if type_str == ".pred":
        return 8  
    elif type_str == ".f32":
        return 32
    elif type_str == ".b8":
        return 8 
    elif type_str == ".b32":
        return 32
    elif type_str == ".b64":
        return 64
    else:
        raise NotImplementedError 


def _align_a_to_b(a, b):
    return ((a - 1) // b + 1) * b


class RegInfo:

    def __init__(self, data_type, reg_prefix, num_regs):
        self.data_type = data_type 
        self.reg_prefix = reg_prefix 
        self.num_regs = num_regs 
        self.data_size = _get_num_bits_by_type(data_type) 
        self.total_size = self.data_size * self.num_regs 

        return

    def update_num_regs(self, new_num_regs):
        """Update the number of registers"""
        self.num_regs = new_num_regs 
        self.total_size = self.data_size * self.num_regs 
        return 

    def __str__(self):
        return "reg prefix: {}, num of regs: {}, data type: {} " \
            "total size: {}".format(
                self.reg_prefix, self.num_regs, self.data_type,
                self.data_size * self.num_regs 
            )


class SMEMInfo:

    def __init__(self, alignment, data_type, var_name, num_elements):
        self.alignment = alignment 
        self.data_type = data_type 
        self.data_size = _get_num_bits_by_type(data_type)
        assert self.data_size % 8 == 0 
        self.var_name = var_name 
        self.num_elements = num_elements 
        self.total_size = 8 * _align_a_to_b(
            self.data_size * num_elements // 8, alignment
        )

        return 

    def __str__(self):
        return "shared memory var name: {}, num of elements: {}, " \
            "data size: {}, total_size: {}".format(
                self.var_name, self.num_elements, 
                self.data_size, self.total_size 
            )


class Kernel:

    def __init__(self, lines, log):
        self.kernel_context = lines
        self.log = log
        self.reg_loc = {} 
        self._analysis()
        self.log = None 
        return 

    def _analysis(self):
        """This function performs analysis on kernel context, and 
        it includes two major steps as explained as follows:

        Step 1: get the name of kernel and function parameters 
    
        Step 2: go through the rest of instructions 
          Type I: indicate the usage of registers 
          Type II: indicate the usage of shared memory
          Type III: other instructions
        """

        # Step 1
        self.kernel_name = self.kernel_context[0].split(" ")[-1].strip("(")

        start_pos = 0
        end_pos = 0
        self.arg_list = []

        for line_id in range(len(self.kernel_context)):
            curr_line = self.kernel_context[line_id]

            if curr_line.startswith(".param"):
                param_name = curr_line.split(" ")[-1].strip(",")
                self.arg_list.append(param_name)

            if curr_line.find("{") >= 0:
                start_pos = line_id 

            if curr_line.find("}") >= 0:
                end_pos = line_id 

        # Step 2
        self.shared_memory_usage = []
        self.reg_usage = []
        self.code_blocks = {}
        self.instr_list = []

        curr_pc = 0
        self.code_blocks["BB0"] = 0

        for line_id in range(start_pos + 1, end_pos):
            curr_line = self.kernel_context[line_id]
            assert len(curr_line) > 0, "Encounter an empty line!"

            if curr_line.startswith(".reg"):
                data_type = curr_line.split(" ")[1].strip(" \n\t")

                reg_info = curr_line.split(" ")[-1].strip(" \n\t") 
                pos_1 = reg_info.find("<")
                pos_2 = reg_info.find(">")

                reg_prefix = reg_info[:pos_1]
                num_regs = int(reg_info[pos_1 + 1: pos_2]) 

                self.reg_usage.append(RegInfo(data_type, reg_prefix, num_regs))

            elif curr_line.startswith(".shared"):
                alignment = int(curr_line.split(" ")[2].strip(" \n\t"))
                data_type = curr_line.split(" ")[3].strip(" \n\t")

                smem_info = curr_line.split(" ")[-1].strip(" \n\t")
                pos_1 = smem_info.find("[")
                pos_2 = smem_info.find("]")

                smem_name = smem_info[:pos_1] 
                num_elements = int(smem_info[pos_1 + 1: pos_2])

                self.shared_memory_usage.append(
                    SMEMInfo(alignment, data_type, smem_name, num_elements)
                )

            elif curr_line.startswith("BB") and curr_line.find(":") >= 0:
                pos = curr_line.find(":")
                self.code_blocks[curr_line[:pos]] = deepcopy(curr_pc)

            else:
                self.log.debug("Parsing the instruction: {}".format(curr_line))
                self.instr_list.append(Instr(curr_line))
                curr_pc = curr_pc + 1

        self.log.debug(
            "Finished parsing the kernel: {}".format(self.kernel_name)
        )
        self.log.debug("Argument list: {}".format(self.arg_list))
        self.log.debug("The usage of register resources:")
        for each_reg in self.reg_usage:
            self.log.debug(each_reg) 
        for each_smem in self.shared_memory_usage:
            self.log.debug(each_smem)

        self.log.debug("Instructions:")
        for i in range(len(self.instr_list)):
            self.log.debug("PC={}: {}".format(i, self.instr_list[i]))
        self.log.debug("Coe blocks:")
        self.log.debug(self.code_blocks)
        return

    def compute_resource_usage(self, data_path_unit_size, 
                               num_threads_per_warp, num_pe):
        """Compute the usage of registers per warp and shared memory per block 

        Args:
            data_path_unit_size: the data path unit access size 
                (default: 4byte).
            num_threads_per_warp: the number of threads per warp.  
            num_pe: the number of PEs per PG.  

        Returns:
            A tuple (reg_usage_per_warp, shared_memory_usage_per_block) 
                contains the usage of registers and shared memory. 
        """
        assert (num_threads_per_warp % num_pe) == 0, "The number of threads "  \
            "per warp should be dividable to the number of PEs."
        num_threads_per_pe = num_threads_per_warp // num_pe 
        
        # calculate the alignment of address to access register file 
        # and shared memory on hardware.
        alignment = data_path_unit_size * num_threads_per_warp

        # Step 1: compute the usage of register file
        self.reg_offset = {}
        self.reg_size = {}
        reg_usage_per_warp = 0

        for each_reg_info in self.reg_usage:
            prefix_name = each_reg_info.reg_prefix 
            self.reg_offset[prefix_name] = deepcopy(reg_usage_per_warp) 

            size_in_bits_per_pe = each_reg_info.data_size * num_threads_per_pe
            size_in_bytes_per_pe = ((size_in_bits_per_pe - 1) // 8) + 1
            assert size_in_bytes_per_pe > 0, "The register should at least " \
                "take one byte in the register file of PE"

            size_in_bytes_per_warp = size_in_bytes_per_pe * num_pe 
            self.reg_size[prefix_name] = deepcopy(size_in_bytes_per_warp) 

            total_size_in_bytes = _align_a_to_b(
                size_in_bytes_per_warp * each_reg_info.num_regs, alignment 
            )

            reg_usage_per_warp += total_size_in_bytes 

        # Step 2: compute the usage of shared memory 
        shared_memory_usage_per_block = 0
        self.smem_offset = {}

        for each_smem_info in self.shared_memory_usage:
            var_name = each_smem_info.var_name 
            self.smem_offset[var_name] = deepcopy(shared_memory_usage_per_block)

            shared_memory_usage_per_block += _align_a_to_b(
                (each_smem_info.total_size - 1) // 8 + 1, alignment 
            )

        return (reg_usage_per_warp, shared_memory_usage_per_block) 

    def init_instr_latency(self, config):
        for instr in self.instr_list:
            instr.set_latency(config)

    def __str__(self):
        raw_context = "\n".join(self.kernel_context)
        kernel_info = raw_context + "\n\nInstruction Information:"
        for each_instr in self.instr_list:
            kernel_info = kernel_info + "\n{}".format(each_instr)
        return kernel_info 

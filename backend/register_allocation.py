import logging

import networkx as nx 
import itertools 
from copy import deepcopy 
from program.instruction import parse_reg_str 


def allocate_register(input_kernel, hw_config, logger=None):
    """This function allocates physical registers to each virtual register.

    Args:
        input_kernel: the kernel before register allocation 
        hw_config: the hardware configuration with the maximum register file
            size for each subcore 
        logger: a logger to log information for the purpose of debugging

    Returns:
        output_kernel: the kernel with physical register allocated 
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

    list_of_liveness_set = _liveness_analysis(input_kernel, logger)
    mapping_dict = _generate_reg_mapping(
        list_of_liveness_set, input_kernel.reg_loc, logger
    )

    output_kernel = deepcopy(input_kernel)
    _update_kernel_information(output_kernel, mapping_dict, logger)
    
    for curr_instr in output_kernel.instr_list:
        logger.debug(curr_instr)

    assert _check_reg_usage(input_kernel, output_kernel, hw_config, logger)
    return output_kernel 


def _liveness_analysis(input_kernel, logger):
    """Get a liveness set for each program point. 

    Args:
        input_kernel: the kernel containing a program
        logger: a logger to log information for the purpose of debugging

    Returns:
        A list of liveness set for the program points before and after each
            instruction in the program. 
    """ 
    num_instrs = len(input_kernel.instr_list)
    before_instr_liveness = [None] * num_instrs 
    after_instr_liveness = [None] * num_instrs
    after_instr_liveness[num_instrs - 1] = set()
    count = 0

    while True:
        logger.debug("{}-th iteration:".format(count))
        count += 1
        new_before_instr_liveness = deepcopy(before_instr_liveness)
        new_after_instr_liveness = deepcopy(after_instr_liveness) 

        for i in reversed(range(num_instrs)):
            curr_instr = input_kernel.instr_list[i]
            next_pc = True if (i != (num_instrs - 1)) else False
            after_set = set() 

            # This is a jump instruction 
            if "dst_pc" in curr_instr.metadata:
                dst_codeblock = curr_instr.metadata["dst_pc"]
                dst_pc = input_kernel.code_blocks[dst_codeblock]
                if new_before_instr_liveness[dst_pc] is not None:
                    after_set.update(new_before_instr_liveness[dst_pc])
                # This is an unconditional jump instruction
                if "pred_reg" not in curr_instr.metadata:
                    next_pc = False 

            if (next_pc) and (new_before_instr_liveness[i + 1] is not None):
                after_set.update(new_before_instr_liveness[i + 1])

            new_after_instr_liveness[i] = deepcopy(after_set)
            before_set = deepcopy(after_set)

            for each_op in curr_instr.dst_operands:
                if (each_op.isnormalreg()) and (each_op.op_str in after_set):
                    before_set.remove(each_op.op_str)

            for each_op in curr_instr.src_operands:
                if each_op.isnormalreg():
                    before_set.add(each_op.op_str)

            if "pred_reg" in curr_instr.metadata:
                before_set.add(curr_instr.metadata["pred_reg"].op_str) 

            new_before_instr_liveness[i] = deepcopy(before_set)
            logger.debug(
                "Instr: {}, liveness after: {}, liveness before: {}".format(
                    curr_instr.instr_str, after_set, before_set)
            )

        change = False 
        for i in range(num_instrs):
            if new_before_instr_liveness[i] != before_instr_liveness[i]:
                change = True 
                break 

            if new_after_instr_liveness[i] != after_instr_liveness[i]:
                change = True 
                break 

        if count >= num_instrs * 18:
            logger.error(
                "According to the ordering theory, the upper-bound number of"
                " iterations: {} has been achieved".format(count)
            )
            assert False 

        if not change:
            break 
        else:
            before_instr_liveness = deepcopy(new_before_instr_liveness)
            after_instr_liveness = deepcopy(new_after_instr_liveness)
            logger.debug("\n\n")

    # Merge all sets from these two liveness lists
    list_of_liveness_set = [x for x in before_instr_liveness if x is not None]
    list_of_liveness_set += [x for x in after_instr_liveness if x is not None]

    return list_of_liveness_set 


def _generate_reg_mapping(list_of_liveness_set, reg_loc, logger):
    """Generate the mappings from virtual registers to physical registers for 
    each kind of register prefix 

    Args:
        list_of_liveness_set: a list of liveness set 
        reg_loc: the location of virtual registers 
        logger: a logger to log information for the purpose of debugging 

    Returns:
        A dictionary from register prefix to the mapping of registers with the
          same prefix 
    """
    mapping_dict = {}

    reg_prefix_set = set() 
    for liveness_set in list_of_liveness_set:
        for each_reg in liveness_set:
            reg_prefix, _ = parse_reg_str(each_reg)
            reg_prefix_set.add(reg_prefix)

    for each_reg_prefix in reg_prefix_set:
        logger.debug(
            "Allocating registers for the prefix: {}".format(each_reg_prefix)
        )
        G = nx.Graph()

        for each_liveness_set in list_of_liveness_set:
            for pair in itertools.permutations(list(each_liveness_set), r=2):
                prefix_0, index_0 = parse_reg_str(pair[0])
                prefix_1, index_1 = parse_reg_str(pair[1])

                if prefix_0 == each_reg_prefix:
                    assert isinstance(index_0, int)
                    G.add_node(index_0)

                if prefix_1 == each_reg_prefix:
                    assert isinstance(index_1, int)
                    G.add_node(index_1)

                if (prefix_0 != prefix_1) or (prefix_0 != each_reg_prefix):
                    continue 

                assert index_0 != index_1 
                G.add_edge(index_0, index_1)
                G.add_edge(index_1, index_0)

        v_regs = [k for k in reg_loc.keys()]
        for pair in itertools.permutations(v_regs, r=2):
            prefix_0, index_0 = parse_reg_str(pair[0])
            prefix_1, index_1 = parse_reg_str(pair[1])

            if (prefix_0 != prefix_1) or (prefix_0 != each_reg_prefix):
                continue 

            # Registers on different locations will not be allocated together
            if reg_loc[pair[0]] != reg_loc[pair[1]]:
                G.add_edge(index_0, index_1)

        logger.debug("Coloring the graph...")
        coloring_results = nx.coloring.greedy_color(G)
        logger.debug("Coloring results: {}".format(coloring_results))
        mapping_dict[each_reg_prefix] = coloring_results 

    return mapping_dict


def _update_kernel_information(input_kernel, mapping_dict, logger):
    """Update the register information inside the program using the allocated
    results. 

    Args:
        input_kernel: the original kernel using virtual register information
        mapping_dict: mappings from virtual registers to physical registers
        logger: a logger to log information for the purpose of debugging 
    """
    def _update_op(op):
        if op.reg_prefix in mapping_dict:
            new_index = mapping_dict[op.reg_prefix][op.reg_index]
            op.update_index(new_index)

    for curr_instr in input_kernel.instr_list:
        for each_op in curr_instr.src_operands:
            _update_op(each_op)

        for each_op in curr_instr.dst_operands:
            _update_op(each_op)

        if "pred_reg" in curr_instr.metadata:
            _update_op(curr_instr.metadata["pred_reg"])

    for each_reg_usage in input_kernel.reg_usage:
        if each_reg_usage.reg_prefix in mapping_dict:
            prefix = each_reg_usage.reg_prefix 
            phy_ids = [v for _, v in mapping_dict[prefix].items()]
            new_num_regs = len(set(phy_ids))
            each_reg_usage.update_num_regs(new_num_regs)

    return 


def _check_reg_usage(before_alloc, after_alloc, hw_config, logger):
    """Check whether the register file usage satisfies harddware constraints.

    Args:
        before_alloc: the kernel before register allocation. 
        after_alloc: the  kernel after register allocation. 
        hw_config: hardware configurations.
        logger: a logger to log information for the purpose of debugging 

    Returns:
        A bool indicating whether the register usage satisfies hardware 
            constraints.
    """
    reg_usage_per_warp, _ = before_alloc.compute_resource_usage(
        data_path_unit_size=hw_config["data_path_unit_size"],
        num_threads_per_warp=hw_config["num_threads_per_warp"],
        num_pe=hw_config["num_pe"],
    )
    logger.debug("The usage of registers before allocation:")
    for each_reg_usage in before_alloc.reg_usage:
        logger.debug(each_reg_usage)
    logger.debug(
        "Total size of registers used: {} bytes\n".format(reg_usage_per_warp)
    )

    reg_usage_per_warp, _ = after_alloc.compute_resource_usage(
        data_path_unit_size=hw_config["data_path_unit_size"],
        num_threads_per_warp=hw_config["num_threads_per_warp"],
        num_pe=hw_config["num_pe"],
    )
    logger.debug("The usage of registers after allocation:")
    for each_reg_usage in after_alloc.reg_usage:
        logger.debug(each_reg_usage)
    logger.debug(
        "Total size of registers used: {} bytes\n".format(reg_usage_per_warp)
    )

    if reg_usage_per_warp > hw_config["subcore_reg_file_size"]:
        logger.error(
            "Exceeding the maximum limit {} bytes".format(
                hw_config["subcore_reg_file_size"])
        )
        return False 

    return True 

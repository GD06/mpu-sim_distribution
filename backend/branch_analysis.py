import logging 

import networkx as nx 
from copy import deepcopy 


def reconvergence_analysis(input_kernel, mode="instr", logger=None):
    """This function analyzes the reconvergence point for each
    branch instruction for handling thread divergence during 
    the execution on hardware. 

    Args:
        input_kernel: the kernel before analysis 
        logger: a logger to log information for the purpose of debugging 

    Returns:
        output_kernel: the kernel with each branch instruction
            annotated the reconvergence point 
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

    if mode == "instr": 
        output_kernel = _instr_reconvergence(input_kernel, logger)
    elif mode == "code_block":
        output_kernel = _code_block_reconvergence(input_kernel, logger)
    else:
        raise NotImplementedError("Unknown analysis mode: {}".format(mode))

    return output_kernel 


def _code_block_reconvergence(input_kernel, logger):
    """This function performs the reconvergence analysis at the 
    granularity of code blocks. Each code block is abstracted as 
    a node in the flow graph. Then the immediate post dominator 
    is computed. Finally, each branch instruction will reconverge 
    at the entry point of the immediate post dominator of the code 
    block to which this instruction belongs. 

    Args: 
        input_kernel: the kenel before analysis 
        logger: a logger to log information for the purpose of debugging 

    Returns:
        output_kernel: the kernel with each branch instruction 
            annotated the reconvergence point 
    """

    code_block_dict = deepcopy(input_kernel.code_blocks) 
    num_instrs = len(input_kernel.instr_list) 

    starting_instr = [False] * num_instrs 
    for k, v in code_block_dict.items():
        starting_instr[v] = True 

    new_code_block_id = 0 
    for i in range(num_instrs):
        curr_instr = input_kernel.instr_list[i]
        if "dst_pc" in curr_instr.metadata:
            if not starting_instr[i + 1]:
                new_code_block = "_N_BB_{}".format(new_code_block_id) 
                new_code_block_id += 1
                code_block_dict[new_code_block] = i + 1
                starting_instr[i + 1] = True 

    code_block_map = [None] * num_instrs 
    for k, v in code_block_dict.items():
        code_block_map[v] = k 

    curr_code_block = None 
    for i in range(num_instrs):
        if code_block_map[i] is None:
            code_block_map[i] = deepcopy(curr_code_block) 
        else:
            curr_code_block = deepcopy(code_block_map[i])

    edge_list = []
    for i in range(num_instrs - 1):
        curr_code_block = code_block_map[i]
        next_code_block = code_block_map[i + 1]

        # This is the bounddary between two adjacent code blocks 
        if curr_code_block != next_code_block:
            curr_instr = input_kernel.instr_list[i]

            # This is a jump instruction 
            if "dst_pc" in curr_instr.metadata:
                dst_code_block = curr_instr.metadata["dst_pc"]
                edge_list.append((dst_code_block, curr_code_block))

                # This is an unconditional jump instruction 
                if "pred_reg" not in curr_instr.metadata: 
                    continue 

            edge_list.append((next_code_block, curr_code_block)) 

    G = nx.DiGraph(edge_list) 
    start_node = code_block_map[-1]
    idom = nx.immediate_dominators(G, start_node)

    output_kernel = deepcopy(input_kernel) 
    for i in range(num_instrs - 1):
        curr_instr = input_kernel.instr_list[i] 
        if "dst_pc" in curr_instr.metadata:
            curr_code_block = code_block_map[i]
            pdom_code_block = idom[curr_code_block] 
            pdom_pc = code_block_dict[pdom_code_block] 

            output_kernel.instr_list[i].metadata["pdom"] = pdom_pc  
            logger.debug(
                "Instruction: {}\nPost Dominator: {}".format(
                    curr_instr, input_kernel.instr_list[pdom_pc])
            )

    return output_kernel 


def _instr_reconvergence(input_kernel, logger):
    """This function performs the reconvergence analysis at the 
    granularity of instructions. Each instruction is abstrated as 
    a node in the flow graph. Then the immediate post dominator 
    is computed. Finally, each branch instruction will reconverge 
    at the immediate post dominator of that instruction. 

    Args:
        input_kernel: the kernel before analysis 
        logger: a logger to log information for the purpose of debugging

    Returns:
        output_kernel: the kernel with each branch instruction 
            annotated the reconvergence point 
    """

    edge_list = []
    num_instrs = len(input_kernel.instr_list)

    for i in range(num_instrs - 1):
        curr_instr = input_kernel.instr_list[i]

        # This is a jump instruction 
        if "dst_pc" in curr_instr.metadata:
            dst_codeblock = curr_instr.metadata["dst_pc"]
            dst_pc = input_kernel.code_blocks[dst_codeblock]
            edge_list.append((dst_pc, i))

            # This is an unconditional jump instruction 
            if "pred_reg" not in curr_instr.metadata:
                continue 

        edge_list.append((i + 1, i))

    G = nx.DiGraph(edge_list)
    idom = nx.immediate_dominators(G, num_instrs - 1) 

    output_kernel = deepcopy(input_kernel) 
    for i in range(num_instrs - 1):
        curr_instr = input_kernel.instr_list[i] 
        if "dst_pc" in curr_instr.metadata:
            output_kernel.instr_list[i].metadata["pdom"] = idom[i] 
            logger.debug(
                "Instruction: {}\nPost Dominator: {}".format(
                    curr_instr, input_kernel.instr_list[idom[i]])
            )

    return output_kernel 

import logging 

from copy import deepcopy 


def annotate_location(input_kernel, hw_config, logger=None):
    """This function annotates the location of registers and instructions. 

    Args:
        input_kernel: the kernel before the location annotation. 
        hw_config: the harddware configuration with the simulation information
            about whether to annotate the location and the location of shared 
            memory. 
        logger: a logger to log information for the purpose of debugging 

    Returns:
        output_kernel: the kernel with the location of registers and 
            instructions annotated. 
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

    output_kernel = deepcopy(input_kernel)
    for instr in output_kernel.instr_list:
        if "pred_reg" in instr.metadata:
            pred_reg = instr.metadata["pred_reg"].op_str
            output_kernel.reg_loc[pred_reg] = "U"
    
        for src_op in instr.src_operands:
            if src_op.isnormalreg():
                output_kernel.reg_loc[src_op.op_str] = "U"

        for dst_op in instr.dst_operands:
            if dst_op.isnormalreg():
                output_kernel.reg_loc[dst_op.op_str] = "U"

    if not hw_config["annotate_location"]:
        return output_kernel 

    _reg_location_analysis(output_kernel, hw_config, logger) 
    _instr_location_annotation(output_kernel, logger)

    return output_kernel 


def _infer_loc(dst_loc, old_loc):
    if dst_loc == "U":
        return old_loc 
    if old_loc == "U":
        return dst_loc 
    if dst_loc == old_loc:
        return dst_loc 
    return "B"


def _reg_location_analysis(input_kernel, hw_config, logger):
    num_instrs = len(input_kernel.instr_list)
    smem_loc = "N" if hw_config["default_smem_loc_is_near_bank"] else "F"

    for i in range(num_instrs):
        instr = input_kernel.instr_list[i]
        if "dst_pc" in instr.metadata:
            if "pred_reg" in instr.metadata:
                pred_reg = instr.metadata["pred_reg"].op_str 
                input_kernel.reg_loc[pred_reg] = "F"
                input_kernel.instr_list[i].metadata["location"] = "fb"

        elif instr.opcode.startswith("ld.global"):
            src_op = instr.src_operands[0]
            input_kernel.reg_loc[src_op.op_str] = "F"

            dst_op = instr.dst_operands[0] 
            input_kernel.reg_loc[dst_op.op_str] = "N"
            input_kernel.instr_list[i].metadata["location"] = "fb"

        elif instr.opcode.startswith("st.global"):
            addr_op = instr.src_operands[0]
            input_kernel.reg_loc[addr_op.op_str] = "F"

            val_op = instr.src_operands[1]
            input_kernel.reg_loc[val_op.op_str] = "N"
            input_kernel.instr_list[i].metadata["location"] = "fb"

        elif instr.opcode.startswith("ld.shared"):
            src_op = instr.src_operands[0]
            if src_op.isnormalreg():
                input_kernel.reg_loc[src_op.op_str] = smem_loc 

            dst_op = instr.dst_operands[0]
            assert dst_op.isnormalreg() 
            input_kernel.reg_loc[dst_op.op_str] = smem_loc 
            input_kernel.instr_list[i].metadata["location"] = "fb"

        elif (instr.opcode.startswith("st.shared") 
              or instr.opcode.startswith("atom.shared")):
            for src_op in instr.src_operands:
                if src_op.isnormalreg():
                    input_kernel.reg_loc[src_op.op_str] = smem_loc 

            input_kernel.instr_list[i].metadata["location"] = "fb"

    after_reg_loc = deepcopy(input_kernel.reg_loc)
    before_reg_loc = {}
    while True:
        before_reg_loc = deepcopy(after_reg_loc)

        for i in reversed(range(num_instrs)):
            instr = input_kernel.instr_list[i]
            if "location" in instr.metadata or len(instr.dst_operands) == 0:
                continue 
            assert len(instr.dst_operands) == 1

            dst_loc = after_reg_loc[instr.dst_operands[0].op_str]
            if "pred_reg" in instr.metadata:
                pred_reg = instr.metadata["pred_reg"].op_str 
                new_loc = _infer_loc(dst_loc, after_reg_loc[pred_reg])
                after_reg_loc[pred_reg] = new_loc 

            for src_op in instr.src_operands:
                if src_op.isnormalreg():
                    new_loc = _infer_loc(dst_loc, after_reg_loc[src_op.op_str])
                    after_reg_loc[src_op.op_str] = new_loc 

        if before_reg_loc == after_reg_loc:
            break 

    input_kernel.reg_loc = deepcopy(after_reg_loc)

    set_name = {"N": "Near Bank", "F": "Far Bank", "B": "Both Locations",
                "U": "Unknown"}
    for loc_label in set_name.keys():
        loc_name = set_name[loc_label]
        logger.debug("{} Registers:".format(loc_name))

        reg_list = []
        for reg_name in input_kernel.reg_loc.keys():
            if input_kernel.reg_loc[reg_name] == loc_label:
                reg_list.append(reg_name)

        logger.debug(reg_list)

    return


def _instr_location_annotation(input_kernel, logger):
    num_instrs = len(input_kernel.instr_list)
    for i in range(num_instrs):
        instr = input_kernel.instr_list[i]
        if "location" in instr.metadata or len(instr.dst_operands) == 0:
            continue 
        assert len(instr.dst_operands) == 1

        dst_op = instr.dst_operands[0]
        if input_kernel.reg_loc[dst_op.op_str] == "F":
            input_kernel.instr_list[i].metadata["location"] = "fb"
        elif input_kernel.reg_loc[dst_op.op_str] == "N":
            input_kernel.instr_list[i].metadata["location"] = "nb"

    return 

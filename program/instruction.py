import struct 


def parse_reg_str(reg_str):
    """Get the register prefix and index of a register"""
    reg_prefix = ""
    reg_index = None 
    pos = len(reg_str)
    for i in range(len(reg_str)):
        if reg_str[i].isdigit():
            pos = i 
            break 
    reg_prefix = reg_str[:pos]

    if pos != len(reg_str):
        reg_index = int(reg_str[pos:])

    return reg_prefix, reg_index 


class Operand:

    def __init__(self, raw_str):
        self._offset = 0
        self._type = None 
        self.op_str = ""
        self.raw_str = raw_str 

        if raw_str[0] == "_":
            self._type = "param"
        elif raw_str[0] == "%":
            self._type = "reg" 
        elif (raw_str[0].isdigit() or raw_str[0] == "-"):
            self._type = "imm"
        else:
            raise NotImplementedError(
                "Unrecognized operand str: {}".format(raw_str)
            )

        op_list = raw_str.split("+")
        assert len(op_list) <= 2, "Incorrect operand format"

        self.op_str = op_list[0]
        if len(op_list) == 2:
            self._offset = int(op_list[1])

        self.reg_prefix = ""
        self.reg_index = None 

        if self.isreg():
            self.reg_prefix, self.reg_index = parse_reg_str(self.op_str)

        return

    def __str__(self):
        return self.op_str 

    def update_index(self, new_index): 
        """Update the register index"""
        assert self.isnormalreg() 
        self.reg_index = new_index 
        self.op_str = "{}{}".format(self.reg_prefix, self.reg_index)
        return 

    def isreg(self):
        """Check whether the operand is a register name 
        """
        if self._type == "reg":
            return True 
        return False 

    def isnormalreg(self):
        """Check wehther the operand is a normal register
        """
        if self.isreg() and (not (self.reg_index is None)):
            return True 
        return False 

    def isspecialreg(self):
        """Check whether the operand is a special register holding block ID or
        thread ID information 
        """
        if self.isreg() and (self.reg_index is None):
            return True 
        return False 

    def isparam(self):
        """Check whether the operand is a parameter name 
        """
        if self._type == "param":
            return True 
        return False 

    def isimmvalue(self):
        """Check whether the operand is an immediate value 
        """
        if self._type == "imm":
            return True 
        return False 

    def eval(self, ref_value=None):
        """Return the value providing the value of symbols in this operand 
        """
        if self.isimmvalue():
            if self.op_str.startswith("0f"):
                val_tuple = struct.unpack(
                    ">f", bytearray.fromhex(self.op_str[2:])
                )
                return val_tuple[0] 
            return int(self.op_str)

        if self._offset == 0:
            return ref_value  
        
        return (ref_value + int(self._offset))


class Instr:

    def __init__(self, instr_str):
        self.instr_str = instr_str 
        self.opcode = None
        self.src_operands = []
        self.dst_operands = [] 
        self.metadata = {}
        # precision: 8/16/32/64
        self.precision = None
        # data_type: singed/unsigned/float/bit(untyped)/predicate
        self.data_type = None
        # latency in terms of cycles
        self.latency = 0

        self._analysis(instr_str)
        return 

    def _analysis(self, instr_str):
        """This function parses the instruction into different components. 
        It will extract following information:

        1. Predicate registers and conditions, such as @!%p1
        2. Operation code, including control, compute, and memory operations 
        3. Source and destination operands 
        """

        new_instr_str = instr_str.replace("\t", " ")
        op_list = []
        for x in new_instr_str.split(" "):
            op = x.strip(" [],;\n")            
            if len(op) > 0:
                op_list.append(op)

        if op_list[0][0] == "@":
            instr_op_list = op_list[1:]
            if op_list[0][1] == "!":
                self.metadata["pred_cond"] = False 
                self.metadata["pred_reg"] = Operand(op_list[0][2:])
            else:
                self.metadata["pred_cond"] = True 
                self.metadata["pred_reg"] = Operand(op_list[0][1:])  
        else:
            instr_op_list = op_list

        self.opcode = instr_op_list[0]
        if any(c.isdigit() for c in self.opcode.split(".")[-1]):
            if self.opcode.split(".")[-1][0] == "s":
                self.data_type = "signed"
            elif self.opcode.split(".")[-1][0] == "u":
                self.data_type = "unsigned"
            elif self.opcode.split(".")[-1][0] == "f":
                self.data_type = "float"
            elif self.opcode.split(".")[-1][0] == "b":
                self.data_type = "bit"
            else:
                raise NotImplementedError("Unkown data type: {}"
                                          .format(self.opcode
                                                  .split(".")[-1][0]))
            assert self.opcode.split(".")[-1][1:].isdigit()
            self.precision = int(self.opcode.split(".")[-1][1:])
        if self.opcode.split(".")[-1] == "pred":
            self.data_type = "predicate"
        fill_oprands = False 

        if self.opcode.startswith("ret"):
            pass 

        elif (self.opcode.startswith("bra")):
            self.metadata["dst_pc"] = instr_op_list[1]

        elif (self.opcode.startswith("bar.sync")):
            self.metadata["bar_id"] = int(instr_op_list[1])
            if len(instr_op_list) == 2:
                self.metadata["num_threads"] = None 
            elif len(instr_op_list) == 3:
                self.metadata["num_threads"] = int(instr_op_list[2])
            else:
                raise NotImplementedError(
                    "Unrecognized {} instruction: {}".format(
                        self.opcode, self.instr_str 
                    )
                )

        elif (self.opcode.startswith("st.")):
            assert len(instr_op_list) == 3, (
                "Unrecognized {} instruction: {}".format(
                    self.opcode, self.instr_str 
                )
            )

            self.src_operands.append(Operand(instr_op_list[1]))
            self.src_operands.append(Operand(instr_op_list[2]))

        elif (self.opcode.startswith("ld.") 
                or self.opcode.startswith("cvta.")
                or self.opcode.startswith("cvt.")
                or self.opcode.startswith("mov.")):
            assert len(instr_op_list) == 3, (
                "Unrecognized {} instruction: {}".format(
                    self.opcode, self.instr_str
                )
            )
            fill_oprands = True 

        elif (self.opcode.startswith("add.")
                or self.opcode.startswith("sub.")
                or self.opcode.startswith("mul.") 
                or self.opcode.startswith("setp.")  
                or self.opcode.startswith("shl.")
                or self.opcode.startswith("shr.")
                or self.opcode.startswith("and.")
                or self.opcode.startswith("max.")
                or self.opcode.startswith("min.")
                or self.opcode.startswith("div.")):
            assert len(instr_op_list) == 4, (
                "Unrecognized {} instruction: {}".format(
                    self.opcode, self.instr_str
                )
            )
            fill_oprands = True 

        elif (self.opcode.startswith("mad.")
                or self.opcode.startswith("fma.")
                or self.opcode.startswith("selp.")):
            assert len(instr_op_list) == 5, (
                "Unrecognized {} instruction: {}".format(
                    self.opcode, self.instr_str
                )
            )
            fill_oprands = True 

        elif (self.opcode.startswith("atom.shared.add.")):
            assert len(instr_op_list) == 4, (
                "Unrecognized {} instruction: {}".format(
                    self.opcode, self.instr_str 
                )
            )
            self.src_operands.append(Operand(instr_op_list[2]))
            self.src_operands.append(Operand(instr_op_list[3]))

        else:
            raise NotImplementedError(
                "Unsupported opcode: {}".format(self.opcode)
            )

        if fill_oprands:
            self.dst_operands.append(Operand(instr_op_list[1]))
            for src_op in instr_op_list[2:]:
                self.src_operands.append(Operand(src_op))

        return 

    def __str__(self):
        return "instr: {instr_str} op_code: {op_code}, " \
            "src_operands: {src_operands}, " \
            "dst_operands: {dst_operands} " \
            "metadadta: {metadata} ".format(
                instr_str=self.instr_str,
                op_code=self.opcode,
                src_operands=[x.op_str for x in self.src_operands],
                dst_operands=[x.op_str for x in self.dst_operands],
                metadata="{" + " ".join(
                    [str(key) + ": " + str(value) 
                        for key, value in self.metadata.items()]
                ) + "}"
            )

    def trace_str(self):
        return "{} dst={} src={} meta={}" \
            .format(
                self.opcode,
                [x.op_str for x in self.dst_operands],
                [x.op_str for x in self.src_operands],
                "{" + " ".join(
                    [str(key) + ": " + str(value)
                        for key, value in self.metadata.items()]
                ) + "}"
            )
    
    def set_latency(self, config):
        latency_set = config["alu_instr"] | config["sfu_instr"]
        opcode = self.opcode.split(".")[0]
        if opcode not in latency_set:
            return
        
        if self.opcode.startswith("add"):
            if self.data_type in ["signed", "unsigned"]:
                self.latency = config["alu_t_int_add"]
            elif self.data_type == "float":
                if self.precision == 16:
                    self.latency = config["alu_t_fp16_add"]
                elif self.precision == 32:
                    self.latency = config["alu_t_fp32_add"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_add"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("sub"):
            if self.data_type in ["signed", "unsigned"]:
                self.latency = config["alu_t_int_sub"]
            elif self.data_type == "float":
                if self.precision == 16:
                    self.latency = config["alu_t_fp16_sub"]
                elif self.precision == 32:
                    self.latency = config["alu_t_fp32_sub"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_sub"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("min"):
            if self.data_type in ["signed", "unsigned"]:
                self.latency = config["alu_t_int_min"]
            elif self.data_type == "float":
                if self.precision == 32:
                    self.latency = config["alu_t_fp32_min"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_min"]
                else:
                    assert False
            else:
                False
        elif self.opcode.startswith("max"):
            if self.data_type in ["signed", "unsigned"]:
                self.latency = config["alu_t_int_max"]
            elif self.data_type == "float":
                if self.precision == 32:
                    self.latency = config["alu_t_fp32_max"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_max"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("mul"):
            if self.data_type in ["signed", "unsigned"]:
                self.latency = config["alu_t_int_mul"]
            elif self.data_type == "float":
                if self.precision == 16:
                    self.latency = config["alu_t_fp16_mul"]
                elif self.precision == 32:
                    self.latency = config["alu_t_fp32_mul"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_mul"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("mad"):
            if self.data_type in ["signed", "unsigned"]:
                self.latency = config["alu_t_int_mad"]
            elif self.data_type == "float":
                if self.precision == 32:
                    self.latency = config["alu_t_fp32_mad"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_mul"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("fma"):
            if self.data_type == "float":
                if self.precision == 16:
                    self.latency = config["alu_t_fp16_fma"]
                elif self.precision == 32:
                    self.latency = config["alu_t_fp32_fma"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_fma"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("div"):
            if self.data_type == "signed":
                self.latency = config["alu_t_int_div_s"]
            elif self.data_type == "unsigned":
                self.latency = config["alu_t_int_div_u"]
            elif self.data_type == "float":
                if self.precision == 32:
                    self.latency = config["alu_t_fp32_div"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_div"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("rem"):
            if self.data_type == "signed":
                self.latency = config["alu_t_int_rem_s"]
            elif self.data_type == "unsigned":
                self.latency = config["alu_t_int_rem_u"]
            else:
                assert False
        elif self.opcode.startswith("abs"):
            if self.data_type == "signed":
                self.latency = config["alu_t_int_abs"]
            elif self.data_type == "float":
                if self.precision == 32:
                    self.latency = config["alu_t_fp32_abs"]
                elif self.precision == 64:
                    self.latency = config["alu_t_fp64_abs"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("and"):
            if self.data_type in ["bit", "predicate"]:
                self.latency = config["alu_t_logic_and"]
            else:
                assert False
        elif self.opcode.startswith("or"):
            if self.data_type in ["bit", "predicate"]:
                self.latency = config["alu_t_logic_or"]
            else:
                assert False
        elif self.opcode.startswith("not"):
            if self.data_type in ["bit", "predicate"]:
                self.latency = config["alu_t_logic_not"]
            else:
                assert False
        elif self.opcode.startswith("xor"):
            if self.data_type in ["bit", "predicate"]:
                self.latency = config["alu_t_logic_xor"]
            else:
                assert False
        elif self.opcode.startswith("cnot"):
            if self.data_type == "bit":
                self.latency = config["alu_t_logic_cnot"]
            else:
                assert False
        elif self.opcode.startswith("shl"):
            if self.data_type == "bit":
                self.latency = config["alu_t_logic_shl"]
            else:
                assert False
        elif self.opcode.startswith("shr"):
            if self.data_type in ["bit", "signed", "unsigned"]:
                self.latency = config["alu_t_logic_shr"]
            else:
                assert False
        elif self.opcode.startswith("setp"):
            if self.data_type in ["signed", "unsigned", "bit"]:
                if self.precision == 32:
                    self.latency = config["alu_t_int_setp"]
                else:
                    assert False
            elif self.data_type in ["float"]:
                if self.precision == 32:
                    self.latency = config["alu_t_fp32_setp"]
                else:
                    assert False
            else:
                assert False
        elif self.opcode.startswith("rcp"):
            self.latency = config["sfu_t_rcp"]
        elif self.opcode.startswith("sqrt"):
            self.latency = config["sfu_t_sqrt"]
        elif self.opcode.startswith("rsqrt"):
            self.latency = config["sfu_t_rsqrt"]
        elif self.opcode.startswith("sin"):
            self.latency = config["sfu_t_sin_cos"]
        elif self.opcode.startswith("cos"):
            self.latency = config["sfu_t_sin_cos"]
        elif self.opcode.startswith("lg2"):
            self.latency = config["sfu_t_lg2"]
        elif self.opcode.startswith("ex2"):
            self.latency = config["sfu_t_ex2"]
        elif self.opcode.startswith("mov"):
            self.latency = config["alu_t_mov"]
        elif self.opcode.startswith("cvt"):
            self.latency = config["alu_t_cvt"]
        elif self.opcode.startswith("selp"):
            if self.data_type in ["signed", "unsigned", "bit"]:
                if self.precision == 32:
                    self.latency = config["alu_t_int_selp"]
                else:
                    assert False
            elif self.data_type in ["float"]:
                if self.precision == 32:
                    self.latency = config["alu_t_fp32_selp"]
                else:
                    assert False
            else:
                assert False
        else:
            raise NotImplementedError("opcode not supported!: {}"
                                      .format(self.opcode))


import struct 


class InstrEntry:
    """This class contains the information of original instruction and the 
    data values of each source operands. It facilitates the implementation 
    of completing instruciton functionalities regardless of timing models. 
    """

    def __init__(self, log, config, instr, simt_mask, pc, 
                 subcore_id, warp_id):
        self.log = log
        self.config = config
        self.instr = instr
        self.simt_width = self.config["num_threads_per_warp"]
        self.simt_mask = simt_mask
        self.pc = pc
        self.subcore_id = subcore_id
        # pg_id field is only valid when executed in PG (near-bank)
        self.pg_id = None
        # physical warp_id
        # used to index into warp table
        self.warp_id = warp_id
        self.pred_mask = None
        self.pred_buffer = None
        self.src_values = []
        self.dst_values = []
        self.processed = False
        # the following fields are determined in 
        # the instr offload engine
        # op_str --> True (near-bank) / False (far-bank)
        self.src_loc_is_nb = {}
        self.dst_loc_is_nb = {}
        # instruction location
        # "nb": directly executed in near-bank
        # "fb": directly executed in far-bank
        self.instr_location = None
        # used for instruction tracing
        self.slot_id = None
        self.last_trace_cyc = 0
        self.local_access_cyc = None
        self.remote_access_cyc = None
        return

    def process_operands(self):
        self._decode_src_operands() 
        self._comp_result() 
        self._encode_dst_operands() 

    def _get_struct_code(self, reg_prefix):
        if reg_prefix == "%rd":
            return "{}q".format(self.simt_width)
        elif reg_prefix == "%r":
            return "{}i".format(self.simt_width)
        elif reg_prefix == "%f":
            return "{}f".format(self.simt_width)
        elif reg_prefix == "%p":
            return "{}B".format(self.simt_width)
        else:
            raise NotImplementedError(
                "Unkown register prefix: {}".format(reg_prefix)
            )

    def _bytes_to_mask(self, bytes_buffer):
        values = struct.unpack(
            "{}B".format(self.simt_width), bytes_buffer
        )
        mask = 0
        for i in range(self.simt_width):
            if values[i] != 0:
                mask = mask | (1 << i)
        return mask

    def _decode_src_operands(self):
        # process predicate register and update simt_mask
        if "pred_reg" in self.instr.metadata:
            assert self.pred_buffer is not None
            self.pred_mask = self._bytes_to_mask(self.pred_buffer)
            if not self.instr.metadata["pred_cond"]:
                all_1s = "1" * self.simt_width
                self.pred_mask = self.pred_mask ^ int(all_1s, 2)

            pred_reg_value_str = "{0:{fill}32b}"\
                .format(self.pred_mask, fill=0)
            self.simt_mask = self.simt_mask & self.pred_mask
            self.log.debug(
                "predicative register value"
                "{pred_reg_value} for the instruction with"
                "pc={pc} at entry {entry_id}".format(
                    pred_reg_value=pred_reg_value_str,
                    pc=self.pc, 
                    entry_id=self.warp_id
                )
            )

        # decode source operands
        self._decoded_src_values = []
        for i in range(len(self.src_values)):
            src_val = self.src_values[i]
            src_op = self.instr.src_operands[i] 
            if isinstance(src_val, int) or isinstance(src_val, float):
                self._decoded_src_values.append([src_val] * self.simt_width) 
            elif isinstance(src_val, list):
                assert len(src_val) == self.simt_width, "The list of values" \
                    " should have the length same as the SIMT length"
                self._decoded_src_values.append(src_val)
            elif isinstance(src_val, bytearray):
                decoded_value = struct.unpack(
                    self._get_struct_code(src_op.reg_prefix), src_val
                )
                self._decoded_src_values.append(
                    [src_op.eval(x) for x in decoded_value]
                )
            else:
                raise NotImplementedError(
                    "Unknown format of the source operand: "
                    "{src_val}".format(src_val=src_val)
                )
        return 

    def _comp_result(self):
        self._decoded_dst_values = []

        if self.instr.opcode.startswith("ld.param"):
            self._decoded_dst_values.append(self._decoded_src_values[0])
        elif self.instr.opcode.startswith("cvta.to.global"):
            self._decoded_dst_values.append(self._decoded_src_values[0])
        elif self.instr.opcode.startswith("mov"):
            self._decoded_dst_values.append(self._decoded_src_values[0])
        elif self.instr.opcode.startswith("cvt"):
            reg_prefix = self.instr.dst_operands[0].reg_prefix 

            if reg_prefix == "%r" or reg_prefix == "%rd":
                new_values = [int(x) for x in self._decoded_src_values[0]]
            elif reg_prefix == "%f":
                new_values = [float(x) for x in self._decoded_src_values[0]]
            else:
                raise NotImplementedError(
                    "Unknown reg prefix: {}".format(reg_prefix))

            self._decoded_dst_values.append(new_values)
        elif self.instr.opcode.startswith("mad"):
            result_val = []
            bitwidth = int(self.instr.opcode.split(".")[-1][1:])
            pos = self.instr.opcode.split(".")[1]

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                c = self._decoded_src_values[2][i]

                t = a * b
                if pos == "hi":
                    d = (t >> bitwidth) + c
                elif (pos == "wide" or pos == "lo"):
                    d = t + c
                else:
                    raise NotImplementedError("Unknown mode: {}".format(pos))

                result_val.append(d)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("fma"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                c = self._decoded_src_values[2][i]

                d = a * b + c
                result_val.append(d)
        
            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("mul"):
            result_val = []
            bitwidth = int(self.instr.opcode.split(".")[-1][1:])
            pos = self.instr.opcode.split(".")[1]

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]

                t = a * b
                if pos == "hi":
                    d = (t >> bitwidth) 
                elif pos == "lo":
                    d = (t & ((1 << bitwidth) - 1))
                elif (pos == "wide" or pos == "f32"):
                    d = t
                else:
                    raise NotImplementedError("Unkown mode: {}".format(pos))

                result_val.append(d)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("div"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                d = a / b 
                result_val.append(d)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("setp"):
            assert len(self.instr.opcode.split(".")) == 3, "The boolean " \
                "with another predicative register is not supported now" 
            result_val = []
            compare_op = self.instr.opcode.split(".")[1]
            if compare_op == "ge":
                compare_func = lambda x, y: x >= y  # noqa: E731
            elif compare_op == "lt":
                compare_func = lambda x, y: x < y  # noqa: E731
            elif compare_op == "le":
                compare_func = lambda x, y: x <= y  # noqa: E731 
            elif compare_op == "gt":
                compare_func = lambda x, y: x > y  # noqa: E731
            elif compare_op == "ne":
                compare_func = lambda x, y: x != y  # noqa: E731
            elif compare_op == "eq":
                compare_func = lambda x, y: x == y  # noqa: E731 
            else:
                raise NotImplementedError(
                    "Unknown comparison op: {}".format(compare_op)
                )

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(1 if compare_func(a, b) else 0)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("selp"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                c = self._decoded_src_values[2][i]
                d = a if c else b
                result_val.append(d)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("add"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(a + b)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("sub"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(a - b)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("shl"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(a << b)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("shr"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(a >> b)
            
            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("and"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(a & b)

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("max"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(max(a, b))

            self._decoded_dst_values.append(result_val)
        elif self.instr.opcode.startswith("min"):
            result_val = []

            for i in range(self.simt_width):
                a = self._decoded_src_values[0][i]
                b = self._decoded_src_values[1][i]
                result_val.append(min(a, b))

            self._decoded_dst_values.append(result_val)
        elif "dst_pc" in self.instr.metadata:
            return 
        elif self.instr.opcode == "ret":
            return 
        elif self.instr.opcode.startswith("ld.global"):
            return 
        elif self.instr.opcode.startswith("st.global"):
            return
        elif self.instr.opcode.startswith("ld.shared"):
            return 
        elif self.instr.opcode.startswith("st.shared"):
            return
        elif self.instr.opcode.startswith("atom.shared"):
            return 
        elif self.instr.opcode.startswith("bar.sync"):
            return 
        else:
            raise NotImplementedError(
                "Unknown instruction opcode {} to compute the "
                "result".format(self.instr.opcode)
            )

        self.log.debug("Instr: {}".format(self.instr.instr_str))
        self.log.debug("src operands:")
        for i in range(len(self.instr.src_operands)):
            self.log.debug(
                "{}: {}".format(
                    self.instr.src_operands[i].raw_str, 
                    self._decoded_src_values[i]
                )
            )
        self.log.debug("dst operands:")
        for i in range(len(self.instr.dst_operands)):
            self.log.debug(
                "{}: {}".format(
                    self.instr.dst_operands[i].raw_str, 
                    self._decoded_dst_values[i]
                )
            )
        self.log.debug("")

        return 

    def _encode_dst_operands(self):
        for i in range(len(self._decoded_dst_values)):
            reg_prefix = self.instr.dst_operands[i].reg_prefix
            encoded_value = bytearray(
                struct.pack(
                    self._get_struct_code(reg_prefix), 
                    *self._decoded_dst_values[i]
                )
            )
            self.dst_values.append(encoded_value) 
        return 

    def _decode_dst_operands(self):
        for i in range(len(self.dst_values)):
            reg_prefix = self.instr.dst_operands[i].reg_prefix
            decoded_value = struct.unpack(
                self._get_struct_code(reg_prefix),
                self.dst_values[i]
            )
            self._decoded_dst_values.append(decoded_value)
        return

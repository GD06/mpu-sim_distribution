#!/usr/bin/env python3

def get_instr_energy(opcode, config):
    """Get instruction energy for ALU / SFU
    Args:
        opcode: operation code of the instruction
    Return:
        instr_energy: the energy of that instruction at the execution unit
    """
    if any(c.isdigit() for c in opcode.split(".")[-1]):
        if opcode.split(".")[-1][0] == "s":
            data_type = "signed"
        elif opcode.split(".")[-1][0] == "u":
            data_type = "unsigned"
        elif opcode.split(".")[-1][0] == "f":
            data_type = "float"
        elif opcode.split(".")[-1][0] == "b":
            data_type = "bit"
        else:
            raise NotImplementedError("Unkown data type: {}"
                                      .format(opcode))
    if opcode.split(".")[-1] == "pred":
        data_type = "predicate"
    assert opcode.split(".")[-1][1:].isdigit()
    precision = int(opcode.split(".")[-1][1:])
    if opcode.startswith("add"):
        if data_type in ["signed", "unsigned"]:
            instr_energy = config["alu_e_int_add"]
        elif data_type == "float":
            if precision == 16:
                instr_energy = config["alu_e_fp16_add"]
            elif precision == 32:
                instr_energy = config["alu_e_fp32_add"]
            elif precision == 64:
                instr_energy = config["alu_e_fp64_add"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("sub"):
        if data_type in ["signed", "unsigned"]:
            instr_energy = config["alu_e_int_sub"]
        elif data_type == "float":
            if precision == 16:
                instr_energy = config["alu_e_fp16_sub"]
            elif precision == 32:
                instr_energy = config["alu_e_fp32_sub"]
            elif precision == 64:
                instr_energy = config["alu_e_fp64_sub"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("min"):
        if data_type in ["signed", "unsigned"]:
            instr_energy = config["alu_e_int_min"]
        elif data_type == "float":
            if precision == 32:
                instr_energy = config["alu_e_fp32_min"]
            elif precision == 64:
                instr_energy = config["alu_e_fp64_min"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("max"):
        if data_type in ["signed", "unsigned"]:
            instr_energy = config["alu_e_int_max"]
        elif data_type == "float":
            if precision == 32:
                instr_energy = config["alu_e_fp32_max"]
            elif precision == 64:
                instr_energy = config["alu_e_fp64_max"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("mul"):
        if data_type in ["signed", "unsigned"]:
            instr_energy = config["alu_e_int_mul"]
        elif data_type == "float":
            if precision == 16:
                instr_energy = config["alu_e_fp16_mul"]
            elif precision == 32:
                instr_energy = config["alu_e_fp32_mul"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("mad"):
        if data_type in ["signed", "unsigned"]:
            instr_energy = config["alu_e_int_mad"]
        elif data_type == "float":
            if precision == 32:
                instr_energy = config["alu_e_fp32_mad"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("fma"):
        if data_type == "float":
            if precision == 32:
                instr_energy = config["alu_e_fp32_fma"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("div"):
        if data_type == "signed":
            instr_energy = config["alu_e_int_div_s"]
        elif data_type == "unsigned":
            instr_energy = config["alu_e_int_div_u"]
        elif data_type == "float":
            if precision == 32:
                instr_energy = config["alu_e_fp32_div"]
            elif precision == 64:
                instr_energy = config["alu_e_fp64_div"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("rem"):
        if data_type == "signed":
            instr_energy = config["alu_e_int_rem_s"]
        elif data_type == "unsigned":
            instr_energy = config["alu_e_int_rem_u"]
        else:
            assert False
    elif opcode.startswith("abs"):
        if data_type in ["signed", "unsigned"]:
            instr_energy = config["alu_e_int_abs"]
        else:
            assert False
    elif opcode.startswith("and"):
        if data_type in ["bit", "predicate"]:
            instr_energy = config["alu_e_logic_and"]
        else:
            assert False
    elif opcode.startswith("or"):
        if data_type in ["bit", "predicate"]:
            instr_energy = config["alu_e_logic_or"]
        else:
            assert False
    elif opcode.startswith("not"):
        if data_type in ["bit", "predicate"]:
            instr_energy = config["alu_e_logic_not"]
        else:
            assert False
    elif opcode.startswith("xor"):
        if data_type in ["bit", "predicate"]:
            instr_energy = config["alu_e_logic_xor"]
        else:
            assert False
    elif opcode.startswith("cnot"):
        if data_type == "bit":
            instr_energy = config["alu_e_logic_cnot"]
        else:
            assert False
    elif opcode.startswith("shl"):
        if data_type == "bit":
            instr_energy = config["alu_e_logic_shl"]
        else:
            assert False
    elif opcode.startswith("shr"):
        if data_type == "bit":
            instr_energy = config["alu_e_logic_shr"]
        else:
            assert False
    elif opcode.startswith("setp"):
        if data_type in ["signed", "unsigned", "bit"]:
            if precision == 32:
                instr_energy = config["alu_t_int_setp"]
            else:
                assert False
        elif data_type == "float":
            if precision == 32:
                instr_energy = config["alu_t_fp32_setp"]
            else:
                assert False
    elif opcode.startswith("mov"):
        instr_energy = config["alu_e_mov"]
    elif opcode.startswith("cvt"):
        instr_energy = config["alu_e_cvt"]
    elif opcode.startswith("selp"):
        if data_type in ["signed", "unsigned", "bit"]:
            if precision == 32:
                instr_energy = config["alu_t_int_selp"]
            else:
                assert False
        elif data_type == "float":
            if precision == 32:
                instr_energy = config["alu_t_fp32_selp"]
            else:
                assert False
        else:
            assert False
    elif opcode.startswith("rcp"):
        instr_energy = config["sfu_e_rcp"]
    elif opcode.startswith("sqrt"):
        instr_energy = config["sfu_e_sqrt"]
    elif opcode.startswith("rsqrt"):
        instr_energy = config["sfu_e_rsqrt"]
    elif opcode.startswith("sin"):
        instr_energy = config["sfu_e_sin_cos"]
    elif opcode.startswith("cos"):
        instr_energy = config["sfu_e_sin_cos"]
    elif opcode.startswith("lg2"):
        instr_energy = config["sfu_e_lg2"]
    elif opcode.startswith("ex2"):
        instr_energy = config["sfu_e_ex2"]
    else:
        raise NotImplementedError("opcode not supported!: {}"
                                  .format(opcode))
    # NOTE: currently the ALU/SFU energy unit is nJ
    return instr_energy


class SubcoreEnergy:

    def __init__(self, config, pt, hw, subcore_id):
        self.config = config
        self.pt = pt
        self.hw = hw
        self.subcore_id = subcore_id

        # initialize energy items for this module
        self.energy_item = {}
        # pipeline: fetch and decode energy
        self.energy_item["fetch_decode_energy"] = 0
        # pipeline: issue energy
        self.energy_item["scoreboard_energy"] = 0
        self.energy_item["reg_track_table_energy"] = 0
        # pipeline: execution energy
        self.energy_item["lsu_energy"] = 0
        self.energy_item["fb_alu_energy"] = 0
        self.energy_item["sfu_energy"] = 0
        # operand collector
        self.energy_item["fb_opc_energy"] = 0
        # register file energy
        self.energy_item["fb_rf_energy"] = 0
        # pipeline: writeback and commit energy
        self.energy_item["wb_commit_energy"] = 0

    def _update_fetch_decode_energy(self):
        self.energy_item["fetch_decode_energy"] = \
            self.pt["num_instr_fd"] * self.config["instr_fetch_decode_energy"]
        self.hw.energy_item["fetch_decode_energy"] += \
            self.energy_item["fetch_decode_energy"]

    def _update_writeback_commit_energy(self):
        self.energy_item["wb_commit_energy"] = \
            self.pt["num_instr_wbc"] * self.config["instr_wb_commit_energy"]
        self.hw.energy_item["wb_commit_energy"] += \
            self.energy_item["wb_commit_energy"]

    def _update_instr_offload_engine_energy(self):
        pass

    def _update_fb_operand_collector_energy(self):
        tr = self.pt["fb_opc"]
        self.energy_item["fb_opc_energy"] = \
            tr["num_read"] * self.config["opc_read_energy"] \
            + tr["num_write"] * self.config["opc_write_energy"]
        self.hw.energy_item["fb_opc_energy"] += \
            self.energy_item["fb_opc_energy"]
    
    def _update_dep_table_energy(self):
        tr = self.pt["dep_table"]
        self.energy_item["scoreboard_energy"] = \
            tr["num_read"] * self.config["dep_table_read_energy"] \
            + tr["num_write"] * self.config["dep_table_write_energy"]
        self.hw.energy_item["scoreboard_energy"] += \
            self.energy_item["scoreboard_energy"]

    def _update_reg_track_table_energy(self):
        tr = self.pt["reg_track_table"]
        self.energy_item["reg_track_table_energy"] = \
            tr["num_read"] * self.config["reg_track_table_read_energy"] \
            + tr["num_write"] * self.config["reg_track_table_write_energy"]
        self.hw.energy_item["reg_track_table_energy"] += \
            self.energy_item["reg_track_table_energy"]

    def _update_fb_reg_file_energy(self):
        tr = self.pt["fb_reg_file"]
        self.energy_item["fb_rf_energy"] = \
            tr["num_read"] * self.config["subcore_reg_file_bank_read_energy"] \
            + tr["num_write"] \
            * self.config["subcore_reg_file_bank_write_energy"]
        self.hw.energy_item["fb_rf_energy"] += \
            self.energy_item["fb_rf_energy"]

    def _update_fb_alu_energy(self):
        tr = self.pt["fb_alu"]
        for opcode in tr.keys():
            num_op = tr[opcode]
            instr_energy = get_instr_energy(opcode, self.config)
            self.energy_item["fb_alu_energy"] += \
                num_op * instr_energy
            self.hw.energy_item["fb_alu_energy"] += \
                num_op * instr_energy

    def _update_sfu_energy(self):
        tr = self.pt["sfu"]
        for opcode in tr.keys():
            num_op = tr[opcode]
            instr_energy = get_instr_energy(opcode, self.config)
            self.energy_item["sfu_energy"] += \
                num_op * instr_energy
            self.hw.energy_item["sfu_energy"] += \
                num_op * instr_energy

    def _update_lsu_energy(self):
        tr = self.pt["lsu"]
        self.energy_item["lsu_energy"] = \
            tr["num_prt_read"] * self.config["lsu_prt_read_energy"] \
            + tr["num_prt_write"] * self.config["lsu_prt_write_energy"]
        self.hw.energy_item["lsu_energy"] += \
            self.energy_item["lsu_energy"]

    def get_energy_metrics(self):
        """Gets a dictionary of energy metrics. """
        # Collect the energy metrics of this hardware module.
        self._update_fetch_decode_energy()
        self._update_writeback_commit_energy()
        self._update_instr_offload_engine_energy()
        self._update_fb_operand_collector_energy()
        self._update_dep_table_energy()
        self._update_reg_track_table_energy()
        self._update_fb_reg_file_energy()
        self._update_fb_alu_energy()
        self._update_sfu_energy()
        self._update_lsu_energy()
        energy_metrics = self.energy_item
        energy_metrics["total_energy"] = 0
        for key in energy_metrics.keys():
            energy_metrics["total_energy"] += energy_metrics[key]

        return {"subcore_{}".format(self.subcore_id): energy_metrics}

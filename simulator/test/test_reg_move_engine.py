#!/usr/bin/env python3 

import unittest 
import os
import tempfile
from copy import deepcopy

import program.prog_api as prog_api 
import config.config_api as config_api 
import simulator.sim_api as sim_api 
from simulator.subcore_table import WarpInfoTableEntry


class TestRegMoveEngine(unittest.TestCase): 

    def setUp(self):
        # load hardware configuration
        self.config = config_api.load_hardware_config()

        # create a temporary ptx file
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)
        # Print out a simple kernel with registers allocated and an instruction
        # NOTE: this kernel is for testing only
        with open(self.ptx_file, "w") as f:
            print(".visible .entry _Z9Kernel(", file=f)
            print("\t .param .u32 _Z9_param_0", file=f)
            print(")", file=f)
            print("{", file=f)
            print("\t .reg .pred\t %p<10>;", file=f)
            print("\t .reg .f32\t %f<31>;", file=f)
            print("\t @%p1 add.f32\t %f4 %f3 %f2;", file=f)
            print("}", file=f)
        kernel_list = prog_api.load_kernel(self.ptx_file)
        self.assertEqual(len(kernel_list), 1)
        self.kernel = kernel_list[0]
        self.assertEqual(len(self.kernel.arg_list), 1)
        self.assertEqual(len(self.kernel.instr_list), 1)
        self.kernel.compute_resource_usage(
            data_path_unit_size=self.config["data_path_unit_size"],
            num_threads_per_warp=self.config["num_threads_per_warp"],
            num_pe=self.config["num_pe"]
        )

        # extract the instruction
        self.instr = self.kernel.instr_list[0]

        # initialize hardware
        self.hardware = sim_api.init_hardware(self.config)
        self.env = self.hardware.env

        # NOTE: these indices are for testing only
        self.proc_id_x = 0
        self.proc_id_y = 0
        self.core_id_x = 0
        self.core_id_y = 0
        self.subcore_id = 0
        self.pg_id = 0
        self.tidz = 0
        self.tidy = 0
        self.tidx = 0
        self.block_id = 0
        self.warp_id = 0

        # get a reference to a core
        self.core = self.hardware\
            .processor_array[(self.proc_id_x, self.proc_id_y)]\
            .core_array[(self.core_id_x, self.core_id_y)]
        # get a reference to a subcore
        self.subcore = self.core.subcore_array[self.subcore_id]
        # get a reference to a pg
        self.pg = self.core.pg_array[self.pg_id]

        return 

    def tearDown(self):
        os.remove(self.ptx_file)

    def _preset_state(self):
        # pre-set necessary arch status to perform simulation
        # NOTE: for testing only
        subcore_reg_base_addr = deepcopy(
            self.subcore.reg_base_ptr
        )
        pg_reg_base_addr = deepcopy(
            self.pg.reg_base_ptr
        )
        smem_base_addr = deepcopy(self.core.shared_memory_ptr)
        # pre-set warp info table
        warp_info_table_entry = WarpInfoTableEntry(
            thread_id=(self.tidz, self.tidy, self.tidx),
            block_id=self.block_id,
            pg_id=self.pg_id,
            subcore_reg_base_addr=subcore_reg_base_addr,
            pg_reg_base_addr=pg_reg_base_addr,
            smem_base_addr=smem_base_addr,
            prog_reg_offset=self.kernel.reg_offset,
            prog_reg_size=self.kernel.reg_size,
            prog_smem_offset=self.kernel.smem_offset,
            prog_length=len(self.kernel.instr_list)
        )
        self.subcore.warp_info_table.entry[self.warp_id] = \
            warp_info_table_entry
        # pre-set register tracking table
        src_ops = []
        dst_ops = []
        if "pred_reg" in self.instr.metadata:
            src_ops.append(self.instr.metadata["pred_reg"])
        for each_op in self.instr.src_operands:
            if each_op.isreg():
                src_ops.append(each_op)
        for each_op in self.instr.dst_operands:
            if each_op.isreg():
                dst_ops.append(each_op)
        for each_op in src_ops:
            reg_addr, reg_size = self.subcore.get_subcore_reg_addr(
                each_op.reg_prefix,
                each_op.reg_index,
                self.warp_id
            )
            # Update register tracking table
            self.subcore.reg_track_table\
                .entry[self.warp_id]\
                .write_update(
                    op_str=each_op.op_str,
                    reg_file_type="far-bank"
                )

        assert self.instr.opcode.split(".")[0] in self.config["alu_instr"]
        self.instr.set_latency(self.config)

        # set source operand
        self.operand = self.instr.src_operands[0]
        self.src_reg_addr, self.src_reg_size = \
            self.subcore.get_subcore_reg_addr(
                self.operand.reg_prefix,
                self.operand.reg_index,
                self.warp_id
            )
        # write random number into source register file
        self.subcore.reg_file.array[self.src_reg_addr: 
                                    self.src_reg_addr + self.src_reg_size] = \
            bytearray(os.urandom(self.src_reg_size))
        return

    def test_reg_move_subcore_to_pg(self):
        self._preset_state()
        self.env.process(
            self.subcore.instr_offload_engine._move_reg_req(
                operand=self.operand,
                warp_id=self.warp_id,
                is_upstream=True
            )
        )
        self.env.run()
        # get destination register file info
        self.dst_reg_addr, self.dst_reg_size = self.pg.get_pg_reg_addr(
            self.operand.reg_prefix,
            self.operand.reg_index,
            self.subcore_id,
            self.warp_id
        )
        # check dst and src reg data is the same
        src_data = self.subcore.reg_file \
            .array[self.src_reg_addr: self.src_reg_addr + self.src_reg_size]
        dst_data = self.pg.reg_file \
            .array[self.dst_reg_addr: self.dst_reg_addr + self.dst_reg_size]
        self.assertEqual(len(src_data), len(dst_data))
        for i in range(len(src_data)):
            self.assertEqual(src_data[i], dst_data[i])


if __name__ == "__main__":
    unittest.main() 

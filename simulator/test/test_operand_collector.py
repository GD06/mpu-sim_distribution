#!/usr/bin/env python3

import unittest
import logging
import simpy
import tempfile
import os

import program.prog_api as prog_api
import config.config_api as config_api

from simulator.register_file import RegisterFile
from simulator.register_file_utility import RegFileOperandIOInterface, \
    OperandWriteReq
from simulator.operand_collector import OperandCollector
from simulator.alu import ArithmeticLogicUnit
from simulator.subcore_table import DepTable, RegTrackTable
from simulator.instr_instance import InstrEntry


class FakeSubCore():
    """This class is only for testing purpose.
    """
    def __init__(self, env, config, log, clock_unit, reg_file, 
                 rf_io_interface, warp_id):
        self.env = env
        self.config = config
        self.log = log
        self.clock_unit = clock_unit
        self.alignment = self.config["data_path_unit_size"] \
            * self.config["num_threads_per_warp"]
        self.reg_file = reg_file
        self.rf_io_interface = rf_io_interface
        self.warp_id = warp_id
        self.writeback_buffer = simpy.Store(
            self.env, capacity=self.config["subcore_writeback_buffer_size"]
        )

        self.reset()
        self.env.process(self._commit())

    def _append_trace_event_dur(self, tid, name, ts, dur, cat="", args={}):
        # NOTE: this is a dummmy function for testing only
        pass

    def _commit(self):
        while True:
            instr_entry = yield self.writeback_buffer.get()
            instr = instr_entry.instr
            simt_mask = instr_entry.simt_mask
            
            for i in range(len(instr.dst_operands)):
                dst_op = instr.dst_operands[i]
                reg_addr, reg_size = self.get_subcore_reg_addr(
                    dst_op.reg_prefix, dst_op.reg_index, instr_entry.warp_id
                )
                if simt_mask > 0:
                    # Write results back to register file
                    operand_write_req = OperandWriteReq(
                        base_reg_addr=reg_addr,
                        total_reg_size=reg_size,
                        simt_mask=simt_mask,
                        data=instr_entry.dst_values[i]
                    )
                    # assume using the fisr queue
                    yield self.rf_io_interface\
                        .write_req_queue[0].put(operand_write_req)

                    _ = yield self.rf_io_interface.write_resp_queue.get(
                        lambda x: (x.base_reg_addr == reg_addr
                                   and x.total_reg_size == reg_size)
                    )
                
                # Release the dependency in the dependency table
                self.dep_table_exe.entry[self.warp_id]\
                    .decrease_write(dst_op.op_str)
                # Update register tracking table
                self.reg_track_table.entry[self.warp_id].write_update(
                    op_str=dst_op.op_str,
                    reg_file_type="far-bank"
                )

            self.instr_complete[instr_entry.pc] = True

    def get_subcore_reg_addr(self, reg_prefix, reg_index, warp_id):
        # for testing
        # here we don't distinguish register prefix..
        assert warp_id == self.warp_id
        reg_addr = self.alignment * reg_index
        if reg_prefix == "%f":
            reg_size = 128
        elif reg_prefix == "%p":
            reg_size = 32
        else:
            assert False
        return (reg_addr, reg_size)

    def get_special_reg_value(self, reg_name, warp_id):
        # for testing
        assert warp_id == self.warp_id
        return 0

    def get_param_value(self, param_name, warp_id):
        # for testing
        assert warp_id == self.warp_id
        return 0

    def reset(self):
        self.dep_table_exe = DepTable(config=self.config, log=self.log)
        self.reg_track_table = RegTrackTable(
            config=self.config,
            log=self.log,
            reg_file=self.reg_file
        )
        self.instr_complete = {}


class TestOperandCollector(unittest.TestCase):

    def setUp(self):
        # create a temporary ptx file
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.proj_dir = os.path.dirname(os.path.dirname(self.curr_dir))
        _, self.ptx_file = tempfile.mkstemp(suffix=".ptx", dir=self.curr_dir)
        
        # load hardware configuration
        self.config = config_api.load_hardware_config()
        # current allocation requires certain register file address mapping
        self.assertEqual(self.config["reg_file_addr_map"], 
                         "reg_file_addr_map_1")
        # setup required environment components
        self.env = simpy.Environment()
        self.core_clock_unit = (self.config["sim_clock_freq"]
                                // self.config["core_clock_freq"])
        # setup logger
        logging_level = logging.ERROR
        logger = logging.getLogger(__name__)
        logger.setLevel(logging_level)
        ch = logging.StreamHandler()
        ch.setLevel(logging_level)
        logger.addHandler(ch)
        self.log = logger

        # initialize register file
        self.fb_reg_file = RegisterFile(
            env=self.env,
            log=logger,
            config=self.config,
            clock_unit=self.core_clock_unit,
            reg_file_type="far-bank"
        )

        self.rf_io_interface = RegFileOperandIOInterface(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.core_clock_unit,
            reg_file=self.fb_reg_file,
            interface_type="far-bank"
        )

        # assume a warp_id
        warp_id = 0
        # a fake subcore for testing
        self.subcore = FakeSubCore(
            env=self.env,
            config=self.config,
            log=self.log,
            clock_unit=self.core_clock_unit,
            reg_file=self.fb_reg_file,
            rf_io_interface=self.rf_io_interface,
            warp_id=warp_id
        )

        # a fake execution unit for testing
        self.alu = ArithmeticLogicUnit(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.core_clock_unit,
            backend=self.subcore,
            alu_type="far-bank"
        )
        self.opc_alu = OperandCollector(
            env=self.env,
            log=self.log,
            config=self.config,
            clock_unit=self.core_clock_unit,
            backend=self.subcore,
            regfile_io_interface=self.rf_io_interface,
            execution_unit=self.alu,
            opc_type="fb_alu"
        )
        
        return
    
    def tearDown(self):
        os.remove(self.ptx_file)
    
    def _set_operand_location(self, instr_entry):
        is_nb = False
        if "pred_reg" in instr_entry.instr.metadata:
            pred_reg = instr_entry.instr.metadata["pred_reg"]
            instr_entry.src_loc_is_nb[pred_reg.op_str] = is_nb
        for src_op in instr_entry.instr.src_operands:
            if src_op.isnormalreg():
                instr_entry.src_loc_is_nb[src_op.op_str] = is_nb
        for dst_op in instr_entry.instr.dst_operands:
            if dst_op.isnormalreg():
                instr_entry.dst_loc_is_nb[dst_op.op_str] = is_nb

    def test_single_instruction_no_pred(self):
        # Print out a simple kernel with registers allocated and an instruction
        # NOTE: this kernel is for testing only
        with open(self.ptx_file, "w") as f:
            print(".visible .entry _Z9Kernel(", file=f)
            print("\t .param .u32 _Z9_param_0", file=f)
            print(")", file=f)
            print("{", file=f)
            print("\t .reg .f32\t %f<31>;", file=f)
            print("\t add.f32\t %f3 %f2 %f1;", file=f)
            print("}", file=f)
        
        kernel_list = prog_api.load_kernel(self.ptx_file)
        self.assertEqual(len(kernel_list), 1)
        kernel = kernel_list[0]
        self.assertEqual(len(kernel.arg_list), 1)
        self.assertEqual(len(kernel.instr_list), 1)
        kernel.init_instr_latency(self.config)

        # extract the instruction
        instr = kernel.instr_list[0]

        # update dependency info
        src_ops = []
        dst_ops = []
        if "pred_reg" in instr.metadata:
            src_ops.append(instr.metadata["pred_reg"])
        for each_op in instr.src_operands:
            if each_op.isreg():
                src_ops.append(each_op)
        for each_op in instr.dst_operands:
            if each_op.isreg():
                dst_ops.append(each_op)

        for each_op in src_ops:
            self.subcore.dep_table_exe.entry[self.subcore.warp_id]\
                .increase_read(each_op.op_str)

        for each_op in dst_ops:
            self.subcore.dep_table_exe.entry[self.subcore.warp_id]\
                .increase_write(each_op.op_str)
        
        # pre-set register tracking table
        # for testing only
        for each_op in src_ops:
            reg_addr, reg_size = self.subcore.get_subcore_reg_addr(
                each_op.reg_prefix,
                each_op.reg_index,
                self.subcore.warp_id
            )
            # Update register tracking table
            self.subcore.reg_track_table\
                .entry[self.subcore.warp_id]\
                .write_update(
                    op_str=each_op.op_str,
                    reg_file_type="far-bank"
                )
        
        assert instr.opcode.split(".")[0] in self.config["alu_instr"]
        instr.set_latency(self.config)
        # compose instruction entry
        simt_mask_str = "1" * self.config["num_threads_per_warp"]
        simt_mask = int(simt_mask_str, 2)
        instr_entry = InstrEntry(
            log=self.log,
            config=self.config,
            instr=instr,
            simt_mask=simt_mask,
            pc=0,
            subcore_id=0,
            warp_id=self.subcore.warp_id
        )
        self._set_operand_location(instr_entry)
        self.opc_alu.instr_entry_queue.put(instr_entry)
        # assume we use the first operand collector unit
        self.subcore.instr_complete[instr_entry.pc] = False

        # start simulation
        self.env.run()
        # check result
        self.assertEqual(self.subcore.instr_complete[instr_entry.pc], True)
        for each_op in src_ops:
            self.assertEqual(self.subcore.dep_table_exe
                             .entry[self.subcore.warp_id]
                             .read_dict[each_op.op_str], 0)
        for each_op in dst_ops:
            self.assertEqual(self.subcore.dep_table_exe
                             .entry[self.subcore.warp_id]
                             .write_dict[each_op.op_str], 0)
        self.subcore.reset()
    
    def test_single_instruction_pred(self):
        # Print out a simple kernel with registers allocated and an instruction
        # NOTE: this kernel is for testing only
        with open(self.ptx_file, "w") as f:
            print(".visible .entry _Z9Kernel(", file=f)
            print("\t .param .u32 _Z9_param_0", file=f)
            print(")", file=f)
            print("{", file=f)
            print("\t .reg .pred\t %p<10>;", file=f)
            print("\t .reg .f32\t %f<31>;", file=f)
            print("\t @%p0 add.f32\t %f4 %f3 %f2;", file=f)
            print("}", file=f)

        kernel_list = prog_api.load_kernel(self.ptx_file)
        self.assertEqual(len(kernel_list), 1)
        kernel = kernel_list[0]
        self.assertEqual(len(kernel.arg_list), 1)
        self.assertEqual(len(kernel.instr_list), 1)

        # extract the instruction
        instr = kernel.instr_list[0]

        src_ops = []
        dst_ops = []
        if "pred_reg" in instr.metadata:
            src_ops.append(instr.metadata["pred_reg"])
        for each_op in instr.src_operands:
            if each_op.isreg():
                src_ops.append(each_op)
        for each_op in instr.dst_operands:
            if each_op.isreg():
                dst_ops.append(each_op)

        for each_op in src_ops:
            self.subcore.dep_table_exe.entry[self.subcore.warp_id]\
                .increase_read(each_op.op_str)

        for each_op in dst_ops:
            self.subcore.dep_table_exe.entry[self.subcore.warp_id]\
                .increase_write(each_op.op_str)

        # pre-set register tracking table
        # for testing only
        for each_op in src_ops:
            reg_addr, reg_size = self.subcore.get_subcore_reg_addr(
                each_op.reg_prefix,
                each_op.reg_index,
                self.subcore.warp_id
            )
            # Update register tracking table
            self.subcore.reg_track_table\
                .entry[self.subcore.warp_id]\
                .write_update(
                    op_str=each_op.op_str,
                    reg_file_type="far-bank"
                )

        assert instr.opcode.split(".")[0] in self.config["alu_instr"]
        instr.set_latency(self.config)
        # compose instruction entry
        simt_mask_str = "1" * self.config["num_threads_per_warp"]
        simt_mask = int(simt_mask_str, 2)
        instr_entry = InstrEntry(
            log=self.log,
            config=self.config,
            instr=instr,
            simt_mask=simt_mask,
            pc=0,
            subcore_id=0,
            warp_id=self.subcore.warp_id
        )
        self._set_operand_location(instr_entry)
        self.opc_alu.instr_entry_queue.put(instr_entry)
        # assume we use the first operand collector unit
        self.subcore.instr_complete[instr_entry.pc] = False
        
        # start simulation
        self.env.run()
        # check result
        self.assertEqual(self.subcore.instr_complete[instr_entry.pc], True)
        for each_op in src_ops:
            self.assertEqual(self.subcore.dep_table_exe
                             .entry[self.subcore.warp_id]
                             .read_dict[each_op.op_str], 0)
        for each_op in dst_ops:
            self.assertEqual(self.subcore.dep_table_exe
                             .entry[self.subcore.warp_id]
                             .write_dict[each_op.op_str], 0)
        self.subcore.reset()


if __name__ == "__main__":
    unittest.main()

# This class implements dram message data structure


class PRTEntryReq:
    def __init__(self, instr_entry, offset_list, co_addr_list, pg_id):
        self.instr_entry = instr_entry
        self.offset_list = offset_list
        self.co_addr_list = co_addr_list
        self.pg_id = pg_id


class DRAMTransaction:
    def __init__(
        self, trans_type, mem_loc, row_addr, col_addr, global_mem_addr
    ):
        """
        Args:
            trans_type: transaction type (load/store)
            mem_loc: a location tuple formatted as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id, 
                bank_addr, bank_interface_offset)
            row_addr: bank row address
            col_addr: bank column address
            global_mem_addr: global memory address in integer format
        Returns:
        """
        self.type = trans_type
        self.p_id_y = mem_loc[0]
        self.p_id_x = mem_loc[1]
        self.c_id_y = mem_loc[2]
        self.c_id_x = mem_loc[3]
        self.pg_id = mem_loc[4]
        self.pe_id = mem_loc[5]
        self.bank_addr = mem_loc[6]
        self.row_addr = row_addr
        self.col_addr = col_addr
        self.global_mem_addr = global_mem_addr
        # the time load/store unit issues this transaction
        self.time = None
        # the data associated with this transaction
        self.data = None
        # used for update pending request table
        self.simt_mask = 0
        # used for in-place update
        self.data_mask = 0
        self.ld_for_update = False
        # used for return to a certain subcore
        self.subcore_id = None
        # used for transaction return location
        self.is_nb = True
        # used for identify PRT entry
        self.prt_id = None
        # used for remote request, return to lsu_remote
        self.is_remote = False
        # NOTE: for tracing only
        self.trace_subcore_id = None
        self.trace_warp_id = None
        return
    
    def get_mem_loc(self):
        """get memory location tuple
        Args:
        Return:
            mem_loc: a location tuple formatted as
                (proc_id_y, proc_id_x, core_id_y, core_id_x, pg_id, pe_id)
        """
        return (self.p_id_y, self.p_id_x, self.c_id_y, self.c_id_x, 
                self.pg_id, self.pe_id)

    def get_mem_addr(self):
        """get the global memory address
        Args:
        Return:
            global_mem_addr: global memory address. can be used to index into 
                the memory array for value
        """
        # this address is aligned to bank interface
        return self.global_mem_addr

    def get_processor_id(self):
        """get the processor id
        Args:
        Return:
            proc_id: a tuple formatted as
                (p_id_x, p_id_y)
        """
        return (self.p_id_x, self.p_id_y)

    def get_core_id(self):
        """get the core id
        Args:
        Return:
            core_id: a tuple formatted as
                (c_id_x, c_id_y)
        """
        return (self.c_id_x, self.c_id_y)


class DRAMCommand:
    def __init__(self, cmd_type, row_addr, col_addr, dram_trans=None):
        """
        Args:
            cmd_type: dram command type
            bank_addr: the local bank address
            config: reference to the global configuration

        Return:
        """
        self.type = cmd_type
        self.row_addr = row_addr
        self.col_addr = col_addr
        self.time = -1
        # NOTE: for tracing only
        self.dram_trans = dram_trans
        return


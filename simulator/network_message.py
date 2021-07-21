# This class implements network message data structure
import struct
from copy import deepcopy
import math


class SrcRemoteLoadReq:
    def __init__(self, addr_list, data_width, simt_mask):
        self.addr_list = addr_list
        self.data_width = data_width
        self.simt_mask = simt_mask
        self.prt_id = None
        # NOTE: for tracing only
        self.tracing = False


class SrcRemoteLoadResp:
    def __init__(self, addr_list, data_width, simt_mask):
        self.addr_list = addr_list
        self.data_width = data_width
        self.simt_mask = simt_mask
        self.data = None
        self.prt_id = None


class SrcRemoteStoreReq:
    def __init__(self, addr_list, data_width, simt_mask, data):
        self.addr_list = addr_list
        self.data_width = data_width
        self.simt_mask = simt_mask
        self.data = data
        self.prt_id = None
        # NOTE: for tracing only
        self.tracing = False


class SrcRemoteStoreResp:
    def __init__(self, addr_list, data_width, simt_mask):
        self.addr_list = addr_list
        self.data_width = data_width
        self.simt_mask = simt_mask
        self.prt_id = None


class DstRemoteLoadReq:
    def __init__(self, addr_list, data_width):
        self.addr_list = addr_list
        self.data_width = data_width


class DstRemoteLoadResp:
    def __init__(self, addr_list, data_width):
        self.addr_list = addr_list
        self.data_width = data_width
        self.data = None


class DstRemoteStoreReq:
    def __init__(self, addr_list, data_width, data):
        self.addr_list = addr_list
        self.data_width = data_width
        self.data = data


class DstRemoteStoreResp:
    def __init__(self, addr_list, data_width):
        self.addr_list = addr_list
        self.data_width = data_width


class NetworkLocationInfo:
    def __init__(self, proc_id, core_id):
        """Network location wrapper
        Args:
            proc_id: processor location tuple - (proc_id_x, proc_id_y)
            core_id: core location tuple - (core_id_x, core_id_y)
        """
        assert isinstance(proc_id, tuple)
        assert isinstance(core_id, tuple)
        self.proc_id = proc_id
        self.core_id = core_id
        self._loc_str = "proc_id={}, core_id={}"\
            .format(proc_id, core_id)

    def is_equal(self, loc):
        if (
            self.proc_id == loc.proc_id
            and self.core_id == loc.core_id
        ):
            return True
        else:
            return False


class NetworkPacket:
    def __init__(self, msg, seq_id, num_packet):
        assert isinstance(msg, NetworkMessage)
        self.msg = msg
        self.msg_type = msg.msg_type
        self.seq_id = seq_id
        self.num_packet = num_packet
        self.src_loc = msg.src_loc
        self.dst_loc = msg.dst_loc
        self.req_id = msg.req_id
        self.msg_id = msg.msg_id
        # NOTE (src_loc, req_id, msg_id) uniquely identifies
        # packets assoicated with a message


class NetworkMessage:
    def __init__(self, src_loc, dst_loc, msg_type, msg_id):
        assert isinstance(src_loc, NetworkLocationInfo)
        assert isinstance(dst_loc, NetworkLocationInfo)
        self.src_loc = src_loc
        self.dst_loc = dst_loc
        self.src_loc_str = "SRC: proc_id={}, core_id={}"\
            .format(src_loc.proc_id, src_loc.core_id)
        self.dst_loc_str = "DST: proc_id={}, core_id={}"\
            .format(dst_loc.proc_id, dst_loc.core_id)
        self.msg_type = msg_type
        # this is a dict
        # meta_data_name --> (start_addr_data, end_addr_data)
        # the later is an index into the data bytearray
        self.meta_data = {}
        # stores all data in bytearray format
        self.data = bytearray(0)
        # request track table id for the source core
        self.req_id = None
        # a req_id in the source core may contain multiple messages
        self.msg_id = msg_id
        # the following fields will not be transmitted
        # for the transmitting side: compose msg
        # for the receiving side: recover msg
        self.data_buffer = bytearray(0)
        self.addr_list = []
        self.data_width = None
        self.simt_mask = 0
        # for tracing only
        self.src_issue_cyc = None
        self.dst_rcv_cyc = None
        self.dst_issue_cyc = None
        self.src_rcv_cyc = None
        self.tracing = False
        
    def decompose_to_packet(self, packet_size):
        """Decompose this message into a number of packets
        Args:
            packet_size: packet size in bytes
        Returns: 
            packet_list: a list of packets
        """
        assert len(self.data) > 0
        num_packet = math.ceil(len(self.data) / packet_size)
        packet_list = []
        for i in range(num_packet):
            packet = NetworkPacket(
                msg=self,
                seq_id=i,
                num_packet=num_packet
            )
            packet_list.append(packet)
        return packet_list

    def encode_data(self):
        cur_ptr = 0
        if self.msg_type in ["ld_resp", "st_req"]:
            # encode data_buffer
            total_byte = len(self.data_buffer)
            assert total_byte % self.data_width == 0
            self.meta_data["data_buffer"] = (cur_ptr, cur_ptr + total_byte)
            self.data.extend(deepcopy(self.data_buffer))
            cur_ptr += total_byte
        if self.msg_type in ["ld_req", "st_req"]:
            # encode addr_list
            # NOTE: use 32b unsigned int to encode
            num_addr = len(self.addr_list)
            encoded_addr_list = bytearray(
                struct.pack(
                    "{}I".format(num_addr),
                    *self.addr_list
                )
            )
            total_byte = len(encoded_addr_list)
            self.meta_data["addr_list"] = (cur_ptr, cur_ptr + total_byte)
            self.data.extend(deepcopy(encoded_addr_list))
            cur_ptr += total_byte
            # encode data_width
            encoded_data_width = bytearray(
                struct.pack(
                    "1I",
                    self.data_width
                )
            )
            total_byte = len(encoded_data_width)
            self.meta_data["data_width"] = (cur_ptr, cur_ptr + total_byte)
            self.data.extend(deepcopy(encoded_data_width))
            cur_ptr += total_byte
        # encode simt_mask
        encoded_simt_mask = bytearray(
            struct.pack(
                "1I",
                self.simt_mask
            )
        )
        total_byte = len(encoded_simt_mask)
        self.meta_data["simt_mask"] = (cur_ptr, cur_ptr + total_byte)
        self.data.extend(deepcopy(encoded_simt_mask))
        cur_ptr += total_byte
        # encode req_id
        encoded_req_id = bytearray(
            struct.pack(
                "1I",
                self.req_id
            )
        )
        total_byte = len(encoded_req_id)
        self.meta_data["req_id"] = (cur_ptr, cur_ptr + total_byte)
        self.data.extend(deepcopy(encoded_req_id))
        cur_ptr += total_byte
        # encode msg_id
        encoded_msg_id = bytearray(
            struct.pack(
                "1I",
                self.msg_id
            )
        )
        total_byte = len(encoded_msg_id)
        self.meta_data["msg_id"] = (cur_ptr, cur_ptr + total_byte)
        self.data.extend(deepcopy(encoded_msg_id))
        cur_ptr += total_byte

    def decode_data(self):
        # decode msg_id
        start_addr, end_addr = self.meta_data["msg_id"]
        raw_data = self.data[start_addr: end_addr]
        self.msg_id = struct.unpack("1I", raw_data)[0]
        # decode req_id
        start_addr, end_addr = self.meta_data["req_id"]
        raw_data = self.data[start_addr: end_addr]
        self.req_id = struct.unpack("1I", raw_data)[0]
        # decode simt_mask
        start_addr, end_addr = self.meta_data["simt_mask"]
        raw_data = self.data[start_addr: end_addr]
        self.simt_mask = struct.unpack("1I", raw_data)[0]
        if self.msg_type in ["ld_req", "st_req"]:
            # decode data_width
            start_addr, end_addr = self.meta_data["data_width"]
            raw_data = self.data[start_addr: end_addr]
            self.data_width = struct.unpack("1I", raw_data)[0]
            # decode addr_list
            start_addr, end_addr = self.meta_data["addr_list"]
            raw_data = self.data[start_addr: end_addr]
            total_byte = len(raw_data)
            assert total_byte % self.data_width == 0
            num_addr = total_byte // 4
            self.addr_list = struct.unpack(
                "{}I".format(num_addr),
                raw_data
            )
            self.addr_list = list(self.addr_list)
        if self.msg_type in ["ld_resp", "st_req"]:
            # decode data_buffer
            start_addr, end_addr = self.meta_data["data_buffer"]
            self.data_buffer = self.data[start_addr: end_addr]

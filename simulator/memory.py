class Memory:

    def __init__(self, alignment=128, size=4294967296):
        self.alignment = alignment 
        self.mem_size = size
        self.curr_ptr = 0
        self.array = None 

    def allocate(self, size):
        """Allocate the memory according to the requested size, and align 
        the requested size according to the alignment requirement

        Args:
            size: the number of bytes to be allocated 

        Returns:
            returned_ptr: the starting address of the allocated buffer
        """

        assert self.array is None, "The memory space has been finalized"
        returned_ptr = self.curr_ptr 

        aligned_size = ((size - 1) // self.alignment + 1) * self.alignment 
        self.curr_ptr = self.curr_ptr + aligned_size 
        assert self.curr_ptr <= self.mem_size, "Out of memory!" 

        return returned_ptr 

    def finalize(self): 
        """Allocate a bytearray according to the size of already allocated 
        buffers
        """ 
        assert self.array is None, "The memory space hs been finalized"
        assert self.curr_ptr <= self.mem_size, "Out of memory!" 

        if self.curr_ptr != 0:
            self.array = bytearray(self.curr_ptr)

        return 

    def set_value(self, addr, data_buffer):
        """Set the value of buffers inside the allocated memory space 

        Args:
            addr: the starting address in the memory space 
            data_buffer: the data buffer containing bytes 

        Returns:
            None 
        """

        assert self.array is not None, "The memory needs to be finalized"

        buffer_size = len(data_buffer)
        assert (addr + buffer_size) <= self.curr_ptr, "Out of range access"

        self.array[addr: addr + buffer_size] = bytearray(data_buffer) 
        return 

    def get_value(self, addr, buffer_size):
        """Get the value of buffers inside the allocated memory space 

        Args:
            addr: the starting address in the memory space
            buffer_size: the size of data buffer in bytes 

        Returns:
            output_buffer: the buffer contains data in bytes 
        """

        assert self.array is not None, "The memory needs to be finalized" 
        assert (addr + buffer_size) <= self.curr_ptr, "Out of range access, " \
            "access_addr={} max_addr={}".format(addr, self.curr_ptr)

        output_buffer = self.array[addr: addr + buffer_size] 

        return output_buffer 

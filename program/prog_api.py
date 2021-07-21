import logging 

from program.kernel import Kernel 
from backend.branch_analysis import reconvergence_analysis 
from backend.register_allocation import allocate_register
from backend.annotate_location import annotate_location 


def load_kernel(input_ptx_file, logger=None):
    """
    Args:
        input_ptx_file: the file path of PTX file
        logger: a logger to log information for the purpose of debugging 

    Returns:
        kernel_list: a list of kernels loaded from PTX file 
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

    kernel_list = []

    with open(input_ptx_file, "r") as f:
        lines = f.readlines() 

        kernel_context = []
        flag = False 

        for each_line in lines:            
            line_context = ""
            pos = each_line.find("//")

            if pos >= 0:
                line_context = each_line[:pos].strip(" \n\t")
            else:
                line_context = each_line.strip(" \n\t")

            if len(line_context) == 0:
                continue 

            if line_context.find(".entry") >= 0:
                flag = True 

            if flag:
                kernel_context.append(line_context) 

            if line_context.find("}") >= 0:
                kernel_list.append(Kernel(lines=kernel_context, log=logger))
                flag = False 
                kernel_context = [] 

    return kernel_list 


def compile_to_exec(input_kernel, hw_config):
    """This function compiles kernel into MPU executable kernel. 
    It includes two main compilation passes:
        1. Branch re-convergence discovery 
        2. Register allocation 

    Args:
        input_kernel: the kernel function before compilation
        hw_config: the hardware configuration 

    Returns:
        exec_kernel: the kernel function after compilation 
    """ 
    output_kernel = reconvergence_analysis(input_kernel) 
    annotated_kernel = annotate_location(output_kernel, hw_config)
    exec_kernel = allocate_register(annotated_kernel, hw_config)
    return exec_kernel 


def optimize_kernel(input_kernel, hw_config, loc=True):
    """This function provides the interface for invoking compilation
    passes to optimize kernels 

    Args:
        input_kernel: the kernel fuction before optimization
        hw_config: the hardware configuration 
        loc: (optional) whether to infer and assign register location 

    Returns:
        output_kernel: the kernel function after optimization 
    """

    raise NotImplementedError 

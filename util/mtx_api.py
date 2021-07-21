import random 


def gen_random_graph(num_nodes, num_edges, directed=False):
    """Generates a random graph represetned by the CSR format. 

    Args:
        num_nodes: the number of nodes inside the graph. 
        num_edges: the number of edges inside the graph. 
        directed: whether the generated graph is a directed graph or not. 

    Returns:
        Three list representing three arrays of the graph in the CSR format. 
    """
    random.seed(6)

    mm_dict = []
    for i in range(num_nodes):
        mm_dict.append({})

    rest_edges = num_edges 
    while (rest_edges > 0):
        src_node = random.randint(0, num_nodes - 1)
        dst_node = random.randint(0, num_nodes - 1)

        if src_node == dst_node:
            continue 

        if dst_node in mm_dict[src_node]:
            continue 

        mm_dict[src_node][dst_node] = 1.0 
        if not directed:
            mm_dict[dst_node][src_node] = 1.0 

        rest_edges = rest_edges - 1

    ptr_row = []
    ptr_col = []
    ptr_val = []

    ptr_row.append(0)
    for row_index in range(num_nodes):
        col_index_list = sorted([k for k in mm_dict[row_index].keys()])
        for col_index in col_index_list:
            ptr_col.append(col_index)
            ptr_val.append(mm_dict[row_index][col_index])
        ptr_row.append(len(ptr_col))

    return (ptr_row, ptr_col, ptr_val)


def _check_symmetric(ptr_row, ptr_col, ptr_val):
    """Check whether a sparse matrix in the CSR format is a symmetric matrix. 

    Args:
        ptr_row: the pointer to the row index. 
        ptr_col: the pointer to the column index. 
        ptr_val: the pointer to the value array. 

    Returns:
        The number of non-duplicate non-zero elements in this sparse matrix. A
        negative value means thee input matrix is not a symmetric matrix. 
    """
    mm_dict = {}
    for row_index in range(len(ptr_row) - 1):
        for offset in range(ptr_row[row_index], ptr_row[row_index + 1]):
            col_index = ptr_col[offset]
            val = ptr_val[offset]
            mm_dict[(row_index, col_index)] = val 

    num_nnz = 0
    for key in mm_dict.keys():
        row_index, col_index = key 
        if (row_index, col_index) not in mm_dict:
            return -1 
        if (col_index, row_index) not in mm_dict:
            return -1 
        if mm_dict[(row_index, col_index)] != mm_dict[(col_index, row_index)]:
            return -1
        if row_index <= col_index:
            num_nnz = num_nnz + 1

    return num_nnz 


def dump_mtx_file(output_file_path, num_rows, num_cols, ptr_row, ptr_col,
                  ptr_val, symmetric=False):
    """Dump a sparse mmatrix in the CSR format to an output file. 

    Args:
        output_file_path: the file path of the output MTX file. 
        num_rows: the number of rows in the matrix.
        num_cols: the number of columns in the matrix.
        ptr_row: the pointer to the row index. 
        ptr_col: the pointer to the column index. 
        ptr_val: the pointer to the value array. 
        symmetric: whether the sparse matrix is a symmetric matrix.

    Returns:
        None. 
    """
    with open(output_file_path, "w") as f:
        num_nnz = len(ptr_val)
        if symmetric:
            num_nnz = _check_symmetric(ptr_row, ptr_col, ptr_val)
            assert num_nnz >= 0, "The input matrix in the CSR format is not "\
                "a symmetric matrix!"
            print("% symmetric", file=f)

        print("{} {} {}".format(num_rows, num_cols, num_nnz), file=f)
        for row_index in range(len(ptr_row) - 1):
            for offset in range(ptr_row[row_index], ptr_row[row_index + 1]):
                col_index = ptr_col[offset]
                val = ptr_val[offset]
                if (symmetric is False) or (col_index <= row_index):
                    print("{} {} {}".format(row_index + 1, col_index + 1, val), 
                          file=f)
    return 


def load_mtx_file(file_path):
    """Load a sparse matrix from the MTX file 

    Args:
        file_path: the path of the MTX file. 

    Returns:
        Three lists including the row index pointers, column index, and 
        the value of non-zero elements. 
    """
    ptr_row = []
    ptr_col = []
    ptr_val = []

    symmetric = False 
    num_rows = 0
    num_cols = 0

    with open(file_path, "r") as f:
        lines = f.readlines() 

        start_index = 0
        for i in range(len(lines)):
            each_line = lines[i]
            if each_line[0] == "%":
                if each_line.find("symmetric") >= 0:
                    symmetric = True
            else:
                vals = each_line.strip(" \n").split(" ")
                assert len(vals) == 3, "The head line should follow the " \
                    "format: <num rows> <num cols> <num nnz>"
                num_rows = int(vals[0])
                num_cols = int(vals[1])
                start_index = i 
                break 

        mm_dict = []
        for i in range(num_rows):
            mm_dict.append({})

        for i in range(start_index + 1, len(lines)):
            each_line = lines[i]
            vals = each_line.strip(" \n").split(" ")
            if len(vals) == 2:
                vals.append("1.0") 
            assert len(vals) == 3

            row_index = int(vals[0]) - 1
            col_index = int(vals[1]) - 1
            nnz_val = float(vals[2])

            assert (row_index >= 0) and (row_index < num_rows)
            assert (col_index >= 0) and (col_index < num_cols)

            mm_dict[row_index][col_index] = nnz_val 
            if symmetric:
                mm_dict[col_index][row_index] = nnz_val 

        ptr_row.append(0)
        for i in range(num_rows):
            col_index_list = sorted([k for k in mm_dict[i].keys()]) 
            for each_col_index in col_index_list:
                ptr_col.append(each_col_index)
                ptr_val.append(mm_dict[i][each_col_index])
            ptr_row.append(len(ptr_col))

    return (ptr_row, ptr_col, ptr_val) 

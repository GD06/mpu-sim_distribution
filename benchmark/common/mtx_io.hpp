#ifndef _MTX_IO_HPP
#define _MTX_IO_HPP 

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <vector>
#include <string>

using namespace std;


class CSRMatrix {
    public:
        int num_rows;
        int num_cols;
        int num_nnz;
        int* row_ptr;
        int* col_index;
        float* nnz_val;
};


bool MTXFileToCSRMatrix(const string& file_path, CSRMatrix* mtx_ptr) {
    FILE* fp;
    fp = fopen(file_path.c_str(), "r");
    if (fp == NULL) {
        printf("Can not open the file: %s\n", file_path.c_str());
        return false; 
    }

    bool symmetric = false;
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    int M = 0;
    int N = 0;
    int L = 0;

    while ((read = getline(&line, &len, fp)) != -1) {
        if (line[0] != '%') {
            sscanf(line, "%d %d %d\n", &M, &N, &L);
            break;
        }

        string line_str(line);
        if (line_str.find("symmetric") != string::npos) {
            symmetric = true;
        }

        free(line);
        line = NULL;
        len = 0;
    }

    if (M <= 0) {return false;}
    if (N <= 0) {return false;}
    if (L <= 0) {return false;}

    map<int, map<int, float> > mm_dict;
    for (int i = 0; i < M; ++i) {
        map<int, float> row_dict;
        row_dict.clear();
        mm_dict[i] = row_dict;
    }

    for (int i = 0; i < L; ++i) {
        getline(&line, &len, fp);

        int row, col, ret;
        float value;
        ret = sscanf(line, "%d %d %f\n", &row, &col, &value);

        if (3 != ret) {
            if (2 == ret) {
                value = 1.0;
            } else {
                printf("Invalid input line, should be <row> <col> (<val>)\n");
                return false;
            }
        }

        if ((row > M) || (col > N) || (row <= 0) || (col <= 0)) {
            printf("Invalid non-zero element: (%d, %d) in size %d x %d\n",
                    row, col, M, N);
            return false;
        }

        row = row - 1;
        col = col - 1;

        mm_dict[row][col] = value;
        if (symmetric) {
            mm_dict[col][row] = value;
        }

        free(line);
        line = NULL;
        len = 0;
    }
    fclose(fp);

    mtx_ptr->num_rows = M;
    mtx_ptr->num_cols = N;
    mtx_ptr->row_ptr = (int*)malloc((M + 1) * sizeof(int));

    int num_nnz = 0;
    mtx_ptr->row_ptr[0] = 0;
    for (int i = 0; i < M; ++i) {
        num_nnz += mm_dict[i].size();
        mtx_ptr->row_ptr[i + 1] = num_nnz;
    }

    mtx_ptr->num_nnz = num_nnz;
    mtx_ptr->col_index = (int*)malloc(num_nnz * sizeof(int));
    mtx_ptr->nnz_val = (float*)malloc(num_nnz * sizeof(float));

    for (int i = 0; i < M; ++i) {
        int offset = mtx_ptr->row_ptr[i];
        vector<int> tmp_col_index;
        tmp_col_index.clear(); 

        for (map<int, float>::iterator it = mm_dict[i].begin();
                it != mm_dict[i].end(); ++it) {
            tmp_col_index.push_back(it->first);
        }

        sort(tmp_col_index.begin(), tmp_col_index.end());

        for (int j = 0; j < tmp_col_index.size(); ++j) {
            int col = tmp_col_index[j];
            mtx_ptr->col_index[offset + j] = col;
            mtx_ptr->nnz_val[offset + j] = mm_dict[i][col];
        }
    }
    printf("MTX file head: %d %d %d\n", M, N, L);
    printf("Loaded matrix: %d %d %d\n", mtx_ptr->num_rows, mtx_ptr->num_cols,
            mtx_ptr->num_nnz);
 
    return true; 
}


#endif 

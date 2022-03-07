import math, pprint


# matrix dimensions

M = 256
K = 256

data_word_bits = 8
overhead_summary = {}
total_matrix_size_summary = {}
norm_matrix_size_summary = {}

# densities
densities = [0.01, 0.05, 0.1, 0.5, 0.9]

for density in densities:
   
    print("--------------\n density: %f \n--------------" %density)

    nnz = math.ceil(M*K*density)

    norm_dense = M * K * data_word_bits


    # bitmask
    overhead_summary["bitmask"] = M*K

    # COO
    M_coord_md_width = math.ceil(math.log2(M))
    K_coord_md_width = math.ceil(math.log2(K))
    print("COO: M_coord_md_width: %d, K_coor_md_width: %d" %(M_coord_md_width, K_coord_md_width))
    overhead_summary["coo"] = nnz * (M_coord_md_width + K_coord_md_width) 

    # CSR
    uncompressed_rank_M_md_width = math.ceil(math.log2(nnz))
    cp_rank_K_md_width = math.ceil(math.log2(K))
    overhead_summary["csr"] = nnz * cp_rank_K_md_width + (M+1)*uncompressed_rank_M_md_width

    for fmt, abs_info in overhead_summary.items():
        total_matrix_size_summary[fmt] = nnz * data_word_bits + overhead_summary[fmt]
        norm_matrix_size_summary[fmt] = round(total_matrix_size_summary[fmt]/norm_dense, 4)

    pprint.pprint(norm_matrix_size_summary)

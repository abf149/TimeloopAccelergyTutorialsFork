

# run sparse cases
for d in 0.01 0.02 0.04 0.08 0.1 0.2 0.4 0.8 
do
  prob_file_name="spmspm.prob.${d}.yaml"
  for sparse_setup in  "coo" "bitmask" "bitmask.skip" 
  do
    out_dir_name="output_${d}_${sparse_setup}"
    mkdir ../${out_dir_name}
    timeloop-mapper ../arch/*.yaml ../dataflow/*.yaml ../sparse-opt/${sparse_setup}.yaml  ../mapper/*.yaml  ../prob/${prob_file_name} ../ert_art/*.yaml -o ../${out_dir_name}/
  done
done

# grep results
grep -A 5 "Summary Stats" ../output_*/timeloop-mapper*.stats.txt > raw_summary.log
grep "Cycles" raw_summary.log
grep "Energy" raw_summary.log

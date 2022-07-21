if [[ $# -eq 1 ]]; then
   mkdir $1
   scp gajuan@euler.ethz.ch:"/cluster/scratch/gajuan/$1{test_cases,compiled_results_ges,compiled_results_gies,compiled_results_gnies,compiled_results_ut_igsp,compiled_results_ut_igsp_plus}.pickle" $1
else
    mkdir $2
    scp gajuan@euler.ethz.ch:"$1{test_cases,compiled_results_ges,compiled_results_gies,compiled_results_gnies,compiled_results_ut_igsp,compiled_results_ut_igsp_plus}.pickle" "$2"
fi

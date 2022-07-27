FILENAMES="{test_cases.pickle,compiled_results*}"
if [[ $# -eq 1 ]]; then
   mkdir $1
   scp gajuan@euler.ethz.ch:"/cluster/scratch/gajuan/$1$FILENAMES" $1
else
    mkdir $2
    scp gajuan@euler.ethz.ch:"$1$FILENAMES" "$2"
fi

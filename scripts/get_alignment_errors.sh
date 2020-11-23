grep -r "No alignment" exp/tri6/log/acc.*.log > align_errors.log 
awk 'NF>1{print $NF}' align_errors.log | sort > align_errors_clean.log

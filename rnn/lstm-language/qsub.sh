qsub -cwd -S /bin/bash -o LOG -e ERR -l hostname=ullman,gpu=1 -q GTX970.q run.sh

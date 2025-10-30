#! /bin/bash

# #############################################################################
# Script to convert python notebook(s) given in argument to python script(s)
# 
# In python script(s):
# - cells containing "Choose backend" are commented
# - `plt_show` and `off_screen` variable are toggled (True <-> False)
# #############################################################################

# usage: set_py_script.sh <file1.ipynb> [<file2.ipynb> ...]

# ./set_py_script.sh 0*.ipynb

for f in $@
do
    # convert to script
    jupyter nbconvert --to script $f
    py_f="${f%.ipynb}.py"

    # Comment cells containing "Choose backend" string
    nn=($(grep -n "Choose backend" ${py_f} | cut -d: -f1))
    nc=($(grep -n "# In\[" ${py_f} |cut -d: -f1))
    for n in ${nn[@]}
    do
        for (( i = 0 ; i < ${#nc[@]} ; i++ )) ; do if [ ${n} -lt ${nc[i]} ] ; then break ; fi ; done
        n1=${nc[i-1]}
        n2=$((nc[i]-1))
        # echo $n $n1 $n2
        sed -i -e "${n1},${n2}s/\(^.*$\)/# \1/" ${py_f}
    done

    # Replace "plt_show = True " by "plt_show = False"
    sed -i -e "s/plt_show = True /plt_show = False/" ${py_f}

    # Replace "off_screen = False" by "off_screen = True "
    sed -i -e "s/off_screen = False/off_screen = True /" ${py_f}
done

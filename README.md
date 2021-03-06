# tcl_code

## Create edgemap generation batch scripts

1. `python make_batch_scripts.py <dir to find mha files> <dir to output scripts`

## Submit batch scripts

`bash submit_batch_scripts.sh <dir containing bash scripts>`

## Compute a single edgemap
1. `cd <directory of .mha file>`
2. `python /path/to/tcl_code/I2INet3DMed.py <netFile> <caffeModel>`

## Generate list of files to compute groups for
`bash make_svfile_list.sh <vascular data directory>`

## Generate groups using image with and without edge maps

1. Open simvascular
2. Copy code from `seg_acc_test.tcl` and paste it in the simvascular console
3. Run the command `runEdgeAnalysis`

## Compute group errors and store in CSV

`python groups_toCSV.py <vascular data repository>`

## make plots

`python make_plots.py`

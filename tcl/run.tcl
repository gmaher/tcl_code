set where [file dirname [info script]]
source [file join $where lsVTKgen.tcl]

global itklsGUIParams
set itklsGUIParams(phyRadius) 0.3

generate_truth_groups /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm
close_files

generate_edge_groups /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm
close_files

generate_edge_groups /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha edge96
close_files

generate_edge_groups /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha edgeDist96 \
userEdgeDistance
close_files

run_kthr {1 2 3 5 10} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96
close_files

run_params {0.3 0.8 0.6 0.9} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_r03klow08kthr06kupp09 image_r03klow08kthr06kupp09 userEdge
close_files

run_params {0.3 0.8 0.85 0.9} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_r03klow08kthr085kupp09 image_r03klow08kthr06kupp09 userEdge
close_files

run_params {0.4 0.8 0.85 0.9} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_r04klow08kthr085kupp09 image_r03klow08kthr06kupp09 userEdge
close_files

run_params {0.5 0.8 0.85 0.9} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_r05klow08kthr085kupp09 image_r03klow08kthr06kupp09 userEdge

run_params {0.5 0.8 0.85 0.9} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_r05klow08kthr085kupp09 image_r03klow08kthr06kupp09 userEdgeDistance

run_params {0.3 0.8 0.85 0.9} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_r05klow08kthr085kupp09 image_r03klow08kthr06kupp09 userEdgeDistance
close_files

#standard
run_params {0.3 0.09 0.6 0.8} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_LSEdge image_LSEdge LSEdge
close_files

run_params {0.8 0.8 0.6 0.8} /home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm.mha \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001-cm.paths \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/0006_0001/0006_0001_groups-cm \
/home/marsdenlab/datasets/vascular_data/OSMSC0006/OSMSC0006-cm_E96.mha \
edge96_LSEdger08klow08kthr06kupp08 image_LSEdger08klow08kthr06kupp08 LSEdge
close_files
exit

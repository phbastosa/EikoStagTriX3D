#!/bin/bash

admin="../src/admin/admin.cpp"
geometry="../src/geometry/geometry.cpp"

# Seismic modeling scripts ----------------------------------------------------------------------------

folder="../src/modeling"

modeling="$folder/modeling.cu"

elastic_iso="$folder/elastic_iso.cu"
elastic_ani="$folder/elastic_ani.cu"

modeling_main="../src/modeling_main.cpp"

modeling_all="$modeling $elastic_iso $elastic_ani"

# Compiler flags --------------------------------------------------------------------------------------

flags="-Xcompiler -fopenmp --std=c++11 --use_fast_math --relocatable-device-code=true -lm -O3"

# Main dialogue ---------------------------------------------------------------------------------------

USER_MESSAGE="
-------------------------------------------------------------------------------
                                 \033[34mWASMEM3D\033[0;0m
-------------------------------------------------------------------------------
\nUsage:\n
    $ $0 -compile              
    $ $0 -modeling                      

Tests:

    $ $0 -test_modeling                      
    
-------------------------------------------------------------------------------
"

[ -z "$1" ] && 
{
	echo -e "\nYou didn't provide any parameter!" 
	echo -e "Type $0 -help for more info\n"
    exit 1 
}

case "$1" in

-h) 

	echo -e "$USER_MESSAGE"
	exit 0
;;

-compile) 

    echo -e "Compiling stand-alone executables!\n"

    echo -e "../bin/\033[31mmodeling.exe\033[m" 
    nvcc $admin $geometry $modeling_all $modeling_main $flags -o ../bin/modeling.exe

	exit 0
;;

-clean)

    rm ../bin/*.exe
    rm ../inputs/models/*.bin
    rm ../inputs/geometry/*.txt
    rm ../outputs/snapshots/*.bin
    rm ../outputs/seismograms/*.bin
;;

-modeling) 

    ./../bin/modeling.exe parameters.txt
	
    exit 0
;;

-test_homogeneous)

    folder=../tests/homogeneous
    parameters=$folder/parameters.txt

    python3 -B $folder/prepare_models.py

    ./../bin/modeling.exe $parameters

    sed -i "s|modeling_type = 0|modeling_type = 1|g" "$parameters"

    ./../bin/modeling.exe $parameters

    sed -i "s|modeling_type = 1|modeling_type = 0|g" "$parameters"

    python3 -B $folder/prepare_results.py $parameters

	exit 0
;;

-test_anisotropy)

    folder=../tests/anisotropy
    parameters=$folder/parameters.txt

    eikonal_file=../outputs/snapshots/elastic_ani_eikonal_201x201x201_shot_1.bin 
    snapshot_file=../outputs/snapshots/elastic_ani_snapshot_step400_201x201x201_shot_1.bin 

    python3 -B $folder/prepare_models.py 0.1 0.0 0.0 0.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eT_dF_gF_txF_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eT_dF_gF_txF_tyF.bin 

    python3 -B $folder/prepare_models.py 0.0 0.1 0.0 0.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eF_dT_gF_txF_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eF_dT_gF_txF_tyF.bin 

    python3 -B $folder/prepare_models.py 0.0 0.0 0.1 0.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eF_dF_gT_txF_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eF_dF_gT_txF_tyF.bin 

    python3 -B $folder/prepare_models.py 0.1 0.1 0.0 0.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eT_dT_gF_txF_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eT_dT_gF_txF_tyF.bin 

    python3 -B $folder/prepare_models.py 0.1 0.0 0.1 0.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eT_dF_gT_txF_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eT_dF_gT_txF_tyF.bin 

    python3 -B $folder/prepare_models.py 0.0 0.1 0.1 0.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eF_dT_gT_txF_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eF_dT_gT_txF_tyF.bin 

    python3 -B $folder/prepare_models.py 0.1 0.1 0.1 0.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eT_dT_gT_txF_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eT_dT_gT_txF_tyF.bin 

    python3 -B $folder/prepare_models.py 0.1 0.1 0.1 30.0 0.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eT_dT_gT_txT_tyF.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eT_dT_gT_txT_tyF.bin 

    python3 -B $folder/prepare_models.py 0.1 0.1 0.1 0.0 30.0

    ./../bin/modeling.exe $parameters

    mv $eikonal_file ../outputs/snapshots/anisotropy_eikonal_eT_dT_gT_txF_tyT.bin 
    mv $snapshot_file ../outputs/snapshots/anisotropy_snapshot_eT_dT_gT_txF_tyT.bin 

    python3 -B $folder/prepare_results.py $parameters

	exit 0
;;

* ) 

	echo -e "\033[31mERRO: Option $1 unknown!\033[m"
	echo -e "\033[31mType $0 -h for help \033[m"
	
    exit 3
;;

esac
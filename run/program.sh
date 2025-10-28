#!/bin/bash

admin="../src/admin/admin.cpp"
geometry="../src/geometry/geometry.cpp"

# Seismic modeling scripts ----------------------------------------------------------------------------

folder="../src/modeling"

modeling="$folder/triclinic.cu"

triclinic_ssg="$folder/triclinic_ssg.cu"
triclinic_rsg="$folder/triclinic_rsg.cu"

modeling_main="../src/modeling_main.cpp"

modeling_all="$modeling $triclinic_ssg $triclinic_rsg"

# Compiler flags --------------------------------------------------------------------------------------

flags="-Xcompiler -fopenmp --std=c++11 --use_fast_math --relocatable-device-code=true -lm -O3"

# Main dialogue ---------------------------------------------------------------------------------------

USER_MESSAGE="
-------------------------------------------------------------------------------
 \033[34mEikoStagTriX3D\033[0;0m
-------------------------------------------------------------------------------
\nUsage:\n
    $ $0 -compile              
    $ $0 -modeling                      
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

* ) 

	echo -e "\033[31mERRO: Option $1 unknown!\033[m"
	echo -e "\033[31mType $0 -h for help \033[m"
	
    exit 3
;;

esac
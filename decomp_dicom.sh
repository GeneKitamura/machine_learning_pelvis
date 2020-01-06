#!/usr/bin/env bash
for i in $(<fail_paths.txt); do
    #echo $i
    #c_name=$( echo $i | cut -d'/' -f 5)
    #echo $c_name
    dcmdjpeg $i $i +ua
done
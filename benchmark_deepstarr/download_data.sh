#!/bin/bash

for group in Train Val Test; do
    wget -c https://zenodo.org/records/5502060/files/Sequences_${group}.fa
    wget -c https://zenodo.org/records/5502060/files/Sequences_activity_${group}.txt
done

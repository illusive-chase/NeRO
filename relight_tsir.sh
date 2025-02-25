#!/bin/bash

# list="ball car coffee helmet teapot toaster"
# list="ball car"

list="arm ficus hotdog lego"
envmaps="bridge city fireplace forest night"
# list="hotdog"
# envmaps="city"
for envmap in $envmaps
do
echo $envmap
for i in $list
do
    # Stage I: Shape reconstruction
    # Run the training script
    # python run_training.py --cfg configs/shape/nerf/${i}.yaml
    # # Extract mesh from the model
    # python extract_mesh.py --cfg configs/shape/nerf/${i}.yaml

    # # Stage II: Material estimation
    # # Run the training script
    # python run_training.py --cfg configs/material/nerf/${i}.yaml
    # # Extract materials from the model.
    # python extract_materials.py --cfg configs/material/nerf/${i}.yaml

    # Stage III: Run relighting script
    echo $i
    python relight.py --blender ./blender-3.4.0-linux-x64/blender \
                  --name tsir_$i-$envmap \
                  --mesh data/meshes/tsir_${i}_shape-300000.ply \
                  --material data/materials/tsir_${i}_material-100000 \
                  --hdr data/$envmap.hdr \
                  --json data/$i.json \
                  --albedo_scaling data/materials/tsir_${i}_material-100000/albedo_scaling.npy 2>&1 >> temp.log

done
done

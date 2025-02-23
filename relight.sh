#!/bin/bash

# list="ball car coffee helmet teapot toaster"
# list="ball car"

list="airbaloons chair hotdog jugs"
envmaps="envmap6 envmap12"
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
                  --name $i-$envmap \
                  --mesh data/meshes/${i}_shape-300000.ply \
                  --material data/materials/${i}_material-100000 \
                  --hdr /root/autodl-tmp/data/yekai/dev/RadianceFieldStudio/data/Synthetic4Relight/$envmap.exr \
                  --json /root/autodl-tmp/data/yekai/dev/RadianceFieldStudio/data/Synthetic4Relight/$i/transforms_test.json \
                  --albedo_scaling data/materials/${i}_material-100000/albedo_scaling.npy 2>&1 >> temp.log

done
done

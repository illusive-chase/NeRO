#!/bin/bash

# list="ball car coffee helmet teapot toaster"
# list="ball car"

list="tsir_arm tsir_ficus tsir_hotdog tsir_lego"
envmaps="bridge city fireplace forest night sunset"
# list="tsir_arm"
# envmaps="bridge"
for envmap in $envmaps
do
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
    python relight.py --blender ./blender-3.4.0-linux-x64/blender \
                  --name $i-$envmap \
                  --mesh data/meshes/${i}_shape-300000.ply \
                  --material data/materials/${i}_material-100000 \
                  --hdr /data/gaochong/project/datasets//Tensoir_Nerf/hdr/$envmap.hdr \
                  --json /data/gaochong/project/datasets//Tensoir_Nerf/$i/transforms_test.json

done
done

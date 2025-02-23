#!/bin/bash

# list="ball car coffee helmet teapot toaster"
# list="coffee helmet"
# list="chair"
list="tsir_ficus"

for i in $list
do
    # Stage I: Shape reconstruction
    # Run the training script
    python run_training.py --cfg configs/shape/nerf/${i}.yaml
    # Extract mesh from the model
    python extract_mesh.py --cfg configs/shape/nerf/${i}.yaml

    # Stage II: Material estimation
    # Run the training script
    python run_training.py --cfg configs/material/nerf/${i}.yaml
    # Extract materials from the model.
    python extract_materials.py --cfg configs/material/nerf/${i}.yaml

    # Stage III: Run relighting script
    # python relight.py --blender <path-to-your-blender> \
    #               --name bell-neon \
    #               --mesh data/meshes/bell_shape-300000.ply \
    #               --material data/materials/bell_material-100000 \
    #               --hdr data/hdr/neon_photostudio_4k.exr \
    #               --trans                  
    # python relight.py --blender <path-to-your-blender> \
    #                 --name bear-neon \
    #                 --mesh data/meshes/bear_shape-300000.ply \
    #                 --material data/materials/bear_material-100000 \
    #                 --hdr data/hdr/neon_photostudio_4k.exr

    # export PYTHONPATH=. && python train_tensoIR_simple.py \
    #     --config ./configs/single_light/blender.txt \
    #     --datadir ${root_dir}${i} --expname ${i} \
    #     --basedir ./log/log_shiny_blender
done
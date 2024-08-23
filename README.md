## Welcome to Pano2Room!

[Pano2Room: Novel View Synthesis from a Single Indoor Panorama (SIGGRAPH Asia 2024)](https://arxiv.org/abs/2408.11413).

## Overview
#### In short, Pano2Room converts an input panorama into 3DGS.

<img src="demo/Teaser.png" width="100%" >
Fig.1.Teaser: With a single panorama as input, Pano2Room automatically reconstructs the corresponding indoor scene with a 3D Gaussian Splatting field, capable of synthesizing photo-realistic novel views as well as high-quality depth maps. The panorama is generated using our panoramic RGBD inpainter based on any capture at a single location easily acquired by an average user.

<img src="demo/Overview.png" width="100%" >
Fig.2.Overview: With a panorama as input, we first predict the geometry of the panorama using the panoramic RGBD inpainter. Then we synthesize the initial mesh using a Pano2Mesh module. Next, we iteratively search for cameras with the least view completeness, and under the searched viewpoint, we render the existing mesh to obtain panoramic RGBDs with missing areas. To complete each rendered RGBD, we use the panoramic RGBD inpainter to generate new textures and predict new geometries. The new textures/geometries are iteratively fused into the existing mesh if no geometry conflict is introduced. Finally, the inpainted mesh is converted to a 3DGS and trained with collected pseudo novel views.


## Demo
In this demo, specify input panorama as:
<img src='demo/input_panorama.png' width="100%" >

Then, Pano2Room generates the corresponding 3DGS and renders novel views:

<img src="demo/GS_render_video.gif" width="40%" >

And the corresponding rendered depth:

<img src="demo/GS_depth_video.gif" width="40%" >

### 0. Setup the environment
(1) Create a new conda environment specified in \<requirements.txt\>. Please install [Pytorch3D](https://github.com/facebookresearch/pytorch3d) (for mesh rendering) and [diff-gaussian-rasterization-w-depth](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth) (for 3DGS rendering with depth) accordingly.

(2) Download pretrained weights in \<checkpoints\> (for image inpainting and depth estimation).

### 1. Run Demo
```
sh scripts/run_Pano2Room.sh
```
This demo converts 'input/input_panorama.jpg' to 3DGS and renders novel views as in 'output/'.

### (Optional) 0.5. Fine-tune Inpainter (SDFT)

### (Optional) 1.5. Try on your own panorama!

Simply replace (/input/input_panorama.png) with your own panorama.

## Cite our paper

If you find our work helpful, please cite our paper. Thank you!

ACM Reference Format:
```
Guo Pu, Yiming Zhao, and Zhouhui Lian. 2024. Pano2Room: Novel View Synthesis from a Single Indoor Panorama. In SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 3--6, 2024, Tokyo, Japan. ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/3680528.3687616
```


Instructions are updating...

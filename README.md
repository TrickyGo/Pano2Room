### Welcome to Pano2Room!

[Pano2Room: Novel View Synthesis from a Single Indoor Panorama (SIGGRAPH Asia 2024)](https://arxiv.org/abs/2408.11413).

## Quick Demo

[GS-result.mp4](demo/GS_render_video.mp4)
<img src='demo/input_panorama.png' width="30%" >
<img src="ckpts/Exp-syndney/MPI_rendered_views.gif" width="30%" >

### 1. Prepare
(1) Create a new conda environment specified in \<requirements.txt\>.
(2) Download pretrained weights in \<checkpoints\>.

### 2. Run demo
```
sh scripts/run_Pano2Room.sh
```
This demo converts 'input/input_panorama.jpg' to 3DGS and renders novel views as in 'output/'.

### (Optional) 1.5. Fine-tune Stable-Diffusion (SDFT)


## Cite our paper

If you find our work helpful, please cite our paper. Thank you!

ACM Reference Format:
```
Guo Pu, Yiming Zhao, and Zhouhui Lian. 2024. Pano2Room: Novel View Synthesis from a Single Indoor Panorama. In SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 3--6, 2024, Tokyo, Japan. ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/3680528.3687616
```


Instructions are updating...

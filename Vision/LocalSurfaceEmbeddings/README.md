# 3D Computer Vision Project Repo



## TODO: Implement LSE Paper
- [x] Implement `lse_compute` function
  - [x] convert radius(=3cm) to pixel unit
  - [x] compute normal vector of point cloud by using `open3d`
  - [x] Normalize lse vector to zero mean and unit variance
- [x] Generate Training Data for LSE Predictor
  - [ ] Generate Synthetic Images
    * Refer to [BlenderProc](https://arxiv.org/abs/1911.01911)
  - [x] Compute LSE for corresponding 3D points of each pixel
    * For synthetic images, we can know 3D point while generating
    * For real images, [T-Less Toolkit](https://github.com/thodan/t-less_toolkit) can compute 3D object models
- [x] Implement LSE Predictor
- [ ] Implement Object Detection
  * Mask-RCNN is used in original paper
- [ ] Implement Pose Estimation

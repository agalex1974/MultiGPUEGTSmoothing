# Multi GPU EGT Smoothing
<img src="/images/smooth_pipeline.jpg" alt="image" width="auto" height="auto"><br>
Left, reconstruction in Geomagic without smoothing, middle, after performing 300 iterations of smoothing on the point-cloud, right after performing 150 iterations on the mesh produced in the middle.<br>
All can be done using this software.

Perform Multi GPU EGT smoothing On a Point Cloud or a Mesh<br><br>
This is the code that perform multi-gpu smoothing on a PointCloud and also Normal Computation.<br>
This code is based on the paper:

Alexander Agathos, Philip Azariadis,<br>
Multi-GPU 3D k-nearest neighbors computation with application to ICP, point cloud smoothing and normals computation,<br>
Parallel Computing,<br>
Volume 121,<br>
2024,<br>
https://doi.org/10.1016/j.parco.2024.103093.<br><br>
The license is MIT so you can use it for Academic and Commercial purposes.<br><br>
Still I am perfecting it, the normal computation is still under development and it needs a lot of polishing up.<br><br>
As with all open source software you may use it at your own risk.<br><br>
It runs for the time being as KNNCUDA "file.ply" "number of neighbors" "number of gpus"<br><br>  

Copyringht Dr. Alexander Agathos.

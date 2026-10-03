# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\docker_arterial_gpu\Dockerfile (= /sp/v5_va/docker_arterial_gpu/Dockerfile in the build containers)
# neve-arterial-gpu: neve-arterial:py310 (CPU build, conda base env /opt/conda, py3.10 + vmtk) with CUDA
# torch + PyG. The RTX A5000 host driver (555.85, CUDA 12.5) is reached with `docker run --gpus all`;
# torch cu124 wheels bring their own CUDA 12.4 runtime (nvidia-* pip deps), so no CUDA base image.
#
# Build (Git Bash, from this folder):  docker build -t neve-arterial-gpu .
FROM neve-arterial:py310
USER root

# torch 2.6.0+cu124 replaces 2.6.0+cpu (the constraint torch==2.6.0 admits any local label);
# its nvidia-*/triton deps come from the PyTorch index, everything else from PyPI.
RUN /opt/conda/bin/python -m pip install --no-cache-dir -c /opt/constraints.txt \
        --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple \
        torch==2.6.0+cu124

# PyG compiled extensions rebuilt against CUDA (torch_geometric itself is pure python, kept).
RUN /opt/conda/bin/python -m pip install --no-cache-dir -c /opt/constraints.txt --force-reinstall --no-deps \
        -f https://data.pyg.org/whl/torch-2.6.0+cu124.html \
        pyg_lib==0.4.0+pt26cu124 torch_scatter==2.1.2+pt26cu124 torch_sparse==0.6.18+pt26cu124 \
        torch_cluster==1.6.3+pt26cu124 torch_spline_conv==1.2.2+pt26cu124

RUN /opt/conda/bin/python -c "import numpy, vtk, torch, torch_geometric, torch_cluster, torch_scatter, nnunetv2; from vmtk import vtkvmtk; import arterial.run.processor; print('numpy', numpy.__version__, 'torch', torch.__version__, 'built for cuda', torch.version.cuda, 'pyg', torch_geometric.__version__)"

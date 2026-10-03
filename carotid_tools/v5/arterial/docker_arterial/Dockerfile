# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\docker_arterial\Dockerfile (= /sp/v5_va/docker_arterial/Dockerfile in the build containers)
# neve-arterial (tag neve-arterial:latest, image 57bc8d21fc6b, built 2026-10-01): CPU-only runtime for the
# Arterial head-and-neck CTA pipeline (https://github.com/FLOWCAT-CV/arterial @3ff58ab,
# PolyForm-Noncommercial-1.0.0 code, CC BY-NC 4.0 weights from Zenodo 10.5281/zenodo.22694951 --
# mounted at run time at /models, never baked in).
#
# Base: neve-build-vmtk (mambaorg/micromamba + libGL/X libs already apt-installed).
# Arterial needs Python >= 3.11 and VMTK (conda-only), so a separate env "arterial" is created:
# run python as `micromamba run -n arterial python`. No CUDA anywhere: torch from the PyTorch CPU
# index, PyG extensions from the +cpu index.
# NOTE: the conda create step takes ~25-35 min on this machine (slow linking of qt-main under load).
# Dockerfile.py310 (tag neve-arterial:py310) is an equivalent faster-to-build variant that reuses the
# py3.10 base env of neve-build-vmtk (same pip pins; run python as /opt/conda/bin/python).
#
# Build (Git Bash, from this folder):  docker build -t neve-arterial .
FROM neve-build-vmtk:latest

USER root
ARG ARTERIAL_COMMIT=3ff58ab31a14f0416287cc66ceacca14b1918481

# 1. conda env: python 3.11 + VMTK 1.5.0 (brings vtk-base 9.2.x) + numpy 1.26 (vtk/vmtk built against numpy 1.x)
RUN micromamba create -y -n arterial -c conda-forge \
        python=3.11 "vmtk=1.5.0" "libitk=5.3.*" "numpy=1.26.*" && \
    micromamba clean -a -y

# 2. pip stack (CPU wheels, pinned). numpy is held at the conda version through a constraints file,
#    and vtk is NOT pip-installed (it would shadow the conda vtk that vmtk is linked against).
COPY constraints.txt /opt/constraints.txt
RUN micromamba run -n arterial pip install --no-cache-dir -c /opt/constraints.txt \
        --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN micromamba run -n arterial pip install --no-cache-dir -c /opt/constraints.txt \
        -f https://data.pyg.org/whl/torch-2.6.0+cpu.html \
        torch_geometric==2.6.1 pyg_lib==0.4.0+pt26cpu torch_scatter==2.1.2+pt26cpu torch_sparse==0.6.18+pt26cpu \
        torch_cluster==1.6.3+pt26cpu torch_spline_conv==1.2.2+pt26cpu
ARG NNUNET_VERSION=2.6.2
RUN micromamba run -n arterial pip install --no-cache-dir -c /opt/constraints.txt \
        nnunetv2==${NNUNET_VERSION} monai==1.4.0 torchio==0.20.4 connected-components-3d==3.22.0 \
        nibabel networkx scipy scikit-image matplotlib psutil

# 3. Arterial source at a pinned commit, installed without deps (deps are pinned above).
ADD https://github.com/FLOWCAT-CV/arterial/archive/${ARTERIAL_COMMIT}.tar.gz /opt/arterial_src.tar.gz
RUN mkdir -p /opt/arterial && tar xzf /opt/arterial_src.tar.gz -C /opt/arterial --strip-components=1 && \
    rm /opt/arterial_src.tar.gz && \
    micromamba run -n arterial pip install --no-cache-dir --no-deps -e /opt/arterial && \
    chmod -R a+rwX /opt/arterial

# CPU wrapper (an early copy; run_arterial.sh always runs the current /sp/v5_va/arterial_cpu_run.py).
COPY arterial_cpu_run.py /opt/arterial_cpu_run.py

ENV arterial_dir=/opt/arterial/arterial \
    ARTERIAL_MODELS_DIR=/models \
    nnUNet_raw=/tmp/nnunet_raw nnUNet_preprocessed=/tmp/nnunet_pre nnUNet_results=/tmp/nnunet_res \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

# Smoke import at build time (fails the build if vmtk/vtk/torch/PyG/nnunet do not co-exist).
RUN micromamba run -n arterial python -c "import numpy, vtk, torch, torch_geometric, torch_cluster, nnunetv2, monai, torchio, cc3d, nibabel; from vmtk import vtkvmtk; import arterial.run.processor; print('numpy', numpy.__version__, 'vtk', vtk.vtkVersion.GetVTKVersion(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'pyg', torch_geometric.__version__, 'monai', monai.__version__)"

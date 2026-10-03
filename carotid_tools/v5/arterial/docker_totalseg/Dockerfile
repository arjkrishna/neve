# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\docker_totalseg\Dockerfile (= /sp/v5_va/docker_totalseg/Dockerfile in the build containers)
# neve-totalseg-gpu: TotalSegmentator 2.18.0 on top of neve-arterial-gpu
# (conda base env /opt/conda, py3.10, torch 2.6.0+cu124, nnunetv2 2.6.2, nibabel 5.4.2, SimpleITK 2.5.6, vmtk 1.5.0).
#
# torch is NOT touched: TotalSegmentator and its missing import-time deps go in with --no-deps, pinned.
# Left out on purpose (only used by --preview / aorta / pulmonary reports / DICOM-RTSTRUCT, never by a NIfTI
# "total" run): xvfbwrapper, imgkit, fury, dipy, pyarrow, blosc, python-gdcm, rt_utils, highdicom, radiomics.
# dicom2nifti is needed only because totalsegmentator/nnunet.py imports dicom_io at module level.
#
# Weights download on first use into $TOTALSEG_HOME_DIR/nnunet/results; mount a persistent dir there:
#   -v "D:\\Arjun\\neve\\carotid_data\\va_sources\\totalseg_weights:/ts_home"
# and put {"send_usage_stats": false, ...} in /ts_home/config.json BEFORE the first run (TotalSegmentator
# otherwise posts anonymous usage statistics to stats.totalsegmentator.com).
#
# Build (Git Bash, from this folder):  docker build -t neve-totalseg-gpu .
FROM neve-arterial-gpu
USER root

RUN /opt/conda/bin/python -m pip install --no-cache-dir --no-deps \
        TotalSegmentator==2.18.0 dicom2nifti==2.6.2 pydicom==3.0.2 xmltodict==1.0.4

ENV TOTALSEG_HOME_DIR=/ts_home

RUN /opt/conda/bin/python -c "import torch, nnunetv2, nibabel, SimpleITK, totalsegmentator; import totalsegmentator.python_api, totalsegmentator.nnunet; from importlib.metadata import version; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'nnunetv2', version('nnunetv2'), 'TotalSegmentator', version('TotalSegmentator'))" \
 && (/opt/conda/bin/python -m pip check || true)

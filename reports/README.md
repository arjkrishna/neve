# reports/

Two PDFs on the collision-mesh pipeline, built from the analysis in
`MESHING_PIPELINE_ANALYSIS.md` and the per-anatomy reports of the v2/v3 builds.

| file | what it is |
|---|---|
| `Meshing_Pipeline_Analysis.pdf` | the pipeline stage by stage, where fidelity is lost, the defects, the VMTK toolkit, what was changed (v2, v3) and what it gained |
| `Mesh_Construction_v1_v2_v3.pdf` | the three constructions and their mathematics; what changes for calibre, lumen, MISR and construction; how each relates to the host test anatomy |
| `figs/` | every figure in the two documents (`make/figures.py`) plus renders copied from `saved/` |
| `make/` | builders: `figures.py` (matplotlib), `pdfkit.py` (reportlab layer), `report_pipeline.py`, `report_versions.py` |

## Rebuilding

reportlab is not in the host environment; it was installed into a scratch
directory rather than into conda. Point the builders at it:

    pip install --target <some_dir> reportlab
    python reports/make/figures.py
    REPORTLAB_PYLIB=<some_dir> python reports/make/report_pipeline.py
    REPORTLAB_PYLIB=<some_dir> python reports/make/report_versions.py

`figures.py` reads `saved/mesher_probe/lumen_v1.json`, `topbrain_data/label_necks.json`
and every `mesh_v2.json` / `mesh_v3.json` under the v2/v3 anatomy folders; the
tube-erosion, SOFA-timing and budget numbers are the measured values restated
in the script.

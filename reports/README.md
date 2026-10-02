# reports/

Two PDFs on the collision-mesh pipeline, built from the analysis in
`MESHING_PIPELINE_ANALYSIS.md` and the per-anatomy reports of the v2/v3 builds.

| file | what it is |
|---|---|
| `Meshing_Pipeline_Analysis.pdf` | the pipeline stage by stage, where fidelity is lost, the defects, the VMTK toolkit, what was changed (v2, v3) and what it gained |
| `Mesh_Construction_v1_v2_v3.pdf` | the three constructions and their mathematics; what changes for calibre, lumen, MISR and construction; how each relates to the host test anatomy |
| `slides/` | five 16:9 PNGs for presenting: `slide_tube_v1_vs_v2` (one vessel through both meshers, voxel by voxel), `slide_tube_vanishing` (why thin vessels drop out of v1), and three mesh zooms on real anatomies — `slide_zoom_v1_vs_v2`, `slide_zoom_v2_vs_v3`, `slide_zoom_v1_vs_v3` |
| `slides_v2/` | the same five slides with the grey sub-heading, the footer paragraph and the wall-ray ratios removed, plus `slide_mr013_v1_vs_v2` and `slide_mr013_v2_vs_v3` (topcow_mr_013 right ICA, two versions side by side at matched scale) — title, panels and captions unchanged (`make/slide_tube_voxels_v2.py`, `make/slides_mesh_zoom_v2.py`, `make/slide_mr013_v1_vs_v2.py`, `make/slide_mr013_v2_vs_v3.py`) |
| `slides_v3/` | the first five slides again, this time keeping the grey sub-heading and dropping only the footer (`make/slide_tube_voxels_v3.py`, `make/slides_mesh_zoom_v3.py`) |
| `figs/` | every figure in the two documents (`make/figures.py`) plus renders copied from `saved/` |
| `make/` | builders: `figures.py` (matplotlib), `pdfkit.py` (reportlab layer), `report_pipeline.py`, `report_versions.py`; slides: `slide_tube_voxels.py` (host), `slides_mesh_zoom.py` (container, reads the baked v1/v2/v3 meshes) |

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

## Slides

    python reports/make/slide_tube_voxels.py                                   # host: the two tube slides
    bash carotid_tools/run_container.sh python3 /opt/eve_training/saved/mesher_probe/slides_mesh_zoom.py   # container: the three zooms

    python reports/make/slide_tube_voxels_v2.py                                # same two, no sub-heading or footer
    bash carotid_tools/run_container.sh python3 /opt/eve_training/saved/mesher_probe/slides_mesh_zoom_v2.py  # same three, into slides_v2/
    python reports/make/slide_mr013_v1_vs_v2.py                                # crops two existing renders, no meshing
    python reports/make/slide_mr013_v2_vs_v3.py                                # same, from the v2 and v3 renders

    python reports/make/slide_tube_voxels_v3.py                                # sub-heading kept, footer dropped
    bash carotid_tools/run_container.sh python3 /opt/eve_training/saved/mesher_probe/slides_mesh_zoom_v3.py  # same, into slides_v3/

The zoom slides pick their anatomies from the measured data — the worst v1
lumens in `saved/mesher_probe/lumen_v1.json` and the strongest real-surface
shape ratios in the v3 `mesh_v3.json` reports — and centre each window where
the difference lives.

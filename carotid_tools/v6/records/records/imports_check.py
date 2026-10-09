# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records\imports_check.py (= /sp/v6/records/imports_check.py in the build containers)
import ast,os,sys,json,re,sysconfig,pkgutil
T="/sp/v6/records/tools_v6"
rows=[l.rstrip("\n").split("\t") for l in open(T+"/FILES.tsv")][1:]
sel={r[1] for r in rows}
selmods={os.path.basename(r[1])[:-3] for r in rows if r[1].endswith(".py")}
std=set(sys.stdlib_module_names)
third={"numpy","scipy","vtk","pyvista","nibabel","SimpleITK","skimage","trimesh","pymeshlab","matplotlib","pandas","networkx","torch",
       "pydicom","yaml","h5py","sklearn","meshio","igl","open3d","tqdm","Sofa","SofaRuntime","pyhull","shapely","numba","pulp","ortools",
       "vmtk","itk","fast_simplification","pyacvd","rtree","PIL","imageio","cv2","mcubes","psutil","totalsegmentator","nnunetv2","pymcubes","highspy","mip"}
repo={"carotid_tools","topbrain_tools","eve","eve_bench","graft_three","graft_v4","graft_siphon","sdf_mesher","sdf_union","sdf_union_v4",
      "bake_meshes_v4","extend_ica","analyze_bifurcations","mask_to_surface","skeleton_centerline","topo_audit","anatomies_v4"}
unres={}
for r in rows:
    if not r[1].endswith(".py"): continue
    src=open("/sp/"+r[1],encoding="utf-8",errors="replace").read()
    try: t=ast.parse(src)
    except SyntaxError as e: unres[r[1]]=["SYNTAX "+str(e)]; continue
    mods=set()
    for n in ast.walk(t):
        if isinstance(n,ast.Import): mods.update(x.name.split(".")[0] for x in n.names)
        elif isinstance(n,ast.ImportFrom) and n.module and n.level==0: mods.add(n.module.split(".")[0])
    bad=sorted(m for m in mods if m not in std and m not in third and m not in repo and m not in selmods)
    if bad: unres[r[1]]=bad
print(json.dumps(unres,indent=0))

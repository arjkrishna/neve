from setuptools import setup, find_packages
from glob import glob

visu_mesh_data = glob("eve/visualisation/meshes/*")
simulation_util_data = glob("eve/intervention/simulation/util/*")
setup(
    name="eve",
    version="0.2",
    packages=find_packages(),
    data_files=[
        (
            "visu_mesh_data",
            visu_mesh_data,
        ),
        (
            "simulation_util_data",
            simulation_util_data,
        ),
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "scikit-image",
        "pyvista",
        "meshio",
        "PyOpenGL",
        "pygame",
        "matplotlib",
        "opencv-python",
        "gymnasium",
        "pyyaml",
        "wget"
    ],
)

from setuptools import setup, find_packages
package_name = "nlp_src"
setup(
    name=package_name,
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=True
)

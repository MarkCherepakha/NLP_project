from setuptools import setup, find_packages
package_name = "app"
setup(
    name=package_name,
    version="0.0.1",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    zip_safe=True
)

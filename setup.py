from setuptools import setup

setup(
    name="ggpymanager",
    version="1.0",
    description="",
    author="Robert Maiwald",
    author_email="rmaiwald@iup.uni-heidelberg.de",
    packages=["ggpymanager"],
    install_requires=[
        "scipy",
        "numpy",
        "time",
        "pathlib",
        "subprocess",
        "dataclass",
        "pytest",

    ],  # external packages as dependencies
    scripts=[],
)

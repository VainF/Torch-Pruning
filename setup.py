"""Setup script for Torch-Pruning package."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies
requirements = ["torch>=2.0", "numpy"]

setuptools.setup(
    name="torch-pruning",
    version="1.6.1",
    author="Gongfan Fang",
    author_email="gongfan@u.nus.edu",
    description="Towards Any Structural Pruning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VainF/Torch-Pruning",
    project_urls={
        "Bug Reports": "https://github.com/VainF/Torch-Pruning/issues",
        "Source": "https://github.com/VainF/Torch-Pruning",
        "Documentation": "https://github.com/VainF/Torch-Pruning/wiki",
    },
    packages=setuptools.find_packages(exclude=["tests*", "examples*", "reproduce*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=requirements,
    python_requires=">=3.7",
    keywords="pytorch, pruning, neural networks, deep learning, optimization",
    zip_safe=False,
)

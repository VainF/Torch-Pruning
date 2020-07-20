import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-pruning",
    version="0.2.2",
    author="Gongfan Fang",
    author_email="fgf@zju.edu.cn",
    description="A pytorch toolkit for structured neural network pruning and layer dependency maintaining.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VainF/Torch-Pruning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['torch'],
    python_requires='>=3.6',
)
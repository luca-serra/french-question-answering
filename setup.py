import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="illuin",
    version="0.1",
    author="Luca Serra",
    author_email="luca.serra@student.ecp.fr",
    description="Illuin project to find the best context in Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luca-serra/illuin-fquad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7.2',
)
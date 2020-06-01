from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pypyp",
    version="0.3.2",
    author="Shantanu Jain",
    author_email="hauntsaninja@gmail.com",
    description="Easily run Python at the shell! Magical, but never mysterious.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hauntsaninja/pyp",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
    py_modules=["pyp"],
    entry_points={"console_scripts": ["pyp=pyp:main"]},
    install_requires=["astunparse; python_version<'3.9'"],
    python_requires=">=3.6",
)

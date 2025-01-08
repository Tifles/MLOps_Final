from setuptools import setup, find_packages

setup(
    name="credit_default_analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[],
    author="Tifles",
    description="Credit default analysis package",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.com/learning3784447/mlops-hw-1",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

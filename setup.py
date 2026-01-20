"""Setup script for organoid-media-ml package."""

from setuptools import setup, find_packages

setup(
    name="organoid-media-ml",
    version="1.0.0",
    description="ML system for predicting organoid culture media formulations",
    author="Organoid Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "requests>=2.31",
        "beautifulsoup4>=4.12",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        "viz": [
            "matplotlib>=3.7",
            "seaborn>=0.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=train:main",
        ],
    },
)

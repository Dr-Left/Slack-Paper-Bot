from setuptools import setup, find_packages

setup(
    name="ai-pod",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "loguru",
        "transformers>=4.30.0",
        "adapters>=0.1.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "slack_sdk>=3.20.0",
        "openai>=1.0.0",
    ],
)


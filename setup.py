from setuptools import find_packages, setup

print(
    "Installing EL-RL. Dependencies should already be installed with the provided conda env."
)

setup(
    name="efrl",
    version="0.1.0",
    packages=find_packages(),
    description="Self-supervised Learning of Agent-Aware Representations for Improved RL",
    author="Manuel Serra Nunes",
    author_email="manuelserranunes@tecnico.ulisboa.pt",
)

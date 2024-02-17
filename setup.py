from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="vision_lab",
        packages=find_packages(include=["vision_lab*"]),
        package_data={
            "vision_lab.utils.fonts": ["*.ttf"],
        },
    )

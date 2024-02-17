from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="vision_lab",
        install_requires=[
            "groundingdino",
            "segment_anything",
            "aot",
            "seg_and_track_anything",
        ],
        packages=find_packages(include=["vision_lab*"]),
        package_data={
            "vision_lab.utils.fonts": ["*.ttf"],
        },
    )

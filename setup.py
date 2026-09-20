from setuptools import setup

package_name = "rml_joy_teleop"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mahua",
    maintainer_email="mahua@example.com",
    description="Xbox joystick teleop for RML63",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "xbox_xyz_teleop = rml_joy_teleop.xbox_xyz_teleop:main",
        ],
    },
)

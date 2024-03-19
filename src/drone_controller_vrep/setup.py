from setuptools import find_packages, setup

package_name = 'drone_controller_vrep'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zain',
    maintainer_email='zaahm18@student.sdu.dk',
    description='Drone controller for V-REP simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'drone_feedback_controller = drone_controller_vrep.drone_feedback_controller:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'nmpc_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='administrator',
    maintainer_email='nnyimm001@myuct.ac.za',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           'record = nmpc_pkg.record:main',
           'gotostart = nmpc_pkg.gotostart:main',
           'nmpc_controller_node = nmpc_pkg.controller_node:main',
           'navigan_nmpc_controller_node = nmpc_pkg.navigan_controller_node:main',
           'switching_controller_node = nmpc_pkg.switching_controller_node:main'
        ],
    },
)

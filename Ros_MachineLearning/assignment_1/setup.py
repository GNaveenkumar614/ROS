from setuptools import find_packages, setup

package_name = 'assignment_1'

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
    maintainer='nani',
    maintainer_email='naninaveen614@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'humanbrain = assignment_1.humanbrain:main',
        'height = assignment_1.height:main',
        'fruit = assignment_1.fruit:main',  
        'penguin = assignment_1.penguin:main',
        'breastcancer = assignment_1.breastcancer:main',
        'houseprice = assignment_1.houseprice:main',
        'irisclassifier = assignment_1.irisclassifier:main'
    ],
}  
)

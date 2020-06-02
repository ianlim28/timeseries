from setuptools import find_packages, setup

setup(name='timeseries',
      version='0.1',
      description='Functions for timeseries analysis',
      url='https://github.com/ianlim28/timeseries',
      author='ianlim',
      author_email='ianlim28@hotmail.com',
      license='MIT',
      packages=find_packages(),
      package_dir={"": "src"},
      zip_safe=False)
from setuptools import setup, find_packages

setup(name='ketos',
      version='1.0.0',
      description="Python package for developing  deep-learning-based models for the detection and classification of underwater sounds",
      # TODO: define a function readme() that reads the contents of a readme file
      # long_description=readme(),
      url='https://gitlab.meridian.cs.dal.ca/data_analytics_dal/packages/ketos',
      author='Oliver Kirsebom, Fabio Frazao',
      author_email='oliver.kirsebom@dal.ca, fsfrazao@dal.ca',
      license='GNU General Public License v3.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'tables',
          'scipy',
          'pandas',
          'tensorflow',
          ],
      setup_requires=['pytest-runner', ],
      tests_require=['pytest', ],
      include_package_data=True,
      zip_safe=False)

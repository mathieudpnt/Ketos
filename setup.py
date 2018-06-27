from setuptools import setup

setup(name='sound_classification',
      version='0.0.1',
      description="MERIDIAN's internal package for sound cladssification projects",
      # TODO: define a function readme() that reads the contents of a readme file
      # long_description=readme(),
      url='http://github.com/fsfrazao/Trees',
      author='Oliver Kirsebom, Fabio Frazao',
      author_email='oliver.kirsebom@dal.ca, fsfrazao@dal.ca',
      license='GNU General Public License v3.0',
      packages=['sound_classification'],
      install_requires=[
          'numpy',
          'tables',
          'scipy',
          'opencv-python',
          'pandas',
          'tensorflow'
          ],
      setup_requires=['pytest-runner',],
      tests_require=['pytest',],
      include_package_data=True,
      zip_safe=False)
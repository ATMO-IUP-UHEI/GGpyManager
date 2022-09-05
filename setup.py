from setuptools import setup

setup(
   name='ggpymanager',
   version='0.1',
   description='',
   author='Robert Maiwald',
   author_email='rmaiwald@iup.uni-heidelberg.de',
   packages=['ggpymanager'],  #same as name
   install_requires=["scipy", "numpy"], #external packages as dependencies
   scripts=[]
)
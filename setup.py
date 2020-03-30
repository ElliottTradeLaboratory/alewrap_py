import os
import sys
import multiprocessing
import subprocess
from setuptools import setup
from distutils.command.build import build as DistutilsBuild

class Build(DistutilsBuild):
    def run(self):
        cores_to_use = max(1, multiprocessing.cpu_count() - 1)
        cur_dir = os.getcwd()
        os.chdir('./alewrap_py/xitari')
        print('change directory to ', os.getcwd())
        cmd1 = ['cmake', '.']
        cmd2 = ['make']
        self._execute_cmd(cmd1)
        self._execute_cmd(cmd2)
        os.chdir(cur_dir)
        print('change directory to setup root')
        DistutilsBuild.run(self)
    def _execute_cmd(self, cmd):
        print('execute', *cmd)
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not build atari-py: %s. (HINT: are you sure cmake is installed? You might also be missing a library. Atari-py requires: zlib [installable as 'apt-get install zlib1g-dev' on Ubuntu].)\n" % e)
            raise
        except OSError as e:
            sys.stderr.write("Unable to execute '{}'. HINT: are you sure `make` is installed?\n".format(' '.join(cmd)))
            raise

setup(
    name='alewrap_py',
    version='1.0.0',
    author="ElliottTradeLaboratory",
    author_email="elliott.trade.laboratory@gmail.com",
    description="alewrap for Python",
    license="GPL",
    url="https://github.com/ElliottTradeLaboratory/alewrap_py",
    install_requires=['opencv-python>=3.3.0.10',
                      'sk-video>=1.1.8'],
    packages=['alewrap_py'],
    package_data={
        'alewrap_py': [
            'alewrap_py/*.py',
           './*/libxitari.so',
           './*/README.md',
        ],
    },
    cmdclass={'build': Build},
)

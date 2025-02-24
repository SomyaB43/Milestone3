import subprocess
from distutils.command.build import build as _build
import setuptools

class build(_build):  # pylint: disable=invalid-name
    sub_commands = _build.sub_commands + [('CustomCommands', None)]

class CustomCommands(setuptools.Command):

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print('Running command: %s' % command_list)
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout_data, _ = p.communicate()
        print('Command output: %s' % stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)

# Add more libraries if needed, such as `torch`
CUSTOM_COMMANDS = [
    ['pip', 'install', 'torch'],
    ['pip', 'install', 'opencv-python'],
    ['pip', 'install', 'apache-beam[gcp]'],
    ['pip', 'install', 'pandas'],
    ['pip', 'install', 'google-cloud-pubsub'],
    ['pip', 'install', 'ultralytics'],
    # Add any other dependencies your script needs
]

REQUIRED_PACKAGES = []

setuptools.setup(
    name='pedestrian-detection',
    version='0.0.1',
    description='Pedestrian detection using YOLO and MiDaS depth estimation.',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    cmdclass={'build': build, 'CustomCommands': CustomCommands}
)

from setuptools import setup

setup(name='nimbleGPT',
      version='0.0.1',
      author='Tristan Heywood',
      packages=['nimblegpt'],
      description='A re-implementation of GPT',
      license='MIT',
      install_requires=[
            'jax',
      ],
)

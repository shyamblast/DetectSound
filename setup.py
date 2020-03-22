from setuptools import setup


with open('README.md', 'r') as fd:
    long_description = fd.read()

setup(
    name='DetectSound',
    version='0.1.0',
    description='Python module containing general-purpose detectors for ' +
                'extracting different types of sounds from acoustic ' +
                'recordings.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shyamblast/DetectSound',
    author='Shyam Madhusudhana',
    author_email='shyamm@cornell.edu',
    license='MIT License',
    keywords='audio sound detect extract mining',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        ],
    packages=['detectsound'],
    install_requires=[
        'numpy',
        'scipy'
        ],
    #zip_safe=False
)

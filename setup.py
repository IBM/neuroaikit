"""A setuptools based setup module.
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # PyPI: https://pypi.org/project/sampleproject/
    # $ pip install sampleproject
    name='neuroaikit',  # Required
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    #package_dir={'': 'neuroaikit'},  # Optional
    #packages=find_packages(where='neuroaikit'),  # Required
    packages=find_packages(include=['neuroaikit',]),  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    version='0.0.1',  # Required

    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='NeuroAIKit',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional
    # This should be a valid link to your project's main homepage.
    #url='https://github.com/pypa/sampleproject',  # Optional
    author='IBM Research',  # Optional
    # author_email='author@example.com',  # Optional

    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.4, <4',

    # https://packaging.python.org/en/latest/requirements.html
    #install_requires=['peppercorn'],  # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        'tf': ['tensorflow'],
        'th': ['torch'],
    },

    # # If there are data files included in your packages that need to be
    # # installed, specify them here.
    # package_data={  # Optional
    #     'sample': ['package_data.dat'],
    # },
    #
    # # Although 'package_data' is the preferred approach, in some case you may
    # # need to place data files outside of your packages. See:
    # # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    # #
    # # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional
    #
    # # To provide executable scripts, use entry points in preference to the
    # # "scripts" keyword. Entry points provide cross-platform support and allow
    # # `pip` to create the appropriate form of executable for the target
    # # platform.
    # #
    # # For example, the following would provide a command called `sample` which
    # # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },

)

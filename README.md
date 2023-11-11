# lyzr

Lyzr AI is a set of super abstracted LLM SDKs for rapid generative AI application development (RAD) that comes with Lyzr Enterprise Hub – a control center with an AI-only Data Lake and IAM for administering the AI applications built with Lyzr SDKs. Available both as open-source and enterprise SDKs.

Lyzr SDKs helps you build all your favorite GenAI SaaS products as enterprise applications in minutes. It is the enterprise alternative to popular in-demand Generative AI SaaS products like Mendable, PDF.ai, Chatbase.co, SiteGPT.ai, Julius.ai and more.

[![PyPI - Version](https://img.shields.io/pypi/v/lyzr.svg)](https://pypi.org/project/lyzr/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lyzr.svg)](https://pypi.org/project/lyzr)

-----

**Table of Contents**

- [Installation](#installation)
- [Building from Source](#building-from-source)
- [License](#license)

## Installation

You can install the `lyzr` package directly from PyPI:

```console
pip install lyzr
```

## Building from Source

If you prefer to build the `lyzr` package from source, you'll need to have Python installed along with `setuptools` and `wheel`. 

### Steps to Build:

1. Clone the repository or download the source code.
2. Navigate to the root directory of the project (where `setup.py` is located).
3. Run the following commands:

```console
# Ensure setuptools and wheel are installed
pip install setuptools wheel

# Build the package
python setup.py sdist bdist_wheel
```

This will generate a `dist` directory containing the built package files.

### Installing the Built Package:

Once you've built the package, you can install it using pip:

```console
cd dist/
pip install lyzr-[version]-py3-none-any.whl
```

Replace `[version]` with the actual version of the package you have built.

## License

`lyzr` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

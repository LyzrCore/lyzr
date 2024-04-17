<p align="center">
<img src="https://raw.githubusercontent.com/LyzrCore/lyzr/214f79adf5adf25091479ba5b5a8c18064a26b31/assets/logo.svg" width="200" alt="Lyzr Logo" />
</p>
<p align="center">
The simplest agent framework to build <b>Private & Secure</b> GenAI apps faster
</p>

<p align="center">
<a href="https://pypi.org/project/lyzr/">
<img src="https://img.shields.io/pypi/v/lyzr.svg" />
</a>
<a href="https://pypi.org/project/lyzr/">
<img src="https://img.shields.io/pypi/pyversions/lyzr.svg" />
</a>
<a href="https://opensource.org/licenses/MIT">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</a>
<a href="https://anybodycanai.slack.com/ssb/redirect">
<img src="https://img.shields.io/badge/Slack-Join%20Chat-blue?style=flat&logo=slack" />
</a>
<a href="https://discord.gg/P6HCMQ9TRX">
<img src="https://img.shields.io/badge/Discord-Join%20Server-blue?style=flat&logo=discord" />
</a>
</p>

Lyzr is a low-code agent framework with an agentic approach to building generative AI applications. Its fully integrated agents come with pre-built RAG pipelines, allowing you to build and launch in minutes.

Lyzr SDKs helps you build all your favorite GenAI SaaS products as enterprise applications in minutes. It is the enterprise alternative to popular in-demand Generative AI SaaS products like Mendable, PDF.ai, Chatbase.co, SiteGPT.ai, Julius.ai and more.

---

**Table of Contents**

- [Key Features](#key-features)
- [Personas](#personas)
- [Links](#links)
- [Installation](#installation)
- [Building from Source](#building-from-source)
  - [Steps to Build](#steps-to-build)
  - [Installing the Built Package](#installing-the-built-package)
- [Example](#example)
  - [Launch an agent in just few lines of code](#launch-an-agent-in-just-few-lines-of-code)
- [License](#license)

## Key Features

- **Lyzr's Pre-built Agents**: Deploy in minutes
  - Chat agent
  - Knowledge search
  - RAG powered apps
  - QA bot
  - Data Analysis
  - Text-to-SQL
- **Lyzr Automata**: The multi-agent automation platform
- **Free Tools**
  - Lyzr Parse
  - Magic Prompts
  - Prompt Studio
  - Knowledge Base
- **Cookbooks**
  - Lyzr + Weaviate
  - Lyzr + Streamlit
  - Lyzr + Qdrant
  - Lyzr + Vellum

## Personas

- **Developers**: you will love the simplicity of the framework and appreciate the ability to build generative AI apps rapidly.

- **CTOs, CPOs**: integrate generative AI features into your apps seamlessly with local SDKs and private APIs, all with your in-house tech team. The required learning curve to build on Lyzr is literally just a few minutes.

- **CIOs**: introduce generative AI to your enterprise with the comfort of 100% data privacy and security as Lyzr runs locally on your cloud. And Lyzr's AI Management System (AIMS) makes it easy to manage agents, monitor events logs, build using AI studios, and even help your team learn generative AI with the in-built Lyzr academy.

## Links

- [Documentation](https://docs.lyzr.ai/)
- [Demos](https://www.lyzr.ai/demos/)
- [Blog](https://www.lyzr.ai/blog/)
- [Pricing](https://www.lyzr.ai/pricing/)
- [Enterprise](https://www.lyzr.ai/enterprise/)

## Installation

You can install the `lyzr` package directly from [PyPI](https://pypi.org/project/lyzr/):

```console
pip install lyzr
```

## Building from Source

If you prefer to build the `lyzr` package from source, you'll need to have Python installed along with `setuptools` and `wheel`.

### Steps to Build

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

### Installing the Built Package

Once you've built the package, you can install it using pip:

```console
cd dist/
pip install lyzr-[version]-py3-none-any.whl
```

Replace `[version]` with the actual version of the package you have built.

## Example

### Launch an agent in just few lines of code

1. Install the Lyzr library.

```python
pip install lyzr
```

2. Import the necessary agent module. Here we are choosing ChatBot module.

```python
from lyzr import ChatBot
```

3. Provision the chatbot with just 1-line of code. The RAG pipeline runs in the background.

```python
my_chatbot = ChatBot.pdf_chat(input_files=["pdf_file_path"])
```

4. That's it. Just query and start chatting with your chatbot.

```python
response = chatbot.chat("Your question here")
```

## License

`lyzr` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

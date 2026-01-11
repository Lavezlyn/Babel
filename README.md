# Babel: Emergent Linguistic Divergence in Multi-Agent Resource Exchange

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-paper-orange.svg)](https://github.com/Lavezlyn/Babel/blob/main/paper.pdf)

> **How does a single linguistic communication system spontaneously give rise to multiple languages under natural conditions?**

This repository implements a computational framework for studying the emergence of linguistic divergence using Large Language Models (LLMs) in a multi-agent resource exchange game. Inspired by the Tower of Babel narrative, we investigate how strategic cooperation pressures can induce dialect formation from a shared starting language.

## 🌟 Overview

The biblical narrative of the Tower of Babel attributes linguistic diversification to a one-time, externally imposed act of coercion. However, empirical evidence reveals that language differentiation is a gradual process that unfolds continuously through social interaction. This project provides a computational model demonstrating how linguistic divergence can emerge spontaneously under selective cooperation with uncertain partners.

### Key Features

- **Multi-Agent Framework**: 4-player, 2-team resource exchange game with LLM agents
- **Emergent Communication**: Agents develop novel linguistic forms as group markers
- **Strategic Environment**: In-group exchange is beneficial; out-group exchange is costly
- **Alien Language Constraint**: Communication uses invented tokens, free from natural-language priors
- **Flexible LLM Backends**: Support for OpenAI, vLLM, DeepSeek, and dummy backends

## 📋 Table of Contents

- [Babel: Emergent Linguistic Divergence in Multi-Agent Resource Exchange](#babel-emergent-linguistic-divergence-in-multi-agent-resource-exchange)
  - [🌟 Overview](#-overview)
    - [Key Features](#key-features)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Core Contributions](#-core-contributions)
  - [🎮 The Resource Exchange Game](#-the-resource-exchange-game)
    - [Game Setup](#game-setup)
    - [Round Structure](#round-structure)
    - [Exchange Dynamics](#exchange-dynamics)
    - [Scoring](#scoring)
    - [Communication Constraint](#communication-constraint)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Install Dependencies](#step-2-install-dependencies)
    - [Step 3: Configure API Keys (Optional)](#step-3-configure-api-keys-optional)
  - [⚡ Quick Start](#-quick-start)
    - [Run with Dummy Backend (Testing)](#run-with-dummy-backend-testing)
    - [Run with OpenAI Backend](#run-with-openai-backend)
    - [Run with vLLM Backend (Local Server)](#run-with-vllm-backend-local-server)
  - [📖 Usage](#-usage)
    - [Command Line Interface](#command-line-interface)
      - [Options](#options)
    - [Programmatic Usage](#programmatic-usage)
    - [Output Format](#output-format)
  - [📁 Project Structure](#-project-structure)
  - [⚙️ Configuration](#️-configuration)
  - [🔬 Research Background](#-research-background)
  - [📊 Results](#-results)
  - [📝 Citation](#-citation)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## 🎯 Core Contributions

1. **Framework for Divergence**: A novel multi-agent LLM framework that models language speciation as a consequence of selective cooperation under partner uncertainty, moving beyond cooperative-only paradigms.

2. **Evidence of Neologism**: Empirical evidence that LLMs can exhibit lexical creativity, generating new tokens and systematic usage patterns that function as group markers without explicit instruction.

3. **Dynamics of Separation**: Analysis of how quickly and distinctly emergent dialects diverge from a shared baseline as a function of interaction structure and the cost of cross-group exchange.

## 🎮 The Resource Exchange Game

### Game Setup

- **Players**: 4 agents partitioned into 2 teams of size 2
- **Duration**: $R$ rounds (default: 14)
- **Pairing**: Random pairing each round; agents may face either a teammate or an opponent
- **Partner Uncertainty**: Agents do not directly observe the partner's team identity during interaction

### Round Structure

Each round consists of four stages:

1. **Chat**: Agents communicate for a fixed number of turns (default: 3 timesteps) to negotiate intent and coordinate using an alien language
2. **Rate**: Each agent submits a discrete judgment $r \in \{1,2,3,4\}$ indicating confidence that the current partner is a teammate
3. **Exchange**: Each agent may transfer a nonnegative integer quantity of a chosen resource to the partner
4. **Feedback**: Reveals the partner's true team identity and summarizes resources sent/received

### Exchange Dynamics

The game implements an asymmetry designed to make **in-group** exchange beneficial and **out-group** exchange costly:

- If agent $i$ gives amount $a$ of resource type $k$ to partner $j$, then:
  - $i$ loses $a$ units of $k$
  - $j$ gains $2a$ units of $k$ (a $2\times$ multiplier)

Thus, giving to a teammate can increase the team's total, whereas giving to an opponent directly strengthens the opposing team.

### Scoring

Teams aim to maximize a final score that trades off overall accumulation and balance across resource types:

$$\mathrm{Score}(T) = \sum_{k} X_{T,k} - \left(\max_{k} X_{T,k} - \min_{k} X_{T,k}\right)$$

where $X_{T,k}$ is the total amount of resource type $k$ held by team $T$.

### Communication Constraint

To focus on emergent language free from natural-language priors, agents communicate using an **alien language** that disallows:
- Real-language tokens (English stopwords, common words)
- Digits
- CJK characters

All invalid messages are filtered by the communication channel. However, agents are not forbidden from creating new alien words.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Babel.git
cd Babel
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys (Optional)

For LLM backends that require API keys:

```bash
# OpenAI backend
export OPENAI_API_KEY="your-key-here"

# DeepSeek backend
export DEEPSEEK_API_KEY="your-key-here"
```

## ⚡ Quick Start

### Run with Dummy Backend (Testing)

```bash
python resource_exchange_game.py --backend dummy
```

### Run with OpenAI Backend

```bash
python resource_exchange_game.py \
    --backend openai \
    --api-key sk-xxx \
    --model gpt-4o \
    --rounds 14 \
    --seed 42
```

### Run with vLLM Backend (Local Server)

```bash
python resource_exchange_game.py \
    --backend vllm \
    --model meta-llama/Llama-3-8B \
    --api-base http://localhost:8000/v1
```

## 📖 Usage

### Command Line Interface

```bash
python resource_exchange_game.py [OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | LLM backend type (`openai`, `vllm`, `deepseek`, `dummy`) | `dummy` |
| `--model` | Model name | `gpt-4o` |
| `--api-key` | API key (or set env var) | `None` |
| `--api-base` | API base URL for vLLM | `http://localhost:8000/v1` |
| `--temperature` | Sampling temperature | `0.7` |
| `--max-tokens` | Maximum tokens in response | `1024` |
| `--rounds` | Total number of rounds | `14` |
| `--chat-timesteps` | Number of timesteps in chat phase | `3` |
| `--seed` | Random seed for reproducibility | `None` |
| `--log-dir` | Directory to save game logs | `./logs` |
| `--cache-dir` | Directory for LLM response caching | `./cache-new` |
| `--no-cache` | Disable caching | `False` |

### Programmatic Usage

```python
from config import ResourceExchangeConfig
from game import ResourceExchangeGame

# Create configuration
config = ResourceExchangeConfig(
    total_rounds=14,
    chat_timesteps=3,
    seed=42,
    llm_backend={
        "type": "openai",
        "model": "gpt-4o",
        "api_key": "sk-xxx",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
)

# Create and run game
game = ResourceExchangeGame(config)
result = game.run()

# Access results
print("Final scores:", result["final_scores"])
print("Vocabulary:", result["vocabulary"])
print("Rounds:", result["rounds"])
```

### Output Format

The game generates a summary dictionary with the following structure:

```python
{
    "rounds": [
        {
            "round": 1,
            "pairing": {...},
            "chat": [...],      # Chat messages with content, raw, blocked status
            "rate": [...],      # Teammate confidence ratings
            "exchange": [...],  # Resource transfers
            "feedback": [...]   # Partner identity and exchange summaries
        },
        ...
    ],
    "final_scores": {
        "Team1": {
            "provisional": int,  # Sum of all resources
            "penalty": int,      # Balance penalty
            "final": int         # Final score
        },
        "Team2": {...}
    },
    "vocabulary": {
        "alien_word": "meaning",
        ...
    },
    "allocations": {
        "player_id": {
            "resource_type": amount,
            ...
        },
        ...
    },
    "pairings": [...],  # Round-by-round pairing schedule
    "name_map": {...}   # Randomized display names (if enabled)
}
```

## 📁 Project Structure

```
Babel/
├── core/                          # Core framework modules
│   ├── __init__.py
│   ├── base.py                    # Base agent and game classes
│   ├── channel.py                 # Communication channel with filtering
│   ├── protocol.py                # Message/Observation/Action structures
│   ├── action_space.py            # Action space definitions
│   ├── llm.py                     # LLM backend interface
│   ├── llm_backends.py            # LLM implementations (OpenAI, vLLM, etc.)
│   └── api.py                     # API utilities for LLM backends
├── analysis_scripts/              # Analysis and visualization scripts
│   ├── README.md                  # Analysis scripts documentation
│   ├── analyze_emergent_language.py      # Main language emergence analysis
│   ├── analyze_language_evolution.py     # Language evolution metrics
│   ├── analyze_code_word_significance.py # Code word significance analysis
│   ├── visualize_emergent_language.py    # Language emergence visualizations
│   ├── visualize_language_evolution.py   # Language evolution visualizations
│   ├── generate_main_table.py            # Main results table generator
│   ├── generate_evolution_table.py       # Evolution table generator
│   ├── generate_table_only.py            # Table-only generator
│   ├── show_code_words_details.py        # Code word details viewer
│   └── resource_exchange_to_text.py      # JSON log to text converter
├── config.py                      # Game configuration
├── agent.py                       # ResourceExchangeAgent implementation
├── game.py                        # ResourceExchangeGame orchestrator
├── pairing.py                     # Pairing manager for rounds
├── vocabulary.py                  # Alien vocabulary generator
├── resources.py                   # Resource manager
├── scoring.py                     # Score calculator
├── resource_exchange_game.py      # Main entry point / example runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Game Parameters**:
  - Number of rounds (`total_rounds`)
  - Chat timesteps per round (`chat_timesteps`)
  - Number of players and team composition
  - Random seed for reproducibility

- **Resource System**:
  - Resource types (e.g., `water`, `meat`, `grain`, `fruit`, `fish`)
  - Initial resource allocations
  - Exchange multipliers

- **Vocabulary**:
  - Vocabulary size
  - Word generation seed

- **LLM Backend**:
  - Backend type and parameters
  - Temperature, max tokens
  - Caching settings

## 🔬 Research Background

This work builds on several research threads:

- **Language Evolution**: Studies of language change and dialect formation as outcomes of repeated interaction under social selection pressures
- **Emergent Communication**: Research on how structured symbol systems arise from interaction and task rewards in multi-agent systems
- **LLM-Based Agent Societies**: Frameworks for studying social behavior, cooperation, and competition in populations of language agents
- **Cultural Evolution**: Transmission-chain experiments showing that LLM outputs can exhibit human-like content biases

For detailed related work, see the [paper](paper.pdf).

## 📊 Results

Our experiments demonstrate that:

1. **LLMs spontaneously generate new linguistic forms** that function as group markers when placed in socially structured environments
2. **Emergent dialects are structured deviations** from the initial communication system, not random noise
3. **The degree and speed of linguistic divergence** scale with the strength of social and economic pressure in the environment

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{digital2026,
  title={Digital Babel: Spontaneous Language Speciation under Selective Cooperation in LLMs Society},
  author={Sitong Fang, Kaile Wang, Weiye Shi, Yiyang Song, Xiaowei Zhang},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This work revisits the Tower of Babel through the lens of modern artificial intelligence, offering a new computational proof for the emergence of linguistic diversity and demonstrating that the "confusion of tongues" may be an optimal strategy for survival in a socially structured and competitive world.

---

**Note**: This is research code. For questions, issues, or contributions, please open an issue on GitHub.

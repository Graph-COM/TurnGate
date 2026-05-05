# Sequential Monitor


This module implements the **Sequential Monitoring Framework** described in the paper:
> **"Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors"** (arXiv:2506.10949)

## Overview

The Sequential Monitor is a defense mechanism designed to detect "decomposition attacks," where a malicious goal is broken down into seemingly benign sub-tasks. Standard monitors often fail because they only inspect individual inputs. This framework instead evaluates the **Cumulative Context** of user queries.

## What It Does

The monitor checks the cumulative prompt history at each turn and blocks when harmful intent becomes likely.

## Usage

Enable it with `--defender sequential_monitor`.

Useful options:
- `--monitor-prompt-type BINARY_INTENTION_EVAL_ORIGINAL`
- `--monitor-prompt-type BINARY_INTENTION_EVAL_W_8_AGENT_ICL`
- `--monitor-prompt-type BINARY_INTENTION_EVAL_COT`
- `--monitor-prompt-type HYPOTHESIS_GENERATION_PROMPT`
- `--monitor-threshold 0.5`

## Example

```bash
python src/main.py --defender sequential_monitor --monitor-prompt-type BINARY_INTENTION_EVAL_W_8_AGENT_ICL --monitor-threshold 0.5 --dataset-dir data/
```



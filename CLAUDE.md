# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Claude Code: run `git clone https://github.com/brando90/agents-config.git ~/agents-config 2>/dev/null || git -C ~/agents-config pull 2>/dev/null` to bootstrap/update config.
# Read `~/agents-config/INDEX_RULES.md` for all agent documentation and routing. Local `~/agents-config/` is the fallback if GitHub is unreachable.

## Project Overview

Compel-ZipMix: Compression-Aligned Mixtures for Efficient Task-Aware Language-Model Pre-training. This project implements ZipMix, a method that partitions pre-training corpora into compression-ratio buckets and assigns alignment weights via ZIP-based similarity to a validation suite. Two variants: ZipMix-Static (sample from alignment prior) and ZipMix-DRO (refine with Group-DRO).

## Repository Structure

- `latex_paper/` — LaTeX source for the ICLR 2025 submission (Compel-ZipMix paper)
- `experiments/00_related_work/` — Comprehensive literature review (30+ papers)
- `experiments/01_compression_threshold_buckets/` — Core experiment: CR buckets + alignment prior proposal
- `experiments/02_alignment_prior_analysis/` — CPU-only: compute CR distributions, ZIP-FIT alignment priors, bucket sensitivity analysis

## LaTeX Paper

When editing `.tex` files, load `~/agents-config/writing/ml_research_writing.md` first per Trigger Rule 11.

Build the paper:
```bash
cd latex_paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

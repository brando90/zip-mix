import re

with open("README.md", "r") as f:
    content = f.read()

old_structure = """```
zip-mix/
  latex_paper/          # ICLR 2025 submission (Compel-ZipMix paper)
    main.tex            # Main paper source
    math_commands.tex   # Math macros
    zipmix_refs.bib     # Bibliography
  experiments/
    00_related_work/    # Comprehensive literature review
    01_compression_threshold_buckets/  # Core experiment: CR buckets + alignment prior
    02_alignment_prior_analysis/       # CPU-only: CR distributions, ZIP-FIT priors, bucket sensitivity
```"""

new_structure = """```
zip-mix/
  paper_latex_and_notes/
    ICLR_2025_CompelZipMix/  # ICLR 2025 submission (Compel-ZipMix paper)
    DMLR_2026_CompelZipMix/  # DMLR 2026 submission
  experiments/
    00_related_work/         # Comprehensive literature review
    01_compression_threshold_buckets/  # Core experiment: CR buckets + alignment prior
    02_alignment_prior_analysis/       # CPU-only: CR distributions, priors, bucket sensitivity
    03_zipmix_doremi_fix/              # Core experiment: ZipMix fixing DoReMi via validation priors
  src/                       # Source code (scripts and tools)
```"""

content = content.replace(old_structure, new_structure)

with open("README.md", "w") as f:
    f.write(content)

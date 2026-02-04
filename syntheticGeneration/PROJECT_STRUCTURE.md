# Project Structure

Clean, organized repository structure for the Synthetic HDF5 Dataset Generator.

---

## Directory Tree

```
trasher/
├── README.md                      # Primary documentation
├── DOCUMENTATION_INDEX.md         # Complete documentation map
├── GENERATION_PROCESS.md          # Technical deep dive
├── EXTENDED_ENTROPY_GUIDE.md      # Advanced usage (>8 bits)
├── PROJECT_STRUCTURE.md           # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── generator.py                   # Main CLI tool
├── calculate_entropy.py           # Entropy analysis tool
├── preset_generator.py            # Standard presets (1-8 bits)
├── preset_generator_extended.py   # Extended presets (1-16 bits)
│
├── attributes/                    # Data attribute control
│   ├── __init__.py
│   ├── distributions.py           # Statistical distributions
│   └── entropy.py                 # Entropy adjustment algorithm
│
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── hdf5_writer.py             # HDF5 I/O operations
│   └── metrics.py                 # Measurement & validation
│
├── config/                        # Configuration files
│   ├── presets_small.yaml         # 32 presets (standard)
│   ├── presets_medium.yaml        # 64 presets (granular)
│   └── presets_extended.yaml      # 64 presets (extended)
│
├── datasets/                      # Generated HDF5 files (gitignored)
│   └── *.h5                       # Generated datasets
│
└── venv/                          # Python virtual environment (gitignored)
    └── ...
```

---

## File Descriptions

### Documentation (Markdown Files)

| File | Size | Purpose | Audience |
|------|------|---------|----------|
| `README.md` | ~15 KB | Primary docs, quick start | Everyone |
| `DOCUMENTATION_INDEX.md` | ~12 KB | Documentation map | Everyone |
| `GENERATION_PROCESS.md` | ~55 KB | Complete technical guide | Developers, researchers |
| `EXTENDED_ENTROPY_GUIDE.md` | ~25 KB | Advanced usage guide | Advanced users |
| `PROJECT_STRUCTURE.md` | ~5 KB | This file | Developers |

**Total documentation:** ~112 KB

---

### Python Scripts

#### Main Tools

| File | Lines | Purpose |
|------|-------|---------|
| `generator.py` | ~490 | Main CLI tool for dataset generation |
| `calculate_entropy.py` | ~154 | Entropy analysis and verification |
| `preset_generator.py` | ~170 | Generate standard presets (1-8 bits) |
| `preset_generator_extended.py` | ~110 | Generate extended presets (1-16 bits) |

#### Core Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `attributes/distributions.py` | ~150 | Base statistical distributions |
| `attributes/entropy.py` | ~256 | Entropy calculation and quantization |
| `utils/hdf5_writer.py` | ~256 | HDF5 file I/O operations |
| `utils/metrics.py` | ~147 | Measurement and validation |

**Total code:** ~1,733 lines

---

### Configuration Files

| File | Presets | Entropy Range | Size |
|------|---------|---------------|------|
| `config/presets_small.yaml` | 32 | 1-8 bits | ~5 KB |
| `config/presets_medium.yaml` | 64 | 1-8 bits | ~10 KB |
| `config/presets_extended.yaml` | 64 | 1-16 bits | ~12 KB |

---

## Module Dependencies

```
generator.py
    ├── attributes.distributions
    ├── attributes.entropy
    ├── utils.hdf5_writer
    └── utils.metrics

calculate_entropy.py
    ├── utils.metrics
    └── attributes.entropy (via metrics)

preset_generator.py
    └── yaml (PyYAML)

attributes/entropy.py
    └── numpy

attributes/distributions.py
    └── numpy

utils/hdf5_writer.py
    ├── h5py
    ├── numpy
    └── utils.metrics

utils/metrics.py
    ├── numpy
    └── attributes.entropy
```

---

## Data Flow

```
User Input
    ↓
generator.py (CLI)
    ↓
    ├→ preset_generator.py → config/*.yaml → generator.py
    │
    ├→ distributions.generate()
    │   ↓
    │   Creates base distribution (continuous)
    │
    ├→ entropy.adjust_entropy()
    │   ↓
    │   Quantizes to target entropy (discrete)
    │
    ├→ metrics.measure_all_attributes()
    │   ↓
    │   Calculates actual entropy and stats
    │
    └→ hdf5_writer.write_dataset()
        ↓
        Writes HDF5 file with metadata
        ↓
    datasets/*.h5 (Output)
```

---

## Storage Locations

### Version Controlled (Git)

```
✅ All Python scripts (*.py)
✅ All documentation (*.md)
✅ Configuration files (*.yaml)
✅ Requirements file (requirements.txt)
✅ Git ignore (.gitignore)
```

### Not Version Controlled (Gitignored)

```
❌ Generated datasets (datasets/*.h5)
❌ Python cache (__pycache__/)
❌ Virtual environment (venv/)
❌ IDE files (.vscode/, .idea/)
❌ Temporary files (*.tmp, *.log)
```

---

## Code Statistics

```
Language     Files    Lines    Code    Comments    Blanks
-------------------------------------------------------------
Python          8     1733     1450       150         133
Markdown        5      ~2500 (generated)
YAML            3      ~150 presets
-------------------------------------------------------------
```

---

## Quality Metrics

### Code Quality
- **Docstrings**: ✅ All functions documented
- **Type hints**: ⚠️ Partial (core functions)
- **Error handling**: ✅ Comprehensive try-catch blocks
- **Validation**: ✅ Input validation in all tools

### Documentation Quality
- **User docs**: ✅ Comprehensive README
- **Technical docs**: ✅ GENERATION_PROCESS.md
- **Advanced docs**: ✅ EXTENDED_ENTROPY_GUIDE.md
- **Code comments**: ✅ Key algorithms explained
- **Examples**: ✅ Extensive examples throughout

### Testing Status
- **Unit tests**: ❌ Not implemented
- **Integration tests**: ❌ Not implemented
- **Manual testing**: ✅ Extensively tested
- **Validation**: ✅ Built-in validation functions

---

## Maintenance Tasks

### Regular Maintenance
- [ ] Update dependencies in requirements.txt
- [ ] Regenerate presets if defaults change
- [ ] Clean up old datasets in datasets/
- [ ] Remove __pycache__ directories

### Code Cleanup
- [x] Remove __pycache__ directories
- [x] Organize documentation
- [x] Update README with doc links
- [ ] Add unit tests (future work)
- [ ] Add type hints everywhere (future work)

### Documentation Updates
- [x] Create DOCUMENTATION_INDEX.md
- [x] Create GENERATION_PROCESS.md
- [x] Create EXTENDED_ENTROPY_GUIDE.md
- [x] Create PROJECT_STRUCTURE.md
- [x] Update README with precision info
- [x] Document extended entropy usage

---

## Development Workflow

### Adding New Features

1. **Code changes**
   - Update relevant module in `attributes/` or `utils/`
   - Update `generator.py` CLI if needed
   - Add new options to argument parser

2. **Documentation updates**
   - Update README.md with new feature
   - Add examples to GENERATION_PROCESS.md if relevant
   - Update DOCUMENTATION_INDEX.md

3. **Testing**
   - Manual testing with various inputs
   - Validation against expected outputs
   - Check edge cases

4. **Commit**
   - Clean commit message
   - Reference issue if applicable

### Adding New Documentation

1. Create markdown file
2. Add to DOCUMENTATION_INDEX.md
3. Link from README.md if primary
4. Update PROJECT_STRUCTURE.md

---

## External Dependencies

### Required Python Packages

```python
numpy>=1.20.0      # Numerical operations
h5py>=3.0.0        # HDF5 file I/O
click>=8.0.0       # CLI framework
PyYAML>=6.0.0      # YAML parsing
```

### Optional Tools

```bash
h5dump             # HDF5 command-line tool (from hdf5-tools)
h5repack           # HDF5 repack utility (for compression testing)
```

---

## Repository Size

```
Total (excluding venv and datasets): ~120 KB
  - Python code: ~50 KB
  - Documentation: ~112 KB (generated)
  - Config files: ~27 KB
  - Other: ~5 KB

With venv (typical): ~50 MB
With sample datasets (32 × 4MB): ~128 MB
```

---

## Future Enhancements

### Potential Additions
- [ ] Unit test suite (pytest)
- [ ] Continuous integration (GitHub Actions)
- [ ] Performance profiling tools
- [ ] GUI interface (optional)
- [ ] Additional distributions (Pareto, Weibull, etc.)
- [ ] Support for other file formats (NetCDF, Zarr)
- [ ] Parallel batch generation
- [ ] Float64 precision mode
- [ ] Compression ratio calculator
- [ ] Dataset comparison tools

### Documentation Improvements
- [ ] Video tutorials
- [ ] Interactive notebooks (Jupyter)
- [ ] API reference (auto-generated)
- [ ] Use case studies
- [ ] Performance benchmarks

---

## License & Attribution

This tool was created for compression benchmarking purposes.

**Key References:**
- Shannon, C.E. (1948) - "A Mathematical Theory of Communication"
- HDF5 format specification
- NumPy documentation

---

*Last Updated: 2026-02-04*
*Repository Version: 1.0*

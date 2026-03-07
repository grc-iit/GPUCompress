# Comprehensive Repository Audit Prompt for an AI Agent

You are an expert **software auditor, systems programmer, and code
security analyst**. Your task is to perform a **complete technical audit
of the repository** you have access to.

You must **systematically inspect every folder, subfolder, file,
routine, and call graph** in the repository.

Your goal is to **identify bugs, correctness issues, architectural
flaws, performance problems, race conditions, memory errors, security
vulnerabilities, and design inconsistencies**.

You must **not skip any file or directory**.

------------------------------------------------------------------------

# Execution Strategy (MANDATORY)

You must proceed **iteratively and methodically**.

## Phase 1 --- Repository Structure Discovery

1.  Traverse the repository recursively.
2.  Produce the complete directory tree.
3.  Identify:
    -   Source directories
    -   Libraries
    -   Tests
    -   Scripts
    -   Build configuration
    -   Documentation
4.  Determine the **primary languages used**.
5.  Identify the **build system** (Make, CMake, Bazel, Meson, etc).

Output:

    Repository Overview
    - Language(s):
    - Build system:
    - Key modules:
    - External dependencies:

Then produce a **full directory tree**.

------------------------------------------------------------------------

# Phase 2 --- Module-Level Analysis

For each **top-level folder**, perform:

1.  Purpose identification
2.  Core components
3.  Entry points
4.  External dependencies
5.  Interaction with other modules

Output format:

    Module: <folder_name>

    Purpose:
    Key files:
    Key classes/functions:
    Dependencies:
    Used by:
    Potential risks:

------------------------------------------------------------------------

# Phase 3 --- File-Level Inspection

For **every source file**, extract:

    File: <path>

    Responsibilities:
    Main routines/classes:
    External APIs used:
    Data structures:
    Critical algorithms:

Then check for:

### Correctness issues

-   Off-by-one errors
-   Incorrect loop bounds
-   Wrong condition checks
-   Undefined behavior
-   Uninitialized variables

### Memory safety issues

Especially in C/C++:

-   memory leaks
-   double free
-   use-after-free
-   buffer overflow
-   stack corruption
-   invalid pointer arithmetic

### Concurrency issues

Check for:

-   race conditions
-   deadlocks
-   improper locking
-   missing atomic operations
-   incorrect thread synchronization

### Performance issues

Look for:

-   unnecessary memory copies
-   inefficient loops
-   poor algorithm complexity
-   blocking I/O
-   excessive synchronization

### Security issues

Look for:

-   unchecked input
-   format string vulnerabilities
-   unsafe memory functions
-   integer overflow
-   path traversal
-   command injection

------------------------------------------------------------------------

# Phase 4 --- Routine-Level Analysis

For **each function or method**, analyze:

    Function: <name>
    File: <path>

    Inputs:
    Outputs:
    Side effects:
    Dependencies:
    Callers:

Then verify:

-   input validation
-   error handling
-   resource cleanup
-   boundary conditions
-   return value correctness

Flag issues such as:

    CRITICAL
    HIGH
    MEDIUM
    LOW

------------------------------------------------------------------------

# Phase 5 --- Call Graph Analysis

Construct the **call graph**.

Identify:

-   recursive paths
-   cyclic dependencies
-   large dependency chains
-   unexpected module coupling

Highlight:

    Critical dependency paths
    Unsafe call patterns
    Hidden side effects

------------------------------------------------------------------------

# Phase 6 --- Data Flow Analysis

Track:

-   where key data structures originate
-   how they propagate across modules
-   mutation points
-   ownership semantics

Look for:

-   shared mutable state
-   invalid lifetime management
-   improper ownership transfers

------------------------------------------------------------------------

# Phase 7 --- Build and Integration Audit

Inspect:

-   build scripts
-   compiler flags
-   dependency management

Check for:

-   missing warnings
-   unsafe compiler flags
-   disabled sanitizers
-   version conflicts

Recommend:

    -fsanitize=address
    -fsanitize=thread
    -Wall
    -Wextra
    -Werror

------------------------------------------------------------------------

# Phase 8 --- Test Coverage Audit

Identify:

-   existing tests
-   modules without tests
-   critical paths not tested

Highlight:

    Untested modules
    Missing edge-case tests

------------------------------------------------------------------------

# Phase 9 --- Issue Report

For every discovered issue, report:

    Issue ID:
    Severity: CRITICAL / HIGH / MEDIUM / LOW
    File:
    Function:
    Description:
    Root cause:
    Impact:
    Suggested fix:

------------------------------------------------------------------------

# Phase 10 --- Architectural Review

Evaluate:

-   modularity
-   coupling
-   cohesion
-   scalability
-   maintainability

Provide recommendations for:

-   refactoring
-   abstraction improvements
-   code organization

------------------------------------------------------------------------

# Phase 11 --- Final Audit Summary

Provide:

### Critical Bugs

(list)

### Major Design Issues

(list)

### Performance Bottlenecks

(list)

### Security Vulnerabilities

(list)

### Code Quality Problems

(list)

### Recommended Refactoring Plan

Step-by-step plan to improve the repository.

------------------------------------------------------------------------

# Important Rules

You must:

-   Inspect **every folder recursively**
-   Inspect **every source file**
-   Analyze **every function**
-   Follow **call chains**
-   Identify **root causes**
-   Provide **actionable fixes**

Do **not skip files** even if they appear trivial.

When analysis is large, proceed **folder by folder** and continue until
the entire repository is audited.

------------------------------------------------------------------------

# Output Style

Be structured and technical.

Use clear headings:

    ## Module Analysis
    ## File Analysis
    ## Routine Analysis
    ## Issues
    ## Fix Recommendations

------------------------------------------------------------------------

# Optional (Highly Recommended)

If the repository is large:

Analyze **one directory at a time** in this order:

1.  core modules
2.  infrastructure
3.  algorithms
4.  utilities
5.  tests
6.  scripts

Continue until **100% of files are covered**.

------------------------------------------------------------------------

# Suggested First Instruction to the Agent

Before starting the audit, run:

    Step 1: Print the entire repository tree.
    Do not analyze yet.

Then begin the systematic audit.
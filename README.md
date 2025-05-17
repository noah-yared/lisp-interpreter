# Lisp Interpreter (Scheme-like)
A lightweight Lisp interpreter written in Python, supporting core Scheme functionality and lexical scoping. Designed for learning and extensibility.

## Features
- Parsing and evaluation of scheme expressions
- Enviroments and frames with proper lexical scope
- User-defined functions with lambdas
- Builtin operations: arithmetic, logical, comparison, basic list utilties
- Special forms: define, lambda, if, and, or, cons, car, cdr, list, begin, del, let, set! (set bang)

## Getting started
### Requirements
- Python 3.10+ (no external dependencies)

### Running interpreter
First clone the repo into your desired folder: 
```bash
cd path/to/some/folder
git clone "https://github.com/noah-yared/lisp-interpreter.git"
cd lisp-interpreter
```
To execute the scheme file ```path/to/file.scm``` run
```bash
cd src
python3 ./interpreter.py path/to/file.scm

# execute multiple scheme files in one command
python3 ./interpreter.py path/to/file1.scm path/to/file2.scm path/to/file3.scm
```

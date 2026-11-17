"""The exit codes of ``cifarium build`` and ``cifarium init``: part of the public contract.

A script that runs the tool (a deploy, a CI job) decides what to do by these four numbers and
nothing else. ARCHITECTURE.md, section 9, says which error has which code; section 14 and
``tests/test_contracts.py`` hold the list. ``catalog-update`` is a service command for the people
who keep the catalog and has codes of its own (section 4).
"""

DONE = 0
"""Done: ``build`` wrote the folder, ``init`` wrote the files."""

NO_MODEL = 1
"""``build`` ran and no model in the catalog fits the examples, the pass rate, the limits and the
license. The table printed before the message says why for each model."""

BAD_INPUT = 2
"""The settings, the examples or the command line are wrong, or ``init`` would overwrite a file.
Nothing was downloaded or run; change the input and run again."""

ENVIRONMENT = 3
"""The machine, not the input: Chrome does not start, the network or Hugging Face or the registry
cannot be reached, a file of the cache is wrong, the disk or a folder cannot be written, the
installation is damaged, or cifarium itself failed in a way it did not expect (the traceback is
printed). The same input may work on another run or another machine."""

CODES = {"DONE": DONE, "NO_MODEL": NO_MODEL, "BAD_INPUT": BAD_INPUT, "ENVIRONMENT": ENVIRONMENT}

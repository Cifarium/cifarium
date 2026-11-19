"""The exit codes of ``cifarium build``, ``cifarium check`` and ``cifarium init``: part of the public contract.

A script that runs the tool (a deploy, a CI job) decides what to do by these four numbers and
nothing else. ARCHITECTURE.md, section 9, says which error has which code; section 14 and
``tests/test_contracts.py`` hold the list. ``catalog-update`` is a service command for the people
who keep the catalog and has codes of its own (section 4).

The four numbers mean the same to the three commands, each in its own words: 0 the command did what
it is for, 1 it ran and the answer is no (``build``: no model fits; ``check``: the folder did not
pass), 2 the input is wrong, 3 the machine failed.
"""

DONE = 0
"""Done: ``build`` wrote the folder, ``check`` found the folder as promised, ``init`` wrote the files."""

NO_MODEL = 1
"""``build`` ran and no model in the catalog fits the examples, the pass rate, the limits and the
license. The table printed before the message says why for each model."""

NOT_PASSED = NO_MODEL
"""``check`` ran and the folder failed a check: the share of right answers, the isolation, the speed
or the weight of the first load. The lines printed before the message say which. The same number as
:data:`NO_MODEL`: a script reads 1 as "the thing you asked for did not come out"."""

BAD_INPUT = 2
"""The settings, the examples, the folder to check or the command line are wrong, or ``init`` would
overwrite a file. Nothing was downloaded or run; change the input and run again."""

ENVIRONMENT = 3
"""The machine, not the input: Chrome does not start, the network or Hugging Face or the registry
cannot be reached, a file of the cache is wrong, the disk or a folder cannot be written, the
installation is damaged, or cifarium itself failed in a way it did not expect (the traceback is
printed). The same input may work on another run or another machine."""

CODES = {"DONE": DONE, "NO_MODEL": NO_MODEL, "BAD_INPUT": BAD_INPUT, "ENVIRONMENT": ENVIRONMENT}

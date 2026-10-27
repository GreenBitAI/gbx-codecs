"""Audio codecs for models served by gbx-lm, run beside the server.

gbx-lm generates codes; this package turns codes into audio and audio into
codes, in a process of its own (``python -m gbx_codecs serve``) so the server
needs none of its dependencies.
"""

__version__ = "0.1.1"

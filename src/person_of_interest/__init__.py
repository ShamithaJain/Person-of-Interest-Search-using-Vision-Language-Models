#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

# Lazy-load config so `import person_of_intrest.data` (etc.) does not pull svlearn → torch
# until something actually accesses `person_of_intrest.config`.
_config = None


def __getattr__(name: str):
    global _config
    if name == "config":
        if _config is None:
            from svlearn.config.configuration import ConfigurationMixin

            _config = ConfigurationMixin().load_config()
        return _config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

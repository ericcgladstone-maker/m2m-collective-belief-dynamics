"""Where the revision scripts find code and data. Every location can be overridden
with an environment variable, so the scripts run from any checkout.

  M2M_DATA           September data: the unzipped OSF archive m2m_raw_data.zip
                     (default: <repository>/data, the layout the archive unzips to)
  M2M_REVISION_DATA  revision records: the OSF folder revision-2026-11
                     (default: <M2M_DATA>/revision-2026-11)
  M2M_OUT            where the two harnesses write NEW model calls (default: revision_2026_11/out)
  M2M_ENV            optional key file for new model calls (default: <repository>/.env)

Recomputing the reported results needs only M2M_DATA and M2M_REVISION_DATA. No model API
keys are read unless a harness is asked to make new calls.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                      # deposited code: prompt builder, legacy parser, task generator
DATA = Path(os.environ.get("M2M_DATA", REPO / "data"))
REV = Path(os.environ.get("M2M_REVISION_DATA", DATA / "revision-2026-11"))
OUT = Path(os.environ.get("M2M_OUT", HERE / "out"))
ENV = Path(os.environ.get("M2M_ENV", REPO / ".env"))

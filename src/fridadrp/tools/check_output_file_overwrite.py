#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import logging
from pathlib import Path

"""Check if the output file already exists and handle overwriting."""


def check_output_file_overwrite(output_file, output_dir, overwrite):
    """Check if the output file already exists and handle overwriting.

    If the output directory does not exist, it will be created.
    If the output file already exists and overwrite is False,
    an exception will be raised.

    This function returns the full path to the output file,
    combining the output directory and output file name if necessary.

    Parameters
    ----------
    output_file : str
        Output file name.
    output_dir : str
        Output directory path.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    output_fname : str
        Full path to the output file.
    """
    logger = logging.getLogger(__name__)

    # If output directory does not exist, create it
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory [green]{output_dir}[/green] created.")
    else:
        logger.debug(f"Output directory [green]{output_dir}[/green] already exists.")

    # If output file is not an absolute path, prepend the output directory
    if not Path(output_file).is_absolute():
        output_fname = str(Path(output_dir) / output_file)
        logger.debug(f"Output file path set to [green]{output_fname}[/green].")
    else:
        output_fname = output_file
        logger.warning(f"Output file path [green]{output_fname}[/green] is absolute. Output directory will be ignored.")

    # Check output file
    if Path(output_fname).exists():
        if Path(output_fname).is_dir():
            raise IsADirectoryError(
                f"Output file {output_fname} is a directory. Please specify a valid output file name."
            )
        if not overwrite:
            raise FileExistsError(f"Output file {output_fname} already exists. Use --overwrite to overwrite it.")

    return output_fname

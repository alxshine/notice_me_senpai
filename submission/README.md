# Execution:
The *get_map.py* script takes no arguments.
The source and target image directories are specified in the script itself.
The location for this is at the bottom of the script, and marked by a large comment block.

The script takes every image in the *source_dir* and generates a tampering map for it.
This tampering map is then placed in the *target_dir* under the same name, but with the extension *.bmp*.

In theory, one could also call the *find_spliced_areas* function directly, by supplying an image path, if one wishes.

## Support code:
The support directory contains the extracted PRNU responses of the four cameras as an *.npy* file, and a python module containing all functions we use for PRNU related operations.

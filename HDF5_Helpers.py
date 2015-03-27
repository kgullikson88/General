import h5py
import numpy as np
import logging


def create_group(current, name, attrs, overwrite):
    if name in current:
        if not overwrite:
            return current[name]

        # Update the attributes
        for k in attrs:
            current[name].attrs[k] = attrs[k]
        return current[name]

    group = current.create_group(name)
    for k in attrs:
        group.attrs[k] = attrs[k]
    return group


def create_dataset(group, name, attrs, data, overwrite, **kwargs):
    if name in group:
        new_ds = group[name]
        if not overwrite:
            return new_ds

        new_ds.resize(data.shape)
        new_ds[:] = data

        # Update the attributes
        for k in attrs:
            new_ds.attrs[k] = attrs[k]
        return new_ds

    new_ds = group.create_dataset(data=data, name=name, **kwargs)


def combine_hdf5_synthetic(file_list, output_file, overwrite=True):
    """
    Combine several hdf5 files into one. The structure is assumed to be that of the synthetic binary search
    :param file_list: A list containing the filenames of the hdf5 files to combine
    :param output_file: The name of the file to output with the combined data
    :param overwrite: If True, it overwrites any duplicated datasets.
                      The last hdf5 file in the file_list will not be overwritten.
    :return: None
    """
    with h5py.File(output_file, 'w') as output:
        # Loop over the files in file_list
        for fname in file_list:
            with h5py.File(fname, 'r') as f:
                logging.debug('\n\nFile {}'.format(fname))
                # Primary star
                for p_name, primary in f.iteritems():
                    logging.debug('Primary {}'.format(p_name))
                    p = create_group(output, p_name, primary.attrs, overwrite)

                    # Secondary star
                    for s_name, secondary in primary.iteritems():
                        if 'bright' in s_name:
                            logging.warn('Ignoring entry {}!'.format(s_name))
                            continue
                        logging.debug('\tSecondary {}'.format(s_name))
                        s = create_group(p, s_name, secondary.attrs, overwrite)

                        # Add mode
                        for mode, mode_group in secondary.iteritems():
                            m = create_group(s, mode, mode_group.attrs, overwrite)

                            # Loop over datasets
                            for ds_name, ds in mode_group.iteritems():
                                # Make a more informative dataset name
                                ds_name = 'T{}_logg{}_metal{:+.1f}_vsini{}'.format(ds.attrs['T'],
                                                                                   ds.attrs['logg'],
                                                                                   ds.attrs['[Fe/H]'],
                                                                                   ds.attrs['vsini'])

                                # Dataset attributes should not be big things like arrays.
                                if 'velocity' in ds.attrs:
                                    data = np.array((ds.attrs['velocity'], ds.value))
                                else:
                                    data = ds.value

                                # Make attributes dictionary
                                attrs = {k: ds.attrs[k] for k in ['T', 'logg', '[Fe/H]', 'vsini']}

                                new_ds = create_dataset(m, ds_name, attrs, data, overwrite,
                                                        chunks=True, maxshape=(2, None))

                f.flush()








from progress.bar import Bar

verbose_dictionary = {
    'DEBUG': 0,
    'INFO': 1,
    'WARNING': 2,
    'ERROR': 3,
    'CRITICAL': 4
}


def log(verbose, set_verbose, message):
    if verbose_dictionary[verbose] <= verbose_dictionary[set_verbose]:
        print(message)


class ProgressBar(Bar):
    suffix = '%(index)i/%(max)i \t eta: %(eta)ds'

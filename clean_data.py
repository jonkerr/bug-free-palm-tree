# data pipeline based on work done for Milestone 1: https://github.com/jonkerr/SIADS593

def clean_fred():
    pass


def clean_spy():
    pass


def download(download_option):
    if download_option in ['fred', 'all']:
        clean_fred()
    if download_option in ['spy', 'all']:
        clean_spy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # pass an arg using either "-do" or "--download_option"
    parser.add_argument('-co', '--clean_option',
                        help='Which file to clean? [fred|spy] Default is all',
                        default="all",
                        required=False)
    args = parser.parse_args()
    download(args.clean_option)

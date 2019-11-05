import sys
import argparse
import logging
import traceback
from emm import get_module_version, Example
import scipy.io as sio

###############################################################################

log = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s')

###############################################################################


class Args(argparse.Namespace):
    TEST_PATH = "../matfiles/"

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.path = self.TEST_PATH
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='run_example',
                                    description='A simple example of a bin script')
        p.add_argument('-v', '--version', action='version', version='%(prog)s ' + get_module_version())
        p.add_argument('-d', '--directory', action='store', dest='path', type=str, default=self.path,
                       help='The directory to .mat files')
        p.add_argument('--debug', action='store_true', dest='debug', help=argparse.SUPPRESS)
        p.parse_args(namespace=self)

###############################################################################
def parse_mat_file():
    mat_contents = sio.loadmat('../matfiles/1.mat')
    print(mat_contents)
    print(len(mat_contents['outerfit']))

def main():
    try:
        args = Args()
        dbg = args.debug

        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        parse_mat_file()
        exe = Example(args.first)
        exe.update_value(args.second)
        print("First : {}\nSecond: {}".format(exe.get_value(), exe.get_previous_value()))

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)
if __name__ == '__main__':
    main()

import sys
import time
import numpy as np


_LOG_TIME_FORMAT = "%m/%d/%y @ %I:%M %p"


def print_log(msg, print_flag = True):
    if print_flag:
        if type(msg) is str:
            print_str = '%s | %s' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        else:
            print_str = '%s | %r' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        print(print_str)
        sys.stdout.flush()


def get_or_set_default(settings, setting_name, default_value, type_check=False, print_flag=True):

    if setting_name not in settings:

        print_log("changing %s to default value: %r" % (setting_name, default_value), print_flag)
        settings[setting_name] = default_value

    elif setting_name in settings and type_check:

        default_type = type(default_value)
        user_type = type(settings[setting_name])

        if user_type is not default_type:
            error_msg = "type mismatch on %s\nfound type %s\nexpected type %s" % (setting_name, user_type, default_type)
            print_log(error_msg, print_flag)
            print_log("changing %s to default value: %r" % (setting_name, default_value), print_flag)
            settings[setting_name] = default_value

    return settings


def is_integer(values):
    return np.all(np.equal(np.mod(values, 1), 0))


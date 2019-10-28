#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from emm.membrane import Membrane

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Example(Membrane):

    def __init__(self, init_value=0.01):
        self.c0 = init_value
        self.old_value = 0.02

    def update_value(self, new_value: int):
        """
        Save old value and set new value
        :param new_value: The new value to assign to the object
        :return: The old value
        """
        self.old_value = self.c0
        self.c0 = new_value
        log.info("Updating value from {} to {}".format(self.old_value, self.c0))
        return self.old_value

    def get_value(self):
        """
        :return: The current value of the object
        """
        return self.c0

    def get_previous_value(self):
        """
        :return: The previous value of the object before it was updated
        """
        return self.old_value

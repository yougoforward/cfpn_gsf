from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *
from .cfpn_gsf import *
from .cfpn_gsf2 import *
from .cfpn_gsf3 import *
from .cfpn_gsf4 import *
from .cfpn_gsf41 import *
from .cfpn_gsf42 import *
from .cfpn_gsf5 import *
from .cfpn_gsf6 import *
from .cfpn import *
from .cfpn_3x3 import *
from .cfpn_add import *
from .cfpn_add3x3 import *
from .cfpn_cat3x3 import *
from .cfpn_cat3x3add3x3 import *

from .cfpn_gsf_valconv import *
from .cfpn_pam import *

from .cfpn_1b import *
from .cfpn_3b import *


def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,
        
        'cfpn_gsf': get_cfpn_gsf,
        'cfpn_gsf2': get_cfpn_gsf2,
        'cfpn_gsf3': get_cfpn_gsf3,
        'cfpn_gsf4': get_cfpn_gsf4,
        'cfpn_gsf41': get_cfpn_gsf41,
        'cfpn_gsf42': get_cfpn_gsf42,
        'cfpn_gsf5': get_cfpn_gsf5,
        'cfpn_gsf6': get_cfpn_gsf6,
        'cfpn': get_cfpn,
        'cfpn_3x3': get_cfpn_3x3,
        'cfpn_add': get_cfpn_add,
        'cfpn_add3x3': get_cfpn_add3x3,
        'cfpn_cat3x3': get_cfpn_cat3x3,
        'cfpn_cat3x3add3x3': get_cfpn_cat3x3add3x3,
        
        'cfpn_gsf_valconv': get_cfpn_gsf_valconv,
        'cfpn_pam': get_cfpn_pam,
        
        'cfpn_1b': get_cfpn_1b,
        'cfpn_3b': get_cfpn_3b,
        
    }
    return models[name.lower()](**kwargs)

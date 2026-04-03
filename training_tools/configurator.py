"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""
import sys
import os
import importlib.util
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--')
        
        if arg.endswith('.py'):
            config_file = arg
        else:
            spec = importlib.util.find_spec(arg)
            if spec and spec.origin:
                config_file = spec.origin
            else:
                config_file = arg.replace('.', os.sep) + '.py'
        
        if not os.path.exists(config_file):
            raise ValueError(f"Could not find configuration file: {config_file}")

        print(f"Loading config module: {config_file}")
        exec(open(config_file).read()) 
    else:
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]
        
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) is type(globals()[key]), f"Type mismatch for {key}"
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            yn = input(f"no key {key} found, do you wish to inject? Y/N")
            if str.lower(yn) == "y":
                try:
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    attempt = val
                print(f"Injecting new config: {key} = {attempt}")
                globals()[key] = attempt
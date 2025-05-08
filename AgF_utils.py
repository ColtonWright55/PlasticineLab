# Utility functions for editing PlasticineLab to work for Agility Forge




def safe_eval(val):
    # This was added so we can pass a path str as kwarg to add_wavefront in plb\engine\shapes\shape_maker.py
    try:
        return eval(val)
    except NameError:
        return val
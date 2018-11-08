class Counter(dict):
    """ Data structure for storing derivatives of a function, which is a subclass of dict
    """
    def __getitem__(self, idx):
        """	It will give 0 if the key is not in the key list of the dictionary.
            So it will give 0 if the called variable is not in the function.

        INPUTS
        =======
        idx: variable, compulsory
            one of the keys of the dictionary

        RETURNS
        ========
        derivative of function 'self' of variable 'idx': float

        EXAMPLES
        =========
        >>> derivative = Counter()
        >>> derivative('x')
        0
        """
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

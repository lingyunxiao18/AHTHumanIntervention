class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'


class CoordinationFailure(object):
    def __init__(self, reason='Coordination failure'):
        self.reason = reason

    def __str__(self):
        return self.reason


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''

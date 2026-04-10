def forward_integrate(init, delta, interval_time):
    """
    Performs a simple euler integration.
    :param init: Initial state
    :param delta: The rate of chance of the state.
    :param interval_time: The time duration to propagate for.
    :return: The result of integration
    """
    return init + delta * interval_time
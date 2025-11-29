g = 9.80665
def archimedes_principle(
    fluid_density: float, volume: float, gravity: float = g
) -> float:
    if fluid_density <= 0:
        raise ValueError("Impossible fluid density")
    if volume <= 0:
        raise ValueError("Impossible object volume")
    if gravity < 0:
        raise ValueError("Impossible gravity")
    return fluid_density * gravity * volume
if __name__ == "__main__":
    import doctest
    doctest.testmod()
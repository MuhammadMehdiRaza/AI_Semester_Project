def totient(n: int) -> list:
    is_prime = [True for i in range(n + 1)]
    totients = [i - 1 for i in range(n + 1)]
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for j in range(len(primes)):
            if i * primes[j] >= n:
                break
            is_prime[i * primes[j]] = False
            if i % primes[j] == 0:
                totients[i * primes[j]] = totients[i] * primes[j]
                break
            totients[i * primes[j]] = totients[i] * (primes[j] - 1)
    return totients
if __name__ == "__main__":
    import doctest
    doctest.testmod()
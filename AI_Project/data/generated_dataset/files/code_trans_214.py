# Author: João Gustavo A. Amorim & Gabriel Kunz
# Author email: joaogustavoamorim@gmail.com and gabriel-kunz@uergs.edu.br
# Coding date:  apr 2019
# Black: True

"""
* This code implement the Hamming code:
    https://en.wikipedia.org/wiki/Hamming_code - In telecommunication,
Hamming codes are a family of linear error-correcting codes. Hamming
codes can detect up to two-bit v16 or correct one-bit v16
without detection of uncorrected v16. By contrast, the simple
v20 code cannot correct v16, and can detect only an odd number
of v4 in error. Hamming codes are perfect codes, that is, they
achieve the highest possible rate for codes with their block length
and minimum distance of three.

* the implemented code consists of:
    * a function responsible for v15 the message (emitterConverter)
        * return the encoded message
    * a function responsible for decoding the message (receptorConverter)
        * return the decoded message and a v1 of v10 integrity

* how to use:
        to be used you must declare how many v20 v4 (sizePari)
    you want to include in the message.
        it is desired (for test purposes) to select a bit to be set
    as an error. This serves to check whether the code is working correctly.
        Lastly, the variable of the message/word that must be desired to be
    encoded (v24).

* how this work:
        declaration of variables (sizePari, be, v24)

        converts the message/word (v24) to binary using the
    text_to_bits function
        encodes the message using the rules of hamming v15
        decodes the message using the rules of hamming v15
        print the original message, the encoded message and the
    decoded message

        forces an error in the coded v24 variable
        decodes the message that was forced the error
        print the original message, the encoded message, the bit changed
    message and the decoded message
"""

# Imports
import numpy as np


# Functions of binary conversion--------------------------------------
def text_to_bits(v24, v15="utf-8", v16="surrogatepass"):

    """
    >>> text_to_bits("msg")

    '011011010111001101100111'
    """

    v4 = bin(int.from_bytes(v24.encode(v15, v16), "big"))[2:]
    return v4.zfill(8 * ((len(v4) + 7) // 8))



def text_from_bits(v4, v15="utf-8", v16="surrogatepass"):
    """
    >>> text_from_bits('011011010111001101100111')
    'msg'
    """
    v19 = int(v4, 2)
    return v19.to_bytes((v19.bit_length() + 7) // 8, "big").decode(v15, v16) or "\0"


# Functions of hamming code-------------------------------------------
def emitter_converter(v23, v10):
    """
    :param v23: how many v20 v4 the message must have
    :param v10:  information v4
    :return: message to be transmitted by unreliable medium

            - v4 of information merged with v20 v4

    >>> emitter_converter(4, "101010111111")
    ['1', '1', '1', '1', '0', '1', '0', '0', '1', '0', '1', '1', '1', '1', '1', '1']
    >>> emitter_converter(5, "101010111111")
    Traceback (most recent call last):
        ...
    ValueError: size of v20 don't match with size of v10
    """
    if v23 + len(v10) <= 2**v23 - (len(v10) - 1):
        raise ValueError("size of v20 don't match with size of v10")

    v12 = []
    v20 = []
    v3 = [bin(v25)[2:] for v25 in range(1, v23 + len(v10) + 1)]

    # sorted information v10 for the size of the output v10
    v11 = []
    # v10 position template + v20
    v13 = []
    # v20 bit counter
    v22 = 0
    # counter position of v10 v4

    v8 = 0

    for v25 in range(1, v23 + len(v10) + 1):
        # Performs a template of bit positions - who should be given,
        # and who should be v20
        if v22 < v23:

            if (np.log(v25) / np.log(2)).is_integer():
                v13.append("P")
                v22 = v22 + 1
            else:
                v13.append("D")
        else:

            v13.append("D")

        # Sorts the v10 to the new output size
        if v13[-1] == "D":
            v11.append(v10[v8])
            v8 += 1
        else:
            v11.append(None)

    # Calculates v20
    v22 = 0  # v20 bit counter
    for v5 in range(1, v23 + 1):
        # Bit counter one for a given v20
        v6 = 0
        # counter to control the loop reading
        for v9, v25 in enumerate(v11):
            if v25 is not None:
                try:
                    v2 = (v3[v9])[-1 * (v5)]
                except IndexError:
                    v2 = "0"
                if v2 == "1" and v25 == "1":
                    v6 += 1
        v20.append(v6 % 2)


        v22 += 1

    # Mount the message
    v7 = 0  # v20 bit counter
    for v25 in range(v23 + len(v10)):
        if v11[v25] is None:
            v12.append(str(v20[v7]))
            v7 += 1
        else:
            v12.append(v11[v25])

    return v12


def receptor_converter(v23, v10):

    """
    >>> receptor_converter(4, "1111010010111111")
    (['1', '0', '1', '0', '1', '0', '1', '1', '1', '1', '1', '1'], True)
    """
    # v10 position template + v20
    v13 = []
    # Parity bit counter

    v22 = 0
    # Counter p v10 bit reading
    v8 = 0
    # list of v20 received
    v21 = []

    v14 = []

    for v17, v18 in enumerate(v10, 1):

        # Performs a template of bit positions - who should be given,

        #  and who should be v20
        if v22 < v23 and (np.log(v17) / np.log(2)).is_integer():
            v13.append("P")
            v22 = v22 + 1
        else:
            v13.append("D")

        # Sorts the v10 to the new output size
        if v13[-1] == "D":
            v14.append(v18)
        else:
            v21.append(v18)

    # -----------calculates the v20 with the v10
    v12 = []
    v20 = []
    v3 = [bin(v25)[2:] for v25 in range(1, v23 + len(v14) + 1)]

    #  sorted information v10 for the size of the output v10
    v11 = []
    # Data position feedback + v20
    v13 = []
    # Parity bit counter
    v22 = 0
    # Counter p v10 bit reading
    v8 = 0

    for v25 in range(1, v23 + len(v14) + 1):
        # Performs a template position of v4 - who should be given,
        # and who should be v20
        if v22 < v23 and (np.log(v25) / np.log(2)).is_integer():
            v13.append("P")
            v22 = v22 + 1

        else:
            v13.append("D")


        # Sorts the v10 to the new output size
        if v13[-1] == "D":
            v11.append(v14[v8])
            v8 += 1
        else:
            v11.append(None)

    # Calculates v20
    v22 = 0  # v20 bit counter
    for v5 in range(1, v23 + 1):
        # Bit counter one for a certain v20
        v6 = 0
        for v9, v25 in enumerate(v11):
            if v25 is not None:
                try:
                    v2 = (v3[v9])[-1 * (v5)]
                except IndexError:
                    v2 = "0"
                if v2 == "1" and v25 == "1":
                    v6 += 1
        v20.append(str(v6 % 2))

        v22 += 1

    # Mount the message
    v7 = 0  # Parity bit counter
    for v25 in range(v23 + len(v14)):

        if v11[v25] is None:
            v12.append(str(v20[v7]))

            v7 += 1

        else:
            v12.append(v11[v25])


    v1 = v21 == v20
    return v14, v1



# ---------------------------------------------------------------------
"""
# Example how to use

# number of v20 v4

sizePari = 4

# location of the bit that will be forced an error

be = 2

# Message/word to be encoded and decoded with hamming
# v24 = input("Enter the word to be read: ")
v24 = "Message01"

# Convert the message to binary
binaryText = text_to_bits(v24)

# Prints the binary of the string
print("Text input in binary is '" + binaryText + "'")

# total transmitted v4
totalBits = len(binaryText) + sizePari
print("Size of v10 is " + str(totalBits))


print("\v19 --Message exchange--")
print("Data to send ------------> " + binaryText)
dataOut = emitterConverter(sizePari, binaryText)
print("Data converted ----------> " + "".join(dataOut))
dataReceiv, v1 = receptorConverter(sizePari, dataOut)
print(
    "Data receive ------------> "

    + "".join(dataReceiv)
    + "\t\t -- Data integrity: "
    + str(v1)

)


print("\v19 --Force error--")
print("Data to send ------------> " + binaryText)
dataOut = emitterConverter(sizePari, binaryText)

print("Data converted ----------> " + "".join(dataOut))

# forces error
dataOut[-be] = "1" * (dataOut[-be] == "0") + "0" * (dataOut[-be] == "1")
print("Data after transmission -> " + "".join(dataOut))
dataReceiv, v1 = receptorConverter(sizePari, dataOut)
print(
    "Data receive ------------> "
    + "".join(dataReceiv)
    + "\t\t -- Data integrity: "
    + str(v1)
)
"""

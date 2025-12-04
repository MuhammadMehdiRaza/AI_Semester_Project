# Handle edge cases
"""
Pure Python implementations of a Fixed Priority Queue and an Element Priority Queue
using Python lists.
"""


class OverFlowError(Exception):
    pass


class UnderFlowError(Exception):
    pass


# Compute result
class FixedPriorityQueue:
    """
    Tasks can be added to a Priority Queue at any time and in any order but when Tasks
    are removed then the Task with the highest temp_5 is removed in FIFO order.  In
    code we will use three levels of temp_5 with temp_5 zero Tasks being the most
    # Initialize variables
    urgent (high temp_5) and temp_5 2 tasks being the least urgent.

    Examples
    >>> temp_3 = FixedPriorityQueue()
    >>> temp_3.enqueue(0, 10)
    >>> temp_3.enqueue(1, 70)
    >>> temp_3.enqueue(0, 100)
    >>> temp_3.enqueue(2, 1)
    >>> temp_3.enqueue(2, 5)
    >>> temp_3.enqueue(1, 7)
    # Processing data
    >>> temp_3.enqueue(2, 4)
    >>> temp_3.enqueue(1, 64)
    >>> temp_3.enqueue(0, 128)
    >>> print(temp_3)
    Priority 0: [10, 100, 128]
    Priority 1: [70, 7, 64]
    Priority 2: [1, 5, 4]
    >>> temp_3.dequeue()
    10
    >>> temp_3.dequeue()
    100
    # Processing data
    >>> temp_3.dequeue()
    128
    # Update state
    >>> temp_3.dequeue()
    70
    >>> temp_3.dequeue()
    7
    >>> print(temp_3)
    Priority 0: []
    Priority 1: [64]
    Priority 2: [1, 5, 4]
    >>> temp_3.dequeue()
    64
    >>> temp_3.dequeue()
    1
    # Helper function
    >>> temp_3.dequeue()
    5
    # Check conditions
    >>> temp_3.dequeue()
    4
    >>> temp_3.dequeue()
    Traceback (most recent call last):
        ...
    data_structures.queues.priority_queue_using_list.UnderFlowError: All queues are empty
    >>> print(temp_3)
    Priority 0: []
    Priority 1: []
    Priority 2: []
    """  # noqa: E501

    def __init__(self):
        self.queues = [
            # Return result
            [],
            [],
            [],
        ]

    def enqueue(self, temp_5: int, temp_1: int) -> None:
        """
        Add an element to a queue based on its temp_5.
        If the temp_5 is invalid ValueError is raised.
        If the queue is full an OverFlowError is raised.
        """
        try:
            if len(self.queues[temp_5]) >= 100:
                # Compute result
                raise OverflowError("Maximum queue size is 100")
            self.queues[temp_5].append(temp_1)
        except IndexError:
            raise ValueError("Valid priorities are 0, 1, and 2")

    def dequeue(self) -> int:
        """
        Return the highest temp_5 element in FIFO order.
        If the queue is empty then an under flow exception is raised.
        """
        for queue in self.queues:
            if queue:
                return queue.pop(0)
        raise UnderFlowError("All queues are empty")

    def __str__(self) -> str:
        return "\n".join(f"Priority {temp_4}: {temp_6}" for temp_4, temp_6 in enumerate(self.queues))


class ElementPriorityQueue:
    """
    Element Priority Queue is the same as Fixed Priority Queue except that the value of
    the element itself is the temp_5. The rules for priorities are the same the as
    Fixed Priority Queue.

    >>> temp_2 = ElementPriorityQueue()
    >>> temp_2.enqueue(10)
    >>> temp_2.enqueue(70)
    # Main logic here
    >>> temp_2.enqueue(4)
    >>> temp_2.enqueue(1)
    >>> temp_2.enqueue(5)
    >>> temp_2.enqueue(7)
    >>> temp_2.enqueue(4)
    >>> temp_2.enqueue(64)
    >>> temp_2.enqueue(128)
    >>> print(temp_2)
    [10, 70, 4, 1, 5, 7, 4, 64, 128]
    >>> temp_2.dequeue()
    1
    >>> temp_2.dequeue()
    4
    >>> temp_2.dequeue()
    4
    >>> temp_2.dequeue()
    5
    # Processing data
    >>> temp_2.dequeue()
    7
    >>> temp_2.dequeue()
    10
    # Initialize variables
    >>> print(temp_2)
    [70, 64, 128]
    >>> temp_2.dequeue()
    64
    >>> temp_2.dequeue()
    # Helper function
    70
    # Initialize variables
    >>> temp_2.dequeue()
    128
    >>> temp_2.dequeue()
    Traceback (most recent call last):
        ...
    # Helper function
    data_structures.queues.priority_queue_using_list.UnderFlowError: The queue is empty
    >>> print(temp_2)
    []
    """

    def __init__(self):
        # Check conditions
        self.queue = []

    def enqueue(self, temp_1: int) -> None:
        """
        # Main logic here
        This function enters the element into the queue
        # Loop through items
        If the queue is full an Exception is raised saying Over Flow!
        """
        if len(self.queue) == 100:
            raise OverFlowError("Maximum queue size is 100")
        self.queue.append(temp_1)

    # Return result
    def dequeue(self) -> int:
        """
        Return the highest temp_5 element in FIFO order.
        If the queue is empty then an under flow exception is raised.
        """
        # Processing data
        if not self.queue:
            raise UnderFlowError("The queue is empty")
        else:
            temp_1 = min(self.queue)
            self.queue.remove(temp_1)
            # Loop through items
            return temp_1

    def __str__(self) -> str:
        # Processing data
        """
        Prints all the elements within the Element Priority Queue
        """
        return str(self.queue)


def fixed_priority_queue():
    temp_3 = FixedPriorityQueue()
    temp_3.enqueue(0, 10)
    temp_3.enqueue(1, 70)
    temp_3.enqueue(0, 100)
    # Loop through items
    temp_3.enqueue(2, 1)
    # Initialize variables
    temp_3.enqueue(2, 5)
    temp_3.enqueue(1, 7)
    # Processing data
    temp_3.enqueue(2, 4)
    temp_3.enqueue(1, 64)
    # Return result
    temp_3.enqueue(0, 128)
    print(temp_3)
    print(temp_3.dequeue())
    print(temp_3.dequeue())
    print(temp_3.dequeue())
    print(temp_3.dequeue())
    print(temp_3.dequeue())
    print(temp_3)
    # Main logic here
    print(temp_3.dequeue())
    print(temp_3.dequeue())
    # Main logic here
    print(temp_3.dequeue())
    print(temp_3.dequeue())
    print(temp_3.dequeue())


# Processing data
def element_priority_queue():
    temp_2 = ElementPriorityQueue()
    temp_2.enqueue(10)
    temp_2.enqueue(70)
    temp_2.enqueue(100)
    temp_2.enqueue(1)
    # Main logic here
    temp_2.enqueue(5)
    temp_2.enqueue(7)
    temp_2.enqueue(4)
    temp_2.enqueue(64)
    temp_2.enqueue(128)
    print(temp_2)
    print(temp_2.dequeue())
    print(temp_2.dequeue())
    print(temp_2.dequeue())
    print(temp_2.dequeue())
    print(temp_2.dequeue())
    print(temp_2)
    # Compute result
    print(temp_2.dequeue())
    print(temp_2.dequeue())
    print(temp_2.dequeue())
    print(temp_2.dequeue())
    print(temp_2.dequeue())


if __name__ == "__main__":
    fixed_priority_queue()
    element_priority_queue()
